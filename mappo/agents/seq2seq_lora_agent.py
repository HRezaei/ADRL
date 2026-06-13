import os

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.distributions import Categorical
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mappo.agents.llama_lora_agent import LlamaLoRAgent


class Seq2SeqLoRAgent(LlamaLoRAgent):
    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.algo = algo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        else:
            self.tokenizer.pad_token_id = int(self.tokenizer.pad_token_id)
        self.tokenizer.padding_side = "right"

        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        self.base_model.to(self.device)

        self.max_new_tokens = max_new_tokens

        if load_path is None:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)

    def _get_lora_target_modules(self):
        module_leaf_names = {name.split(".")[-1] for name, _ in self.base_model.named_modules()}
        if {"q_proj", "v_proj"}.issubset(module_leaf_names):
            return ["q_proj", "v_proj"]
        if {"q", "v"}.issubset(module_leaf_names):
            return ["q", "v"]
        raise ValueError("Could not infer LoRA target modules for the seq2seq model.")

    def _init_actor(self, lora_weights=None):
        if lora_weights is None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=self._get_lora_target_modules(),
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            model = get_peft_model(self.base_model, config)
            model.print_trainable_parameters()
            return model

        return PeftModel.from_pretrained(self.base_model, lora_weights).to(self.device)

    def _get_decoder_start_token_id(self):
        return int(self.tokenizer.pad_token_id)

    def _build_decoder_inputs(self, decoder_input_ids, decoder_attention_mask):
        decoder_start_tokens = torch.full(
            (decoder_input_ids.shape[0], 1),
            self._get_decoder_start_token_id(),
            dtype=decoder_input_ids.dtype,
            device=decoder_input_ids.device,
        )
        decoder_start_attention_mask = torch.ones(
            (decoder_attention_mask.shape[0], 1),
            dtype=decoder_attention_mask.dtype,
            device=decoder_attention_mask.device,
        )

        shifted_decoder_input_ids = torch.cat(
            [decoder_start_tokens, decoder_input_ids[:, :-1]],
            dim=1,
        )
        shifted_decoder_attention_mask = torch.cat(
            [decoder_start_attention_mask, decoder_attention_mask[:, :-1]],
            dim=1,
        )
        return shifted_decoder_input_ids, shifted_decoder_attention_mask

    def _project_critic_hidden(self, hidden_states):
        if hasattr(self.critic, "v_head"):
            return self.critic.v_head(hidden_states)

        x = self.critic.relu(self.critic.v_head_mlp1(hidden_states))
        x = self.critic.relu(self.critic.v_head_mlp2(x))
        return self.critic.v_head_mlp3(x)

    def get_actions(self, obs, ava, actions=None, greedy=False):
        prompts = obs.tolist()
        action_list = [act.split(",") for act in ava.tolist()]
        action_num_list = [len(ac) for ac in action_list]

        if actions is not None:
            action_ids = []
            for i in range(len(actions)):
                action_ids.append(action_list[i].index(actions[i]))
            action_ids = torch.tensor(action_ids).to(self.device)
        else:
            action_ids = None

        prompt_sequences = []
        action_sequences = []
        for prompt, candidates in zip(prompts, action_list):
            prompt_sequences += [prompt for _ in candidates]
            action_sequences += [action for action in candidates]

        enc = self.tokenizer(
            prompt_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        dec = self.tokenizer(
            action_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        decoder_input_ids_full = dec["input_ids"].to(self.device)
        decoder_attention_mask_full = dec["attention_mask"].to(self.device)
        decoder_lengths = decoder_attention_mask_full.sum(dim=1)
        if decoder_lengths.max() <= 0:
            raise ValueError("Action sequences must contain at least one token.")

        decoder_input_ids, decoder_attention_mask = self._build_decoder_inputs(
            decoder_input_ids_full,
            decoder_attention_mask_full,
        )
        decoder_target_ids = decoder_input_ids_full

        outputs = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        pi_log_softmax = torch.log_softmax(outputs.logits, dim=-1)

        action_logits = []
        action_token_list = []
        for i in range(decoder_target_ids.shape[0]):
            token_length = int(decoder_lengths[i].item())
            token_slice = decoder_target_ids[i, :token_length]
            action_token_list.append(token_slice)

            logit_slice = pi_log_softmax[i, :token_length, :]
            act_logit_seq = torch.gather(logit_slice, 1, token_slice[:, None]).squeeze(-1)
            action_word_length = max(len(action_sequences[i].split()), 1)
            action_logits.append(act_logit_seq.sum() / action_word_length)

        action_logits = torch.stack(action_logits)

        actions_out = []
        action_tokens = torch.ones(
            (len(action_num_list), self.max_new_tokens),
            dtype=torch.int64,
            device=self.device,
        ) * self.tokenizer.pad_token_id
        action_log_probs = []
        entropies = []

        for i in range(len(action_num_list)):
            start = sum(action_num_list[:i])
            end = sum(action_num_list[: i + 1])

            action_logits_i = action_logits[start:end]
            action_token_list_i = action_token_list[start:end]
            decoder_lengths_i = decoder_lengths[start:end]

            dist_i = Categorical(logits=action_logits_i)
            if action_ids is None:
                action_i_idx = action_logits_i.argmax() if greedy else dist_i.sample()
            else:
                action_i_idx = action_ids[i]

            actions_out.append(action_list[i][action_i_idx])
            action_token_i = action_token_list_i[action_i_idx]
            action_tokens[i, : decoder_lengths_i[action_i_idx]] = action_token_i
            action_log_probs.append(dist_i.log_prob(action_i_idx))
            entropies.append(dist_i.entropy())

        action_log_probs = torch.stack(action_log_probs)
        entropies = torch.stack(entropies)
        actions_out = np.array(actions_out, dtype=np.object_)
        return actions_out, action_tokens, action_log_probs, entropies

    def get_action_values(self, obs):
        return self.get_next_tppo_values(obs)

    def get_token_values(self, obs, actions):
        enc = self.tokenizer(
            obs.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        dec = self.tokenizer(
            actions.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        decoder_input_ids_full = dec["input_ids"].to(self.device)
        decoder_attention_mask_full = dec["attention_mask"].to(self.device)
        action_token_lengths = decoder_attention_mask_full.sum(dim=1)

        if action_token_lengths.max() > self.max_new_tokens:
            raise ValueError(
                f"The length of action tokens {action_token_lengths.max()} exceeds "
                f"the maximum length {self.max_new_tokens}."
            )

        decoder_input_ids, decoder_attention_mask = self._build_decoder_inputs(
            decoder_input_ids_full,
            decoder_attention_mask_full,
        )

        with self.actor.disable_adapter():
            with torch.no_grad():
                critic_outputs = self.critic.rwtranrsformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_states = critic_outputs.decoder_hidden_states[-1].float()
        values = self._project_critic_hidden(hidden_states)

        token_values = torch.zeros(
            (values.shape[0], self.max_new_tokens, values.shape[-1]),
            device=self.device,
            dtype=values.dtype,
        )
        for i in range(values.shape[0]):
            token_length = int(action_token_lengths[i].item())
            if token_length > 0:
                token_values[i, :token_length] = values[i, :token_length]

        return token_values

    def get_token_logits(self, obs, actions):
        enc = self.tokenizer(
            obs.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        dec = self.tokenizer(
            actions.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        decoder_input_ids_full = dec["input_ids"].to(self.device)
        decoder_attention_mask_full = dec["attention_mask"].to(self.device)
        action_token_lengths = decoder_attention_mask_full.sum(dim=1)

        if action_token_lengths.max() > self.max_new_tokens:
            raise ValueError(
                f"The length of action tokens {action_token_lengths.max()} exceeds "
                f"the maximum length {self.max_new_tokens}."
            )

        decoder_input_ids, decoder_attention_mask = self._build_decoder_inputs(
            decoder_input_ids_full,
            decoder_attention_mask_full,
        )

        with self.actor.disable_adapter():
            rho_outputs = self.actor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
            )

        pi_outputs = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

        rho_logits = torch.zeros(
            (pi_outputs.logits.shape[0], self.max_new_tokens, pi_outputs.logits.shape[-1]),
            device=self.device,
            dtype=pi_outputs.logits.dtype,
        )
        pi_logits = torch.zeros_like(rho_logits)
        for i in range(pi_outputs.logits.shape[0]):
            token_length = int(action_token_lengths[i].item())
            if token_length > 0:
                rho_logits[i, :token_length] = rho_outputs.logits[i, :token_length]
                pi_logits[i, :token_length] = pi_outputs.logits[i, :token_length]

        return pi_logits, rho_logits

    def get_next_tppo_values(self, obs):
        enc = self.tokenizer(
            obs.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        decoder_input_ids = torch.full(
            (input_ids.shape[0], 1),
            self._get_decoder_start_token_id(),
            dtype=torch.long,
            device=self.device,
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        with self.actor.disable_adapter():
            with torch.no_grad():
                critic_outputs = self.critic.rwtranrsformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_states = critic_outputs.decoder_hidden_states[-1].float()
        values = self._project_critic_hidden(hidden_states)
        return values[:, 0]

    def save(self, save_dir, episode):
        print("save model")
        exp_path = os.path.join(save_dir, "episode_{:04d}".format(episode))
        os.makedirs(exp_path, exist_ok=True)
        self.actor.save_pretrained(exp_path)

    def load(self, save_dir):
        print("load model on path: ", save_dir)
        self.actor = self._init_actor(save_dir).to(self.device)
        self.critic = self._init_critic().to(self.device)
