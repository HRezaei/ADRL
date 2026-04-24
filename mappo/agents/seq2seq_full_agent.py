import numpy as np
import torch
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from mappo.agents.llama_full_agent import LlamaFullAgent


class Seq2SeqFullAgent(LlamaFullAgent):
    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.algo = algo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # keep pad different from eos; default to 0 for safety if tokenizer has no pad
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        else:
            # normalize to an explicit int to avoid None further down the line
            self.tokenizer.pad_token_id = int(self.tokenizer.pad_token_id)
        # right padding to align sequences as in other agents
        self.tokenizer.padding_side = "right"

        if load_path is not None:
            model_name = load_path

        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        self.base_model.to(self.device)

        self.max_new_tokens = max_new_tokens

        # For full-scale training, actor is the base model
        self.actor = self.base_model
        # Initialize critic using the inherited method
        self.critic = self._init_critic().to(self.device)

    def get_actions(self, obs, ava, actions=None, greedy=False):
        """
        Compute actions and value function predictions for seq2seq models.

        Unlike causal LMs, the prompt is encoded on the encoder side and the
        candidate action is scored on the decoder side.
        """
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

        # Flatten prompt-action pairs
        prompt_sequences = []
        action_sequences = []
        for p, ac in zip(prompts, action_list):
            prompt_sequences += [p for _ in ac]
            action_sequences += [a for a in ac]

        # Encode prompts
        enc = self.tokenizer(
            prompt_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)

        # Tokenize actions for decoder scoring
        dec = self.tokenizer(
            action_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        decoder_input_ids = dec["input_ids"].to(self.device)
        decoder_attn_mask = dec["attention_mask"].to(self.device)

        # For seq2seq, score decoder tokens conditioned on the encoded prompt
        # We use teacher forcing: predict token t+1 from token t.
        decoder_labels = decoder_input_ids
        decoder_lengths = decoder_attn_mask.sum(dim=1)
        if decoder_lengths.max() <= 1:
            raise ValueError("Action sequences must contain at least one token.")

        # Build decoder inputs/targets
        dec_in = decoder_labels[:, :-1]
        dec_tgt = decoder_labels[:, 1:]

        outputs = self.actor(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_input_ids=dec_in,
            return_dict=True,
        )
        token_logits = outputs.logits  # [B, tgt_len-1, vocab]

        # Compute per-action log-probability
        pi_log_softmax = torch.log_softmax(token_logits, dim=-1)
        action_logits = []
        action_token_list = []
        for i in range(dec_tgt.shape[0]):
            tgt_len = int(decoder_lengths[i].item()) - 1
            token_slice = dec_tgt[i, :tgt_len]
            action_token_list.append(token_slice)

            logit_slice = pi_log_softmax[i, :tgt_len, :]
            act_logit_seq = torch.gather(logit_slice, 1, token_slice[:, None]).squeeze(-1)

            # word normalization consistent with the llama implementation
            action_word_length = len(action_sequences[i].split())
            action_log_softmax = act_logit_seq.sum() / max(action_word_length, 1)
            action_logits.append(action_log_softmax)

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
            end = sum(action_num_list[:i + 1])

            action_logits_i = action_logits[start:end]
            action_token_list_i = action_token_list[start:end]
            dec_lengths_i = decoder_lengths[start:end] - 1

            dist_i = Categorical(logits=action_logits_i)

            if action_ids is None:
                if greedy:
                    action_i_idx = action_logits_i.argmax()
                else:
                    action_i_idx = dist_i.sample()
            else:
                action_i_idx = action_ids[i]

            action_i = action_list[i][action_i_idx]
            actions_out.append(action_i)

            action_token_i = action_token_list_i[action_i_idx]
            action_tokens[i, :dec_lengths_i[action_i_idx]] = action_token_i

            action_log_probs.append(dist_i.log_prob(action_i_idx))
            entropies.append(dist_i.entropy())

        action_log_probs = torch.stack(action_log_probs)
        entropies = torch.stack(entropies)
        actions_out = np.array(actions_out, dtype=np.object_)
        return actions_out, action_tokens, action_log_probs, entropies

    def get_token_values(self, obs, actions):
        """
        Get token-level critic values for seq2seq models.

        The prompt lives on the encoder side, while the action tokens live on the
        decoder side. We therefore run the critic transformer with encoder inputs
        from `obs` and teacher-forced decoder inputs from the shifted action
        sequence, then project decoder hidden states through the critic head.
        """
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
        action_token_lengths = decoder_attention_mask_full.sum(dim=1) - 1

        if action_token_lengths.max() > self.max_new_tokens:
            raise ValueError(
                f"The length of action tokens {action_token_lengths.max()} exceeds "
                f"the maximum length {self.max_new_tokens}."
            )

        decoder_input_ids = decoder_input_ids_full[:, :-1]
        decoder_attention_mask = decoder_attention_mask_full[:, :-1]

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

        if hasattr(self.critic, "v_head"):
            values = self.critic.v_head(hidden_states)
        else:
            x = self.critic.relu(self.critic.v_head_mlp1(hidden_states))
            x = self.critic.relu(self.critic.v_head_mlp2(x))
            values = self.critic.v_head_mlp3(x)

        action_values = torch.zeros(
            (values.shape[0], self.max_new_tokens, values.shape[-1]),
            device=self.device,
            dtype=values.dtype,
        )
        for i in range(values.shape[0]):
            action_values[i, : action_token_lengths[i]] = values[i, : action_token_lengths[i]]

        return action_values

    def get_token_values(self, obs, actions):
        """
        Compute value predictions for decoder tokens for seq2seq models.
        Returns tensor shaped (batch, max_new_tokens, value_dim).
        """
        # Ensure inputs are lists
        obs_list = obs.tolist()
        actions_list = actions.tolist()

        # Encoder inputs
        enc = self.tokenizer(
            obs_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Decoder inputs (full action sequences)
        dec = self.tokenizer(
            actions_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        decoder_input_ids = dec["input_ids"].to(self.device)
        decoder_attn_mask = dec["attention_mask"].to(self.device)
        decoder_lengths = decoder_attn_mask.sum(dim=1)

        if decoder_lengths.max() <= 1:
            raise ValueError("Action sequences must contain at least one token.")

        # Run the seq2seq model to obtain decoder hidden states, then run critic heads
        with torch.no_grad():
            outputs = self.critic.rwtranrsformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # Prefer explicit decoder_hidden_states when available
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1]  # [B, dec_len, hidden]
        else:
            # Fallbacks for different model return structures
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # Some models pack hidden states differently; attempt to use last entry
                decoder_hidden = outputs.hidden_states[-1]
            else:
                # As a last resort, use last_hidden_state
                decoder_hidden = outputs.last_hidden_state

        # Decoder hidden includes the initial decoder token (e.g., BOS). Exclude it to align with targets.
        # decoder_hidden[:, 1:, :] corresponds to predictions for decoder tokens t=1..T-1
        decoder_hidden_nobos = decoder_hidden[:, 1:, :].float()

        # Run through critic MLPs to get per-token values
        x = self.critic.relu(self.critic.v_head_mlp1(decoder_hidden_nobos))
        x = self.critic.relu(self.critic.v_head_mlp2(x))
        values = self.critic.v_head_mlp3(x)

        # values shape: [B, tgt_len, value_dim] (value_dim often 1)
        batch = values.shape[0]
        val_dim = values.shape[-1]
        out = torch.zeros((batch, self.max_new_tokens, val_dim), device=self.device, dtype=values.dtype)

        tgt_lens = (decoder_lengths - 1).cpu().numpy()  # exclude bos
        for i in range(batch):
            l = min(int(tgt_lens[i]), self.max_new_tokens)
            if l > 0:
                out[i, :l] = values[i, :l]

        return out
