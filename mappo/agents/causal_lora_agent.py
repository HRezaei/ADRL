from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model

from mappo.agents.llama_lora_agent import LlamaLoRAgent


class CausalLoRAgent(LlamaLoRAgent):
    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.algo = algo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "right"
        self.tokenizer_add_bos = self.tokenizer_prepends_bos()

        if load_path is not None:
            model_name = load_path

        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        self.base_model.to(self.device)

        # Apply LoRA adapters to the base model for the actor
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        if load_path is None:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)
        else:
            self.load(load_path)

        #self.actor = get_peft_model(self.base_model, lora_config)

        self.max_new_tokens = max_new_tokens
        #self.critic = self._init_critic().to(self.device)

    def tokenizer_prepends_bos(self):
        if self.tokenizer.bos_token_id is None:
            return False

        test = "hello"
        out = self.tokenizer(test, add_special_tokens=True)["input_ids"]

        return len(out) > 0 and out[0] == self.tokenizer.bos_token_id

    def get_actions(self, obs, ava, actions=None, greedy=False):
        """
        Compute actions and value function predictions for the given inputs.
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

        sequences = []
        action_sequences = []
        for p, ac in zip(prompts, action_list):
            sequences += [p + " " + a for a in ac]
            action_sequences += [a for a in ac]  # for llama

        token_seq = self.tokenizer(sequences, return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].to(self.device)
        attn_mask = token_seq["attention_mask"].to(self.device)
        seq_token_lengths = attn_mask.sum(dim=1)

        outputs = self.actor(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        token_logits = outputs.logits[:, :-1, :]
        input_ids = input_ids[:, 1:]  # align logits and ids

        act_token_seq = self.tokenizer(action_sequences, return_tensors="pt", padding=True)
        act_attn_mask = act_token_seq["attention_mask"].to(self.device)
        act_token_lengths = act_attn_mask.sum(dim=1) - (1 if self.tokenizer_add_bos else 0)  # ignore the <bos> token

        actions, action_tokens, action_log_probs, entropies = self.sample_actions(input_ids,
                                                                                  token_logits,
                                                                                  seq_token_lengths,
                                                                                  act_token_lengths,
                                                                                  action_num_list,
                                                                                  action_list,
                                                                                  action_ids,
                                                                                  greedy)
        # print("selected action prob: ", action_log_probs.exp().mean())

        return actions, action_tokens, action_log_probs, entropies
