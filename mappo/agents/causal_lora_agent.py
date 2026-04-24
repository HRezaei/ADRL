from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model

from mappo.agents.causal_full_agent import CausalFullAgent


class CausalLoRAgent(CausalFullAgent):
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
        self.actor = get_peft_model(self.base_model, lora_config)

        self.max_new_tokens = max_new_tokens
        self.critic = self._init_critic().to(self.device)
