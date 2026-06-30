from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from mappo.agents.llama_full_agent import LlamaFullAgent
from mappo.utils.distributed import get_device


class CausalFullAgent(LlamaFullAgent):
    def __init__(self, model_name, max_new_tokens, algo, load_path=None):
        self.device = get_device()

        self.algo = algo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0  # keep pad different from eos
        # change padding side
        self.tokenizer.padding_side = "right"

        if load_path is not None:
            model_name = load_path

        model_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        self.base_model.to(self.device)

        self.max_new_tokens = max_new_tokens

        self.actor = self.base_model
        self.critic = self._init_critic().to(self.device)
