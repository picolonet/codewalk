
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from llm.oai_model import OaiModel
from llm.anthropic_model import AnthropicModel
from llm.llama_model import LlamaModel, create_llama4_scout_model
from llm.llm_model import LlmModel
from llm.lite_llm_model import LiteLlmModel
from dotenv import load_dotenv

class LlmRouter:
    def __init__(self):
        self.llm_model = None
        self.oai_model = None
        self.anthropic_model = None
        self.llama_model = None

    def openai(self):
        load_dotenv()
        if not self.oai_model:
            api_key = os.getenv("OPENAI_API_KEY")
            self.oai_model = OaiModel(model="gpt-4o", api_key=api_key)
        self.llm_model = self.oai_model
        return self.llm_model
    
    def anthropic(self):
        load_dotenv()
        if not self.anthropic_model:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.anthropic_model = AnthropicModel(api_key=api_key)
        self.llm_model = self.anthropic_model
        return self.llm_model

    def llama4(self):
        load_dotenv()
        if not self.llama_model:
            api_key = os.getenv("GROQ_API_KEY")
            self.llama_model = create_llama4_scout_model(api_key=api_key)
        self.llm_model = self.llama_model
        return self.llm_model

    def lite_llm(self):
        load_dotenv()
        api_key = os.environ.get("CODEWALKER_API_KEY")
        base_url = os.environ.get("CODEWALKER_BASE_URL")
        model_name = os.environ.get("CODEWALKER_MODEL_NAME") or "gpt-3.5-turbo"

        lite_llm = LiteLlmModel(
            model=model_name,
            base_url=base_url,
            api_key=api_key,  # Or set OPENAI_API_KEY environment variable
            temperature=0.7)
        self.llm_model = lite_llm
        return lite_llm

    def get(self):
        if not self.llm_model:
            return self.lite_llm()
        return self.llm_model
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently active model."""
        if not self.llm_model:
            return "lite_llm"
        return self.llm_model.model_name()
    
    def set_model(self, model_type: str):
        """Set the model type. Valid types: 'oai', 'claude', 'llama'."""
        model_type = model_type.lower()
        if model_type == "oai":
            return self.openai()
        elif model_type == "claude":
            return self.anthropic()
        elif model_type == "llama":
            return self.llama4()
        elif model_type == "litellm":
            return self.lite_llm()
        else:
            raise ValueError(f"Unknown model type: {model_type}. Valid types: 'oai', 'claude', 'llama'")

_global_llm_router = None

def llm_router() -> LlmRouter:
    global _global_llm_router
    if _global_llm_router is None:
        _global_llm_router = LlmRouter()
    return _global_llm_router
