from cw.llm import oai_model
from cw.llm.oai_model import OaiModel
from llm_model import LlmModel
from dotenv import load_dotenv
import os


class LlmRouter:

    def __init__(self):
        pass

    def openai(self):
        load_dotenv()
        if not self.oai_model:
            api_key = os.getenv("OPENAI_API_KEY")
            self.oai_model = OaiModel(model = OaiModel.OAI_MODEL, api_key=api_key)
        self.llm_model = oai_model
        return self.llm_model
    
    def anthropic(self):
        pass

    def llama4(self):
        pass

    def lite_llm(self):
        pass