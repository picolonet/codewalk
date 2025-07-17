from llm_model import llm, Message


class CodeWalker:
    def __init__(self, code_base_path: str):
        self.code_base_path = code_base_path

    def run_query(self, query: str):
        return llm.complete(messages=[Message(role="user", content=query)])


code_walker = CodeWalker(code_base_path=".")
