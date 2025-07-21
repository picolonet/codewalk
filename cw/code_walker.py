from llm_model import llm, Message
from tool_caller import ToolCaller, get_file_contents, list_directory, search_files


class CodeWalker:
    def __init__(self, code_base_path: str):
        self.code_base_path = code_base_path
        self.tool_caller = ToolCaller(llm)
        self.tool_caller.register_tool_from_function(get_file_contents)
        self.tool_caller.register_tool_from_function(list_directory)
        self.tool_caller.register_tool_from_function(search_files)

    def run_query(self, query: str):
        return self.tool_caller.full_completion_with_tools(messages=[Message(role="user", content=query)])

        


code_walker = CodeWalker(code_base_path=".")
