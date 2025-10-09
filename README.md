
# CodeWalk Repo

Goal is to use agentic exploration for code understanding. Based on the design of claude code and similar CLIs.


## Steps to run

```
python -m venv .venv
pip install -r requirements.txt

pip install -e . # to install codewalk as an editable dependency

cp .env.sample .env  # ensure .env has appropriate keys, at minimum langfuse (free api key). Create a langfuse tracing project and get a key.

cd repo_of_interest

codewalk
```

## Thoughts on agentic exploration of a codebase
1. Create a summary of knowns and unknowns for each file. 
  Pass in a running high level architecture (so far) of the project to help with this summary.

2. Hierarchically summarize each module and update the summary for each file according to that.

3. Update the high level project summary.


## Experimentation steps:

1. Go through each file and create a summary of the code. and each function. See how this looks.
2. Test and see what gitingest does.
3. Try claude-code on a larger repo to see how it handles token limits of 200k for creating claude.md
4. Try gemini-cli to see how it creates a codebase summary -- log all calls. 


### Notes:
1. Gitingest has code for directory traversal including markdown flattening of dir structure as a markdown
   fenced code block. ``` tree
2. xml should also work for dir structure.


### Langfuse:

langfuse-2.60.9
pip install "langfuse<3.0.0"

Needed to fix the error:
AttributeError: 'Langfuse' object has no attribute 'trace'


### Sample Queries

FastApi:  how does an api request get handled through various classes and functions in this project ?



### TODOs:
- Ask Claude and GPT-5 how to design such a system and what tool calls to use. Should AST be part of this ?  
- Clearly capture and record number of LLM calls and tokens (input/output). Thats a key metric to optimize. And the answer quality.
- Print tool calling, assistant messages clearly and in log so one can understand the trace easily similar to claude-trace.
- langfuse session tracking. observastions and multiple traces per trace. upgrade to new version and utilize llm-as-a-judge.
- move logging capability in lite_llm_model to base class. 
- load codewlk.conf settings into cli.py
- Test q: How does request processing work in the current project ?
