
# CodeWalk Repo

Goal is to use agentic exploration for code understanding.


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
- A way to log stats: number of tokens (inp/out), number of LLM calls and cost for a given query or a task
  such as KB construction.
   - Create a tag and add stats to that tag. 
      - CwTask changes to aggregate stats.
      - command line to process query.
      - Log tool calls, kb vs non-kb.
   - TODO: Check stats and ensure that the operations tag is there. 
- Print final kb buider message as a panel and save it ?
- Aggregate stats across a CwTask for queries so we can benchmark. 
- change router to save model config changes to the codewalk.conf file. 
- move logging capability in lite_llm_model to base class. 
- load codewlk.conf settings into cli.py
- Test kb generation with Llama 4 and test 10M context window.
- Run kb generation for fastapi.
