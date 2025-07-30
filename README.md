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