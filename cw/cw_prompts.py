


def cw_analyze_file_prompt(file_path, file_content):
    return 
    f"""
Analyze the following code file and provide a comprehensive summary:

File: {file_path}

For each file indicate whether it is a config file, a documenation file, a script file or a source file.
 A config file contains configuration information for the app, the build system, the test environement and does not
 generally contain core application source code. 

A source file contains application source code in a programming language. A script file contains scripts that help with
execution, testing, building and related facets.

For each file indicate the file type at the top. Subsequently, depending on the file type document them as follows:


For a file that has source code, please document:
1. The file's main purpose and functionality
2. Key functions, classes, or components defined
3. Dependencies on other modules in the project (imports, requires, includes)
4. Any important patterns, algorithms, or design decisions
5. Public APIs or interfaces exposed

For a file thats a config file, just provide a summary of its purpose. For a file thats a documenation file, provide 
a summary of the documentation that would be relevant to understanding the project's code, architecture and implemenation details.

FOr a file thats a script file, provide a summary of the purpose of the file. 

File content:
```
{file_content}
```

Provide a clear, structured summary that would help developers understand this file's role in the project.
"""

def cw_analyze_v0(file_path, file_content):
    return f"""Analyze the following code file and provide a comprehensive summary:

File: {file_path}

Please document:
1. The file's main purpose and functionality
2. Key functions, classes, or components defined
3. Dependencies on other modules in the project (imports, requires, includes)
4. Any important patterns, algorithms, or design decisions
5. Public APIs or interfaces exposed

Code content:
```
{file_content}
```

Provide a clear, structured summary that would help developers understand this file's role in the project."""