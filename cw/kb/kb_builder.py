import os
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from cw.cw_prompts import cwkb_builder_task_system_prompt
from cw.cw_task import CwTask
from cw.util.cw_common import get_env
from llm.llm_model import LlmModel
from llm.llm_common import Message
import fnmatch
from cw.cw_config import get_cw_config, CwConfig
from cw.util.data_logger import get_data_logger


class KBBuilder:
    """Builds a knowledge base for a source repository by analyzing code files and creating summaries."""
    
    def __init__(self, llm_model: LlmModel):
        self.llm_model = llm_model
        self.ignore_list: Set[str] = set()
        
    def _load_ignore_list(self, root_path: str) -> None:
        """Load ignore patterns from .cw_ignore file."""
        ignore_file = os.path.join(root_path, '.cw_ignore')
        self.ignore_list.clear()
        
        if os.path.exists(ignore_file):
            with open(ignore_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.ignore_list.add(line)
        
        # Add common defaults if not specified
        default_ignores = {
            '.git', '.cw_kb', '__pycache__', '.pyc', '.pyo', 
            'node_modules', '.vscode', '.idea', '.DS_Store',
            '*.log', '*.tmp', 'logs', 'temp', '.venv', '.cw'
        }
        self.ignore_list.update(default_ignores)
        print(f"Ignore list: {self.ignore_list}")
    
    def _should_ignore(self, path: str, name: str) -> bool:
        """Check if a file or directory should be ignored.

        Supports:
        - Exact name matches
        - Wildcard patterns (* for glob matching)
        - Regular expressions (patterns starting with 'regex:')
        - File type filtering (only allows source code and documentation files)
        """
        # Check exact name matches
        if name in self.ignore_list:
            return True

        # Get relative path for more comprehensive matching
        rel_path = os.path.relpath(path) if os.path.isabs(path) else path

        # Check pattern matches
        for pattern in self.ignore_list:
            # Handle regex patterns
            if pattern.startswith('regex:'):
                regex_pattern = pattern[6:]  # Remove 'regex:' prefix
                try:
                    if re.search(regex_pattern, name) or re.search(regex_pattern, rel_path):
                        return True
                except re.error:
                    # If regex is invalid, treat as literal string
                    if regex_pattern in name:
                        return True
                continue

            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
            # Handle wildcard patterns
            if '*' in pattern:
                if pattern.startswith('*'):
                    if name.endswith(pattern[1:]):
                        return True
                elif pattern.endswith('*'):
                    if name.startswith(pattern[:-1]):
                        return True
                else:
                    # Pattern has * in the middle - use simple glob matching

                    if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                        return True
            elif pattern in name or pattern in rel_path:
                return True

        # Only process files (not directories) with allowed extensions
        if os.path.isfile(path):
            # Define allowed extensions for source code and documentation
            allowed_extensions = {
                # Programming languages
                '.py', '.pyw', '.pyx', '.pyi',  # Python
                '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',  # JavaScript/TypeScript
                '.java', '.kt', '.kts',  # Java/Kotlin
                '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',  # C/C++
                '.cs',  # C#
                '.go',  # Go
                '.rs',  # Rust
                '.rb', '.rake',  # Ruby
                '.php',  # PHP
                '.swift',  # Swift
                '.m', '.mm',  # Objective-C
                '.scala', '.sc',  # Scala
                '.clj', '.cljs', '.cljc',  # Clojure
                '.lua',  # Lua
                '.pl', '.pm',  # Perl
                '.r', '.R',  # R
                '.dart',  # Dart
                '.ex', '.exs',  # Elixir
                '.erl', '.hrl',  # Erlang
                '.hs', '.lhs',  # Haskell
                '.ml', '.mli',  # OCaml
                '.nim',  # Nim
                '.v', '.vh',  # Verilog
                '.vhd', '.vhdl',  # VHDL

                # Web/Markup/Config
                '.html', '.htm', '.xhtml',  # HTML
                '.css', '.scss', '.sass', '.less',  # CSS
                '.xml', '.xsl', '.xsd',  # XML
                '.json', '.jsonc', '.json5',  # JSON
                '.yaml', '.yml',  # YAML
                '.toml',  # TOML
                '.ini', '.cfg', '.conf',  # Config files
                '.properties',  # Properties
                '.env.example', '.env.template',  # Example env files (not actual .env)

                # Documentation
                '.md', '.markdown', '.rst', '.txt', '.adoc', '.asciidoc',  # Documentation

                # Shell/Scripts
                '.sh', '.bash', '.zsh', '.fish', '.ksh',  # Shell scripts
                '.bat', '.cmd', '.ps1',  # Windows scripts

                # Build/Project files
                '.gradle', '.maven', '.sbt',  # Build tools
                '.cmake', '.make', '.mk',  # Make/CMake
                'Makefile', 'makefile', 'GNUmakefile',  # Makefiles (no extension)
                'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',  # Docker

                # Data query/config
                '.sql', '.hql', '.graphql', '.gql',  # Query languages

                # Other
                '.proto',  # Protocol Buffers
                '.thrift',  # Thrift
                '.tf',  # Terraform
            }

            # Special case: files without extensions that should be processed
            no_extension_allowed = {
                'Makefile', 'makefile', 'GNUmakefile',
                'Dockerfile', 'Jenkinsfile', 'Vagrantfile',
                'Rakefile', 'Gemfile', 'Podfile',
                'LICENSE', 'README', 'CHANGELOG', 'CONTRIBUTING',
                'AUTHORS', 'NOTICE', 'TODO',
            }

            # Check if file has no extension but is in allowed list
            if '.' not in name or name.startswith('.'):
                if name in no_extension_allowed:
                    return False
                # Ignore files without extension that aren't in the allowed list
                return True

            # Get file extension
            _, ext = os.path.splitext(name)
            ext_lower = ext.lower()

            # Ignore binary and non-source files
            if ext_lower not in allowed_extensions:
                # Additional check for files that might be text but not in our list
                # We'll be conservative and ignore them
                return True


        return False
    
    def _create_file_summary(self, file_path: str, relative_path: str) -> str:
        """Generate a summary of a file using the LLM."""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return "Empty file."
                
            prompt = f"""Analyze the following code file and provide a comprehensive summary:

File: {relative_path}

Content:
{content}

Please provide:
1. A brief description of the file's main purpose
2. Key functions/classes and their responsibilities
3. Dependencies on other modules in the project
4. External libraries used
5. Any notable patterns or architectural decisions

Format the response as a structured analysis that would be useful for code understanding and navigation."""

            messages = [Message(role="user", content=prompt)]
            response = self.llm_model.complete(messages, trace_name="file_summary")
            data_logger = get_data_logger()
            data_logger.log_stats(self.llm_model.get_model_name(), prompt_tokens=response.get_prompt_tokens(),
                 completion_tokens=response.get_completion_tokens(), latency_seconds=response.get_latency_seconds(),
                  operation="kb_builder")
            
            return response.content or "Unable to generate summary."
            
        except Exception as e:
            return f"Error analyzing file: {str(e)}"
    
    def _create_module_summary(self, module_path: str, kb_files: List[str], submodules: List[str]) -> str:
        """Generate a summary of an entire module using its KB files."""
        try:
            # Read all .kb files in the module
            kb_contents = []
            for kb_file in kb_files:
                kb_file_path = os.path.join(module_path, kb_file)
                if os.path.exists(kb_file_path):
                    with open(kb_file_path, 'r', encoding='utf-8') as f:
                        kb_contents.append(f"=== {kb_file} ===\n{f.read()}")
            
            if not kb_contents:
                return "No knowledge base files found in this module."
            
            combined_content = "\n\n".join(kb_contents)
            
            prompt = f"""Analyze the following knowledge base files from a software module and create a comprehensive module summary:

Module: {os.path.basename(module_path)}
Submodules: {', '.join(submodules) if submodules else 'None'}

Knowledge Base Files:
{combined_content}

Please provide:
1. Overall purpose and functionality of this module
2. Key components and their relationships
3. Main interfaces and APIs exposed
4. Dependencies on other modules
5. Architecture patterns used
6. How this module fits into the larger system

Create a rich summary that would help an LLM understand this module's role in deep code analysis tasks."""

            messages = [Message(role="user", content=prompt)]
            response = self.llm_model.complete(messages)
            # TODO: Record the number of tokens used for this prompt and response.
            
            return response.content or "Unable to generate module summary."
            
        except Exception as e:
            return f"Error creating module summary: {str(e)}"
    
    def _process_directory(self, dir_path: str, kb_root: str, relative_path: str = "") -> Dict[str, Any]:
        """Process a directory and create knowledge base files."""
        kb_files = []
        submodules = []
        
        # Get all items in directory
        try:
            items = os.listdir(dir_path)
        except PermissionError:
            return {"kb_files": [], "submodules": []}
        
        # Create corresponding KB directory
        kb_dir_path = os.path.join(kb_root, relative_path) if relative_path else kb_root
        os.makedirs(kb_dir_path, exist_ok=True)
        
        # Process files first
        for item in sorted(items):
            item_path = os.path.join(dir_path, item)
            
            if self._should_ignore(item_path, item):
                continue
                
            if os.path.isfile(item_path):
                # Process file - create .kb summary
                file_relative_path = os.path.join(relative_path, item) if relative_path else item
                kb_filename = f"{item}_codewalk.kb"
                kb_file_path = os.path.join(kb_dir_path, kb_filename)
                
                print(f"Processing file: {file_relative_path}")
                summary = self._create_file_summary(item_path, file_relative_path)
                
                with open(kb_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                kb_files.append(kb_filename)
        
        # Process subdirectories
        for item in sorted(items):
            item_path = os.path.join(dir_path, item)
            
            if self._should_ignore(item_path, item):
                continue
                
            if os.path.isdir(item_path):
                subdir_relative_path = os.path.join(relative_path, item) if relative_path else item
                print(f"Processing directory: {subdir_relative_path}")
                
                # Recursively process subdirectory
                self._process_directory(item_path, kb_root, subdir_relative_path)
                submodules.append(item)
        
        # Create module metadata file
        module_info = {
            "kb_files": kb_files,
            "submodules": submodules
        }
        
        module_json_path = os.path.join(kb_dir_path, "cw_kb_module.json")
        with open(module_json_path, 'w', encoding='utf-8') as f:
            json.dump(module_info, f, indent=2)
        
        # Create module summary
        if kb_files or submodules:
            print(f"Creating module summary for: {relative_path or 'root'}")
            module_summary = self._create_module_summary(kb_dir_path, kb_files, submodules)
            
            summary_path = os.path.join(kb_dir_path, "cw_module_summary.kb")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(module_summary)
        
        return module_info
    
    def _create_file_summary_parallel(self, file_info: Dict[str, str]) -> Dict[str, str]:
        """Helper function to create file summary for parallel execution."""
        item_path = file_info['item_path']
        file_relative_path = file_info['file_relative_path']
        kb_file_path = file_info['kb_file_path']
        kb_filename = file_info['kb_filename']
        
        print(f"Processing file: {file_relative_path}")
        summary = self._create_file_summary(item_path, file_relative_path)
        
        with open(kb_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        return {
            'kb_filename': kb_filename,
            'status': 'completed'
        }
    
    def _process_directory_parallel(self, dir_path: str, kb_root: str, relative_path: str = "", max_workers: int = 4) -> Dict[str, Any]:
        """Process a directory and create knowledge base files using parallel file processing."""
        kb_files = []
        submodules = []
        
        # Get all items in directory
        try:
            items = os.listdir(dir_path)
        except PermissionError:
            return {"kb_files": [], "submodules": []}
        
        # Create corresponding KB directory
        kb_dir_path = os.path.join(kb_root, relative_path) if relative_path else kb_root
        os.makedirs(kb_dir_path, exist_ok=True)
        
        # Collect all files to process
        file_tasks = []
        for item in sorted(items):
            item_path = os.path.join(dir_path, item)
            
            if self._should_ignore(item_path, item):
                continue
                
            if os.path.isfile(item_path):
                # Prepare file processing info
                file_relative_path = os.path.join(relative_path, item) if relative_path else item
                kb_filename = f"{item}_codewalk.kb"
                kb_file_path = os.path.join(kb_dir_path, kb_filename)
                
                file_tasks.append({
                    'item_path': item_path,
                    'file_relative_path': file_relative_path,
                    'kb_file_path': kb_file_path,
                    'kb_filename': kb_filename
                })
        
        # Process files in parallel
        if file_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all file processing tasks
                future_to_task = {
                    executor.submit(self._create_file_summary_parallel, task): task 
                    for task in file_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        kb_files.append(result['kb_filename'])
                    except Exception as e:
                        task = future_to_task[future]
                        print(f"Error processing file {task['file_relative_path']}: {str(e)}")
        
        # Process subdirectories (still sequential for now to avoid too much parallelism)
        for item in sorted(items):
            item_path = os.path.join(dir_path, item)
            
            if self._should_ignore(item_path, item):
                continue
                
            if os.path.isdir(item_path):
                subdir_relative_path = os.path.join(relative_path, item) if relative_path else item
                print(f"Processing directory: {subdir_relative_path}")
                
                # Recursively process subdirectory with parallel processing
                self._process_directory_parallel(item_path, kb_root, subdir_relative_path, max_workers)
                submodules.append(item)
        
        # Create module metadata file
        module_info = {
            "kb_files": kb_files,
            "submodules": submodules
        }
        
        module_json_path = os.path.join(kb_dir_path, "cw_kb_module.json")
        with open(module_json_path, 'w', encoding='utf-8') as f:
            json.dump(module_info, f, indent=2)
        
        # Create module summary
        if kb_files or submodules:
            print(f"Creating module summary for: {relative_path or 'root'}")
            module_summary = self._create_module_summary(kb_dir_path, kb_files, submodules)
            
            summary_path = os.path.join(kb_dir_path, "cw_module_summary.kb")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(module_summary)
        
        return module_info
    
    def build_knowledge_base(self, root_path: str) -> None:
        """Build knowledge base for the entire repository."""
        print(f"Building knowledge base for: {root_path}")
        
        # Load ignore patterns
        self._load_ignore_list(root_path)
        
        # Create .cw_kb directory at root
        kb_root = os.path.join(root_path, ".cw_kb")
        os.makedirs(kb_root, exist_ok=True)
        
        # Process the entire directory structure
        self._process_directory(root_path, kb_root)
        
        print(f"Knowledge base built successfully in: {kb_root}")
    
    def build_knowledge_base_parallel(self, root_path: str, max_workers: int = 4) -> None:
        """Build knowledge base for the entire repository using parallel file processing.
        
        Args:
            root_path: Path to the repository root
            max_workers: Maximum number of parallel workers for file processing (default: 4)
        """
        print(f"Building knowledge base (parallel, {max_workers} workers) for: {root_path}")
        
        # Load ignore patterns
        self._load_ignore_list(root_path)
        
        # Create .cw_kb directory at root
        kb_root = os.path.join(root_path, ".cw_kb")
        os.makedirs(kb_root, exist_ok=True)
        
        # Process the entire directory structure with parallel file processing
        self._process_directory_parallel(root_path, kb_root, max_workers=max_workers)
        
        print(f"Knowledge base built successfully in: {kb_root}")
    
    def _build_dependency_graph(self, root_path: str) -> Tuple[Dict[str, int], Dict[str, Set[str]], Dict[str, Dict[str, Any]]]:
        """Build a directed graph of file/directory dependencies and perform topological sort.
        
        Returns:
            - topo_order: Dict mapping node path to topological sort order (0 = no dependencies)
            - dependents: Dict mapping node path to set of paths that depend on it
            - node_info: Dict mapping node path to metadata (type, kb_path, etc.)
        """
        graph = defaultdict(set)  # node -> set of dependencies
        dependents = defaultdict(set)  # node -> set of dependents
        node_info = {}  # node -> metadata
        all_nodes = set()
        
        # Walk the entire directory structure
        for root, dirs, files in os.walk(root_path):
            # Filter out ignored directories and files
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d), d)]
            files = [f for f in files if not self._should_ignore(os.path.join(root, f), f)]
            
            relative_root = os.path.relpath(root, root_path)
            if relative_root == '.':
                relative_root = ''
            
            # Add files (leaf nodes with no dependencies)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                relative_file_path = os.path.join(relative_root, file_name) if relative_root else file_name
                
                all_nodes.add(relative_file_path)
                node_info[relative_file_path] = {
                    'type': 'file',
                    'absolute_path': file_path,
                    'parent_dir': relative_root,
                    'kb_filename': f"{file_name}_codewalk.kb"
                }
            
            # Add directory node
            if relative_root or files or dirs:  # Don't add empty root
                all_nodes.add(relative_root or 'root')
                
                # Directory dependencies: depends on all contained files and subdirectories
                dir_dependencies = set()
                
                # Add file dependencies
                for file_name in files:
                    relative_file_path = os.path.join(relative_root, file_name) if relative_root else file_name
                    dir_dependencies.add(relative_file_path)
                
                # Add subdirectory dependencies
                for dir_name in dirs:
                    subdir_relative = os.path.join(relative_root, dir_name) if relative_root else dir_name
                    dir_dependencies.add(subdir_relative)
                
                dir_key = relative_root or 'root'
                graph[dir_key] = dir_dependencies
                
                # Update dependents mapping
                for dep in dir_dependencies:
                    dependents[dep].add(dir_key)
                
                node_info[dir_key] = {
                    'type': 'directory',
                    'absolute_path': root,
                    'files': files,
                    'subdirs': dirs,
                    'kb_files': [f"{f}_codewalk.kb" for f in files]
                }
        
        # Perform topological sort
        topo_order = {}
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for node in all_nodes:
            in_degree[node] = len(graph[node])
        
        # Kahn's algorithm for topological sorting
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        order = 0
        
        while queue:
            current_level_nodes = list(queue)
            queue.clear()
            
            # All nodes at the same level get the same order (can be processed in parallel)
            for node in current_level_nodes:
                topo_order[node] = order
                
                # Update in-degrees of dependents
                for dependent in dependents[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            order += 1
        
        return topo_order, dependents, node_info
    
    def _process_file_node(self, node_path: str, node_info: Dict[str, Any], kb_root: str) -> str:
        """Process a single file node and create its .kb summary."""
        absolute_path = node_info['absolute_path']
        parent_dir = node_info['parent_dir']
        kb_filename = node_info['kb_filename']
        
        # Create KB directory path
        kb_dir_path = os.path.join(kb_root, parent_dir) if parent_dir else kb_root
        os.makedirs(kb_dir_path, exist_ok=True)
        
        # Generate file summary
        print(f"Processing file: {node_path}")
        # Check if KB file already exists and is non-empty
        kb_file_path = os.path.join(kb_dir_path, kb_filename)

        if os.path.exists(kb_file_path) and os.path.getsize(kb_file_path) > 0:
            print(f"KB file already exists and is non-empty, skipping: {kb_filename}")
            return kb_filename
        summary = self._create_file_summary(absolute_path, node_path)
        
        # Write KB file
        # kb_file_path = os.path.join(kb_dir_path, kb_filename)
        with open(kb_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return kb_filename
    
    def _process_directory_node(self, node_path: str, node_info: Dict[str, Any], kb_root: str) -> None:
        """Process a directory node and create its module summary and metadata."""
        absolute_path = node_info['absolute_path']
        files = node_info['files']
        subdirs = node_info['subdirs']
        kb_files = node_info['kb_files']
        
        # Determine KB directory path
        if node_path == 'root':
            kb_dir_path = kb_root
            relative_path = ''
        else:
            kb_dir_path = os.path.join(kb_root, node_path)
            relative_path = node_path
        
        os.makedirs(kb_dir_path, exist_ok=True)
        
        # Create module metadata file
        module_info = {
            "kb_files": kb_files,
            "submodules": subdirs
        }
        
        module_json_path = os.path.join(kb_dir_path, "cw_kb_module.json")
        with open(module_json_path, 'w', encoding='utf-8') as f:
            json.dump(module_info, f, indent=2)
        
        # Create module summary if there are files or subdirectories
        if kb_files or subdirs:
            print(f"Creating module summary for: {relative_path or 'root'}")
            module_summary = self._create_module_summary(kb_dir_path, kb_files, subdirs)
            
            summary_path = os.path.join(kb_dir_path, "cw_module_summary.kb")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(module_summary)
    
    def build_knowledge_base_topsort(self, root_path: str, num_parallel: int = 4) -> None:
        """Build knowledge base using topological sort for maximum parallelism.
        
        Args:
            root_path: Path to the repository root
            num_parallel: Maximum number of parallel processing tasks
        """
        print(f"Building knowledge base (topological sort, {num_parallel} parallel) for: {root_path}")
        
        # Load ignore patterns
        self._load_ignore_list(root_path)
        
        # Create .cw_kb directory at root
        cw_kb_subdir = get_cw_config().get(CwConfig.KB_DIR_KEY, CwConfig.KB_DIR_DEFAULT)
        kb_root = os.path.join(root_path, cw_kb_subdir)
        os.makedirs(kb_root, exist_ok=True)
        
        # Build dependency graph and get topological order
        topo_order, dependents, node_info = self._build_dependency_graph(root_path)
        
        if not topo_order:
            print("No files or directories to process.")
            return
        
        print(f"Found {len(topo_order)} nodes to process")
        
        # Group nodes by their topological order
        order_groups = defaultdict(list)
        for node_path, order in topo_order.items():
            order_groups[order].append(node_path)
        
        processed_nodes = set()
        
        # Process nodes level by level
        for order in sorted(order_groups.keys()):
            nodes_at_level = order_groups[order]
            print(f"Processing level {order} with {len(nodes_at_level)} nodes")
            
            # Separate files and directories for different processing
            file_nodes = [node for node in nodes_at_level if node_info[node]['type'] == 'file']
            dir_nodes = [node for node in nodes_at_level if node_info[node]['type'] == 'directory']
            
            # Process files in parallel
            if file_nodes:
                with ThreadPoolExecutor(max_workers=num_parallel) as executor:
                    future_to_node = {
                        executor.submit(self._process_file_node, node, node_info[node], kb_root): node
                        for node in file_nodes
                    }
                    
                    for future in as_completed(future_to_node):
                        node = future_to_node[future]
                        try:
                            future.result()
                            processed_nodes.add(node)
                        except Exception as e:
                            print(f"Error processing file node {node}: {str(e)}")
            
            # Process directories sequentially (they're usually quick and need to be done after files)
            for node in dir_nodes:
                try:
                    self._process_directory_node(node, node_info[node], kb_root)
                    processed_nodes.add(node)
                except Exception as e:
                    print(f"Error processing directory node {node}: {str(e)}")
        
        print(f"Knowledge base built successfully in: {kb_root}")
        print(f"Processed {len(processed_nodes)} out of {len(topo_order)} nodes")


# Topsort implementation notes: 
# I've successfully implemented the topologically sorted version of the build_knowledge_base() function. Here's what the new build_knowledge_base_topsort() function does:
#   Key Features:

#   1. Dependency Graph Construction (_build_dependency_graph):
#     - Traverses entire directory structure using os.walk()
#     - Files are leaf nodes with no dependencies (topsort order 0)
#     - Directories depend on all contained files and subdirectories
#     - Uses Kahn's algorithm for topological sorting
#   2. Maximum Parallelism:
#     - Groups nodes by topological order level
#     - Processes all nodes at the same level in parallel
#     - Files at level 0 can all be processed simultaneously
#     - Directories are processed after their dependencies are complete
#   3. Efficient Processing:
#     - Separates file processing (parallel) from directory processing (sequential)
#     - Uses ThreadPoolExecutor with configurable num_parallel parameter
#     - Maintains dependency constraints while maximizing parallel execution

#   Function Signature:

#   def build_knowledge_base_topsort(self, root_path: str, num_parallel: int = 4) -> None

#   The implementation ensures:
#   - Files are processed first (topsort value 0) in parallel batches
#   - Directories are processed only after all their contained files/subdirectories are complete
#   - Maximum parallelism within dependency constraints
#   - Error handling for individual nodes without stopping the entire process

#   This approach should significantly improve performance compared to the original sequential traversal, especially for large codebases with many files that can be processed independently.