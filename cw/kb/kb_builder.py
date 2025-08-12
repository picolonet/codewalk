import os
import json
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from llm.llm_model import LlmModel
from llm.llm_common import Message


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
            '*.log', '*.tmp', 'logs', 'temp'
        }
        self.ignore_list.update(default_ignores)
    
    def _should_ignore(self, path: str, name: str) -> bool:
        """Check if a file or directory should be ignored."""
        # Check exact name matches
        if name in self.ignore_list:
            return True
            
        # Check pattern matches
        for pattern in self.ignore_list:
            if '*' in pattern:
                if pattern.startswith('*'):
                    if name.endswith(pattern[1:]):
                        return True
                elif pattern.endswith('*'):
                    if name.startswith(pattern[:-1]):
                        return True
            elif pattern in name:
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
            response = self.llm_model.complete(messages)
            
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