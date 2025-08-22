import os
import json
from typing import Optional, Dict, List, Any
from pathlib import Path
from cw.console_logger import console_logger
from cw.cw_task import CwTask
from cw.kb.kb_builder import KBBuilder
from cw.llm.llm_router import llm_router
from cw.util.cw_constants import CODEWALKER_KB_PREFIX
from cw.cw_prompts import cwkb_builder_task_system_prompt
from cw.util.cw_common import get_env

class KnowledgeBase:
    """Loads and queries a knowledge base created by KBBuilder."""
    
    def __init__(self, kb_root_path: str):
        """Initialize the knowledge base with the root path to .cw_kb directory.
        
        Args:
            kb_root_path: Path to the knowledge base root directory (usually .cw_kb/)
        """
        self.kb_root = Path(kb_root_path)
        if not self.kb_root.exists():
            raise FileNotFoundError(f"Knowledge base not found at: {kb_root_path}")
        if not self.kb_root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {kb_root_path}")
    
    def _get_module_path(self, module_path: str) -> Path:
        """
           Convert module path to Knowledge base filesystem path.
        """
        if module_path.startswith('/'):
            module_path = module_path[1:]
        if module_path.endswith('/'):
            module_path = module_path[:-1]
        
        return self.kb_root / module_path if module_path else self.kb_root
    
    def get_module_summary(self, module_path: str = "") -> Optional[str]:
        """Get the summary for a given module/folder path.
         Note: A module/folder is a folder that contains code files and might or mignt not
        be an architectural element.
        
        Args:
            module_path: Relative path to the module (empty string for root)
            
        Returns:
            Module summary content or None if not found
        """
        kb_module_path = self._get_module_path(module_path)
        summary_file = kb_module_path / "cw_module_summary.kb"
        
        if not summary_file.exists():
            return None
            
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return None
    
    def get_file_summary(self, file_path: str) -> Optional[str]:
        """Get the summary for a specific file.
        
        Args:
            file_path: Relative path to the file from repository root
            
        Returns:
            File summary content or None if not found
        """
        if file_path.startswith('/'):
            file_path = file_path[1:]
        
        # Split path into directory and filename
        path_parts = Path(file_path).parts
        if len(path_parts) == 1:
            # File is in root
            kb_module_path = self.kb_root
            filename = path_parts[0]
        else:
            # File is in subdirectory
            kb_module_path = self.kb_root / Path(*path_parts[:-1])
            filename = path_parts[-1]
        
        kb_filename = f"{filename}_codewalk.kb"
        summary_file = kb_module_path / kb_filename
        
        if not summary_file.exists():
            return None
            
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return None
    
    def get_module_metadata(self, module_path: str = "") -> Optional[Dict[str, Any]]:
        """Get metadata for a module/folder including its files and submodules.
        Note: A module/folder is a folder that contains code files and might or mignt not
        be an architectural element.
        
        Args:
            module_path: Relative path to the module/folder (empty string for root)
            
        Returns:
            Dictionary with 'kb_files' and 'submodules' lists, or None if not found
        """
        kb_module_path = self._get_module_path(module_path)
        metadata_file = kb_module_path / "cw_kb_module.json"
        
        if not metadata_file.exists():
            return None
            
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_modules(self, module_path: str = "") -> List[str]:
        """List all submodules/folders in a given module folder.
         Note: A module/folder is a folder that contains code files and might or mignt not
        be an architectural element.
        
        Args:
            module_path: Relative path to the module (empty string for root)
            
        Returns:
            List of submodule/folder names
        """
        metadata = self.get_module_metadata(module_path)
        return metadata.get('submodules', []) if metadata else []
    
    def list_files(self, module_path: str = "") -> List[str]:
        """List all files with summaries in a given module.
        
        Args:
            module_path: Relative path to the module (empty string for root)
            
        Returns:
            List of original filenames (without _codewalk.kb suffix)
        """
        metadata = self.get_module_metadata(module_path)
        if not metadata:
            return []
        
        # Remove _codewalk.kb suffix from filenames
        files = []
        for kb_file in metadata.get('kb_files', []):
            if kb_file.endswith('_codewalk.kb'):
                original_name = kb_file[:-12]  # Remove '_codewalk.kb'
                files.append(original_name)
        
        return files
    
    def module_exists(self, module_path: str) -> bool:
        """Check if a module exists in the knowledge base.
        
        Args:
            module_path: Relative path to the module
            
        Returns:
            True if module exists, False otherwise
        """
        kb_module_path = self._get_module_path(module_path)
        return kb_module_path.exists() and kb_module_path.is_dir()
    
    def file_has_summary(self, file_path: str) -> bool:
        """Check if a file has a summary in the knowledge base.
        
        Args:
            file_path: Relative path to the file from repository root
            
        Returns:
            True if file summary exists, False otherwise
        """
        return self.get_file_summary(file_path) is not None

    def does_kb_summary_exist(self) -> bool:
        """Check if the knowledge base summary exists.
        
        Returns:
            True if summary exists, False otherwise
        """
        return (self.kb_root / "kb_summary.md").exists()

    def write_kb_summary(self, summary: str) -> None:
        """Write a markdown formatted summary file for the knowledge base.
        This tool would be the final output that contains the knowledge base summary.
        
        Args:
            summary: The summary content to write to kb_summary.md
        """
        summary_file = self.kb_root / "kb_summary.md"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
        except Exception as e:
            raise IOError(f"Failed to write knowledge base summary: {str(e)}")

    def generate_kb_summary(self) -> None:
        """Generate a comprehensive summary of the project using CwTask."""

        
        prompt = """Generate a comprehensive codebase summary that includes:

1. **Project Overview**: What this project does, its main purpose and goals
2. **Core Architecture**: High-level architectural patterns, design principles, and system structure
3. **Key Components**: Main modules, classes, and their responsibilities
4. **Data Flow**: How data moves through the system, key data structures and transformations
5. **Control Flow**: Main execution paths, entry points, and interaction patterns
6. **Key Features**: Core functionality, APIs, and user-facing capabilities
7. **Dependencies**: External libraries, frameworks, and internal module dependencies
8. **Patterns & Conventions**: Code patterns, naming conventions, and architectural decisions

Use the available knowledge base query tools to explore the codebase systematically. Start with the root module summary, then explore key modules and files to build a comprehensive understanding.

Format the final summary as a well-structured markdown document that would serve as excellent documentation for developers new to the project.

IMPORTANT: When the given summary task is complete, write your output using a write_kb_summary tool call to save your output to the knowledge base.
"""


        kb_task = CwTask(
            user_query=prompt,
            code_base_path=str(self.kb_root.parent),
            task_system_prompt=cwkb_builder_task_system_prompt(get_env()),
            operation_tag="kb_builder"
        )
        
        tool_caller = kb_task.get_tool_caller()
        
        tool_caller.register_tool_from_function(self.get_module_summary)
        tool_caller.register_tool_from_function(self.get_file_summary) 
        tool_caller.register_tool_from_function(self.get_module_metadata)
        tool_caller.register_tool_from_function(self.list_modules)
        tool_caller.register_tool_from_function(self.list_files)
        tool_caller.register_tool_from_function(self.module_exists)
        tool_caller.register_tool_from_function(self.file_has_summary)
        tool_caller.register_tool_from_function(self.write_kb_summary)
        
        result = kb_task.run(prompt)
        

        return result
# Claude code prompt:
# "You are a code analyst tasked with creating comprehensive project documentation from a knowledge base.


def build_knowledge_base_with_summary(kb_root: str, max_workers: int = 4) -> Optional[KnowledgeBase]:
    """Build the knowledge base with a summary."""
    kb_builder = KBBuilder(llm_router.get())
    kb_parent = Path(kb_root).parent
    console_logger.log_text(f"Building knowledge base with {max_workers} workers")
    kb_builder.build_knowledge_base_topsort(root_path=str(kb_parent), num_parallel=max_workers)


    try:
        kb = KnowledgeBase(kb_root)
        
        # Generate summary if it doesn't exist
        if not kb.does_kb_summary_exist():
            console_logger.log_text("Knowledge base summary not found, generating...")
            kb.generate_kb_summary()
        else:
            console_logger.log_text("Knowledge base summary found, skipping generation...")
            
    except (FileNotFoundError, NotADirectoryError):
        console_logger.log_text(f"Knowledge base not found at {kb_root}. Please run knowledge base builder first.")
        kb = None
    return kb
