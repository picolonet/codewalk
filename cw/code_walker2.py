import os
import fnmatch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from cw import cw_config
from cw.cw_task import CwTask
from cw.util.cw_common import get_env
from cw.llm.llm_router import llm_router
from cw.llm.llm_model import Message
from tool_caller import ToolCaller, get_file_contents, list_directory, search_files
from console_logger import console_logger
from cw_prompts import cw_analyze_file_prompt, cwcc_system_prompt, cwkb_builder_task_system_prompt
from cw.util.cw_constants import CODEWALKER_KB_PREFIX

# Load knowledge base. Run generate summary for kb if it doesn't exist.
# Given a query, run using CwTask and knowledge base. 
#
class CodeWalker2:
    def __init__(self, code_base_path: str):
        self.code_base_path = code_base_path
        # Initialize knowledge base
        kb_path = os.path.join(self.code_base_path, CODEWALKER_KB_PREFIX)
        self.kb_enabled = cw_config.get_cw_config().get(cw_config.CwConfig.KB_ENABLED_KEY, cw_config.CwConfig.KB_ENABLED_DEFAULT) == cw_config.CwConfig.KB_ENABLED_VALUE
        
        # Import KnowledgeBase here to avoid circular imports
        from cw.kb.knowledge_base import KnowledgeBase
        
        try:
            self.kb = KnowledgeBase(kb_path)
            
            # Generate summary if it doesn't exist
            if not self.kb.does_kb_summary_exist():
                console_logger.log_text("Knowledge base summary not found, generating...")
                self.kb.generate_kb_summary()
            else:
                console_logger.log_text("Knowledge base summary found, skipping generation...")
                
        except (FileNotFoundError, NotADirectoryError):
            console_logger.log_text(f"Knowledge base not found at {kb_path}. Please run knowledge base builder first.")
            self.kb = None

    def run_query(self, query: str, operation_tag: Optional[str] = None):
        query_task = CwTask(
            user_query=query,
            code_base_path=self.code_base_path,
            task_system_prompt=cwcc_system_prompt(get_env()),
            operation_tag=operation_tag
        )
        tool_caller = query_task.get_tool_caller()
        if (self.kb_enabled):
            tool_caller.register_tool_from_function(self.kb.get_module_summary)
            tool_caller.register_tool_from_function(self.kb.get_file_summary) 
            tool_caller.register_tool_from_function(self.kb.get_module_metadata)
            tool_caller.register_tool_from_function(self.kb.list_modules)
            tool_caller.register_tool_from_function(self.kb.list_files)
            tool_caller.register_tool_from_function(self.kb.module_exists)
            tool_caller.register_tool_from_function(self.kb.file_has_summary)
        else:
            console_logger.log_text("Not registering knowledge base tools.")
        #tool_caller.register_tool_from_function(self.kb.write_kb_summary)
        return query_task.run(query)
    
    def _load_ignore_patterns(self, root_path: str):
        """Load ignore patterns from .cw_ignore file."""
        ignore_file = os.path.join(root_path, '.cw_ignore')
        ignore_patterns = []
        
        if os.path.exists(ignore_file):
            with open(ignore_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        ignore_patterns.append(line)
        
        # Always ignore the .cw_kb directory itself
        ignore_patterns.append('.cw_kb')
        ignore_patterns.append('.cw_kb/*')
        ignore_patterns.append('.cw_ignore')
        
        return ignore_patterns
    
    def _should_ignore(self, path: str, ignore_patterns: list) -> bool:
        """Check if a path should be ignored based on patterns."""
        rel_path = os.path.relpath(path, self.code_base_path)
        
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False
    
    def _create_kb_directory(self, source_path: str, kb_root: str):
        """Create corresponding directory structure in kb_root."""
        rel_path = os.path.relpath(source_path, self.code_base_path)
        kb_path = os.path.join(kb_root, rel_path)
        
        if not os.path.exists(kb_path):
            os.makedirs(kb_path, exist_ok=True)
        
        return kb_path
    
    def _generate_file_summary(self, file_path: str) -> str:
        """Generate a summary of the file using LLM."""
        llm_model = llm_router.get()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except (UnicodeDecodeError, IOError) as e:
            err = f"Unable to process file {file_path} - binary or unreadable file"
            print(err)
            print(f"Exception: {e}")
            return err
        
        prompt = cw_analyze_file_prompt(file_path, file_content)
        #print("final prompt:", prompt)

        try:
            response = llm_model.complete([Message(role="user", content=prompt)])
            return response.content or "Failed to generate summary"
        except Exception as e:
            return f"Error generating summary for {file_path}: {str(e)}"
    
    def run_codewalk(self, kb_basedir=None, code_base_path=None):
        """Run a code walk, file by file, building the knowledge base."""
        if kb_basedir is None:
            kb_basedir = os.path.join(self.code_base_path, CODEWALKER_KB_PREFIX.rstrip('/'))
        
        # Ensure kb_basedir exists
        os.makedirs(kb_basedir, exist_ok=True)

        # Load ignore patterns
        ignore_patterns = self._load_ignore_patterns(self.code_base_path)
        
        if code_base_path:
            self.code_base_path = code_base_path
    
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(self.code_base_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d), ignore_patterns)]
            
            # Skip if current directory should be ignored
            if self._should_ignore(root, ignore_patterns):
                continue
            
            # Create corresponding directory in kb structure
            kb_dir = self._create_kb_directory(root, kb_basedir)
            
            # Process each file
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip ignored files
                if self._should_ignore(file_path, ignore_patterns):
                    continue
                
                print(f"Processing: {file_path}")
                console_logger.update_status(file)
                
                # Generate summary using LLM
                summary = self._generate_file_summary(file_path)
                
                # Create output filename
                base_name = os.path.splitext(file)[0]
                kb_filename = f"{base_name}_codewalk.kb"
                kb_file_path = os.path.join(kb_dir, kb_filename)
                
                # Write summary to kb file
                with open(kb_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# CodeWalk Summary for {file_path}\n\n")
                    f.write(f"Original file: {file_path}\n")
                    f.write(f"Generated on: {os.path.getctime(file_path)}\n\n")
                    f.write("## Summary\n\n")
                    f.write(summary)
                
                print(f"Created: {kb_file_path}")
        
        print(f"CodeWalk completed. Knowledge base created in: {kb_basedir}")
        
    async def _generate_file_summary_async(self, file_path: str) -> str:
        """Generate a summary of the file using LLM asynchronously."""
        llm_model = llm_router.get()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except (UnicodeDecodeError, IOError) as e:
            err = f"Unable to process file {file_path} - binary or unreadable file"
            print(err)
            print(f"Exception: {e}")
            return err
        
        prompt = cw_analyze_file_prompt(file_path, file_content)

        try:
            response = await llm_model.async_complete([Message(role="user", content=prompt)])
            return response.content or "Failed to generate summary"
        except Exception as e:
            return f"Error generating summary for {file_path}: {str(e)}"

    async def _process_file_batch(self, file_batch, kb_dir):
        """Process a batch of files in parallel."""
        tasks = []
        
        for file_info in file_batch:
            file_path, file = file_info
            task = self._process_single_file_async(file_path, file, kb_dir)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                file_path, file = file_batch[i]
                print(f"Error processing {file_path}: {result}")

    async def _process_single_file_async(self, file_path, file, kb_dir):
        """Process a single file asynchronously."""
        print(f"Processing: {file_path}")
        console_logger.update_status(file)
        
        # Generate summary using LLM
        summary = await self._generate_file_summary_async(file_path)
        
        # Create output filename
        base_name = os.path.splitext(file)[0]
        kb_filename = f"{base_name}_codewalk.kb"
        kb_file_path = os.path.join(kb_dir, kb_filename)
        
        # Write summary to kb file
        with open(kb_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# CodeWalk Summary for {file_path}\n\n")
            f.write(f"Original file: {file_path}\n")
            f.write(f"Generated on: {os.path.getctime(file_path)}\n\n")
            f.write("## Summary\n\n")
            f.write(summary)
        
        print(f"Created: {kb_file_path}")
        return kb_file_path

    def run_codewalk_parallel(self, parallel_count=3, kb_basedir=None, code_base_path=None):
        """Run a code walk with parallel processing of files."""
        if kb_basedir is None:
            kb_basedir = os.path.join(self.code_base_path, CODEWALKER_KB_PREFIX.rstrip('/'))
        
        # Ensure kb_basedir exists
        os.makedirs(kb_basedir, exist_ok=True)

        # Load ignore patterns
        ignore_patterns = self._load_ignore_patterns(self.code_base_path)
        
        if code_base_path:
            self.code_base_path = code_base_path

        # Collect all files to process
        all_files = []
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(self.code_base_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d), ignore_patterns)]
            
            # Skip if current directory should be ignored
            if self._should_ignore(root, ignore_patterns):
                continue
            
            # Create corresponding directory in kb structure
            kb_dir = self._create_kb_directory(root, kb_basedir)
            
            # Collect files for processing
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip ignored files
                if self._should_ignore(file_path, ignore_patterns):
                    continue
                
                all_files.append((file_path, file, kb_dir))

        if not all_files:
            print("No files to process.")
            return

        print(f"Found {len(all_files)} files to process with {parallel_count} parallel workers")
        
        # Process files in batches using asyncio
        asyncio.run(self._process_files_parallel(all_files, parallel_count))
        
        print(f"CodeWalk completed. Knowledge base created in: {kb_basedir}")

    async def _process_files_parallel(self, all_files, parallel_count):
        """Process all files with limited parallelism using semaphore."""
        semaphore = asyncio.Semaphore(parallel_count)
        
        async def process_with_semaphore(file_info):
            async with semaphore:
                file_path, file, kb_dir = file_info
                return await self._process_single_file_async(file_path, file, kb_dir)
        
        tasks = [process_with_semaphore(file_info) for file_info in all_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                file_path, file, kb_dir = all_files[i]
                print(f"Error processing {file_path}: {result}")


# code_walker2 = CodeWalker2(code_base_path=".")
