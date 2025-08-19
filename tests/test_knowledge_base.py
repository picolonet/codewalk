#!/usr/bin/env python3
"""Simple test script for KnowledgeBase class."""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add the cw directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cw'))

from kb.knowledge_base import KnowledgeBase


def create_test_kb():
    """Create a temporary test knowledge base structure."""
    temp_dir = tempfile.mkdtemp()
    kb_root = os.path.join(temp_dir, '.cw_kb')
    os.makedirs(kb_root)
    
    # Create root module metadata
    root_metadata = {
        "kb_files": ["main.py_codewalk.kb", "utils.py_codewalk.kb"],
        "submodules": ["src", "tests"]
    }
    with open(os.path.join(kb_root, 'cw_kb_module.json'), 'w') as f:
        json.dump(root_metadata, f)
    
    # Create root module summary
    with open(os.path.join(kb_root, 'cw_module_summary.kb'), 'w') as f:
        f.write("Root module summary: Main application entry point.")
    
    # Create file summaries
    with open(os.path.join(kb_root, 'main.py_codewalk.kb'), 'w') as f:
        f.write("Main.py summary: Entry point for the application.")
    
    with open(os.path.join(kb_root, 'utils.py_codewalk.kb'), 'w') as f:
        f.write("Utils.py summary: Utility functions used throughout the app.")
    
    # Create src subdirectory
    src_dir = os.path.join(kb_root, 'src')
    os.makedirs(src_dir)
    
    src_metadata = {
        "kb_files": ["app.py_codewalk.kb"],
        "submodules": []
    }
    with open(os.path.join(src_dir, 'cw_kb_module.json'), 'w') as f:
        json.dump(src_metadata, f)
    
    with open(os.path.join(src_dir, 'cw_module_summary.kb'), 'w') as f:
        f.write("Src module summary: Core application logic.")
    
    with open(os.path.join(src_dir, 'app.py_codewalk.kb'), 'w') as f:
        f.write("App.py summary: Main application class and logic.")
    
    return kb_root, temp_dir


def test_knowledge_base():
    """Test the KnowledgeBase class."""
    kb_root, temp_dir = create_test_kb()
    
    try:
        # Test initialization
        kb = KnowledgeBase(kb_root)
        print("✓ KnowledgeBase initialized successfully")
        
        # Test get_module_summary
        root_summary = kb.get_module_summary("")
        assert root_summary == "Root module summary: Main application entry point."
        print("✓ Root module summary retrieved")
        
        src_summary = kb.get_module_summary("src")
        assert src_summary == "Src module summary: Core application logic."
        print("✓ Submodule summary retrieved")
        
        # Test get_file_summary
        main_summary = kb.get_file_summary("main.py")
        assert main_summary == "Main.py summary: Entry point for the application."
        print("✓ Root file summary retrieved")
        
        app_summary = kb.get_file_summary("src/app.py")
        assert app_summary == "App.py summary: Main application class and logic."
        print("✓ Subdirectory file summary retrieved")
        
        # Test get_module_metadata
        root_metadata = kb.get_module_metadata("")
        assert "main.py_codewalk.kb" in root_metadata["kb_files"]
        assert "src" in root_metadata["submodules"]
        print("✓ Module metadata retrieved")
        
        # Test list_modules
        modules = kb.list_modules("")
        assert "src" in modules
        assert "tests" in modules
        print("✓ Module listing works")
        
        # Test list_files
        files = kb.list_files("")
        assert "main.py" in files
        assert "utils.py" in files
        print("✓ File listing works")
        
        # Test module_exists
        assert kb.module_exists("src")
        assert not kb.module_exists("nonexistent")
        print("✓ Module existence check works")
        
        # Test file_has_summary
        assert kb.file_has_summary("main.py")
        assert kb.file_has_summary("src/app.py")
        assert not kb.file_has_summary("nonexistent.py")
        print("✓ File summary existence check works")
        
        # Test nonexistent items
        assert kb.get_module_summary("nonexistent") is None
        assert kb.get_file_summary("nonexistent.py") is None
        assert kb.get_module_metadata("nonexistent") is None
        print("✓ Nonexistent item handling works")
        
        print("\n🎉 All tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_knowledge_base()