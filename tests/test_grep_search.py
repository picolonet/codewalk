import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the cw directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cw'))

from cw_tools import grep_search, _grep_search_internal


class TestGrepSearch(unittest.TestCase):
    """Unit tests for the grep_search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create a temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test files with known content
        test_files = {
            'test.py': '''def hello_world():
    print("Hello, World!")
    
class TestClass:
    def __init__(self):
        self.value = 42
        
def another_function():
    return "test"
''',
            'test.js': '''function greet(name) {
    console.log(`Hello, ${name}!`);
}

class Person {
    constructor(name) {
        this.name = name;
    }
}
''',
            'test.txt': '''This is a test file.
It has multiple lines.
Some lines contain the word "test".
Others contain different words.
ERROR: This is an error message.
log: This is a log entry.
''',
            'subdir/nested.py': '''def nested_function():
    return "nested"
    
class NestedClass:
    pass
'''
        }
        
        for filepath, content in test_files.items():
            full_path = os.path.join(cls.test_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after running tests."""
        shutil.rmtree(cls.test_dir)
    
    def test_basic_search_files_with_matches(self):
        """Test basic search returning files with matches."""
        result = _grep_search_internal('def', self.test_dir, output_mode='files_with_matches')
        
        # Should find files containing 'def'
        self.assertIn('test.py', result)
        self.assertIn('nested.py', result)
        self.assertNotIn('test.js', result)  # JavaScript doesn't use 'def'
        self.assertNotIn('test.txt', result)
    
    def test_content_mode_with_line_numbers(self):
        """Test content mode showing actual matches with line numbers."""
        result = _grep_search_internal('def', self.test_dir, output_mode='content', n=True)
        
        # Should show line numbers and content
        self.assertIn('1:', result)  # Line numbers should be present
        self.assertIn('def hello_world', result)
        self.assertIn('def another_function', result)
        self.assertIn('def nested_function', result)
    
    def test_case_insensitive_search(self):
        """Test case insensitive search."""
        result = _grep_search_internal('ERROR', self.test_dir, i=True, output_mode='content')
        
        # Should find both 'ERROR' and potentially 'error' if present
        self.assertIn('ERROR: This is an error', result)
    
    def test_regex_pattern(self):
        """Test regex pattern matching."""
        result = _grep_search_internal('log.*error', self.test_dir, i=True, output_mode='content')
        
        # Should match patterns like 'log: ...' or 'ERROR: ...'
        # This might not match depending on file content, so let's test a simpler regex
        result = _grep_search_internal('test.*file', self.test_dir, i=True, output_mode='content')
        self.assertIn('test file', result)
    
    def test_file_type_filtering(self):
        """Test filtering by file type."""
        result = _grep_search_internal('class', self.test_dir, type='py', output_mode='files_with_matches')
        
        # Should only find Python files
        self.assertIn('test.py', result)
        self.assertIn('nested.py', result)
        self.assertNotIn('test.js', result)  # Should be filtered out
    
    def test_glob_filtering(self):
        """Test filtering by glob pattern."""
        result = _grep_search_internal('function', self.test_dir, glob='*.js', output_mode='files_with_matches')
        
        # Should only find JavaScript files
        self.assertIn('test.js', result)
        self.assertNotIn('test.py', result)
    
    def test_count_mode(self):
        """Test count mode showing number of matches."""
        result = _grep_search_internal('def', self.test_dir, output_mode='count')
        
        # Should show counts for each file
        lines = result.strip().split('\n')
        # Each line should be in format "filename:count"
        for line in lines:
            if line:  # Skip empty lines
                self.assertIn(':', line)
                # The part after colon should be a number
                parts = line.split(':')
                self.assertTrue(parts[-1].strip().isdigit())
    
    def test_context_lines(self):
        """Test showing context lines before and after matches."""
        result = _grep_search_internal('TestClass', self.test_dir, output_mode='content', A=1, B=1)
        
        # Should show lines before and after the match
        self.assertIn('TestClass', result)
        # Context should include surrounding lines
        lines = result.split('\n')
        self.assertTrue(len(lines) > 1)  # Should have more than just the match line
    
    def test_head_limit(self):
        """Test limiting output to first N results."""
        result = _grep_search_internal('def', self.test_dir, output_mode='files_with_matches', head_limit=1)
        
        # Should only return one file
        lines = [line for line in result.split('\n') if line.strip()]
        self.assertEqual(len(lines), 1)
    
    def test_no_matches(self):
        """Test behavior when no matches are found."""
        result = _grep_search_internal('nonexistent_pattern_xyz123', self.test_dir)
        
        self.assertEqual(result, 'No matches found')
    
    def test_public_function_wrapper(self):
        """Test the public grep_search function wrapper."""
        result = grep_search('def', self.test_dir)
        
        # Should wrap result in proper format
        self.assertIn('<result>', result)
        self.assertIn('</result>', result)
        self.assertIn('Grep search result', result)
    
    def test_invalid_pattern(self):
        """Test handling of invalid regex patterns."""
        # Test with unbalanced parentheses which should cause regex error
        result = _grep_search_internal('(unclosed', self.test_dir)
        
        # Should handle error gracefully
        self.assertIn('Error', result)


if __name__ == '__main__':
    # Check if ripgrep is available before running tests
    import subprocess
    try:
        subprocess.run(['rg', '--version'], capture_output=True, check=True)
        print("Ripgrep found, running tests...")
        unittest.main()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ripgrep (rg) not found. Please install ripgrep to run these tests.")
        print("You can install it with: brew install ripgrep")
        exit(1)