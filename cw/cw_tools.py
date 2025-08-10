from util.cw_constants import IGNORE_FILE_LIST
import os



def glob_files(pattern: str, directory: str = ".") -> str:
    """Glob:
    Fast file pattern matching using glob patterns like '*.js' or 'src/**/*.ts'.
    - Fast file pattern matching tool that works with any codebase size
    - Supports glob patterns like "/*.js" or "src//*.ts"
    - Returns matching file paths sorted by modification time
    - Use this tool when you need to find files by name patterns
    - You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful."""

    actual_result = _glob_files_internal(pattern, directory)
    result = f"Glob search result for pattern '{pattern}' in {directory}:\n<result>\n{actual_result}\n</result>"
    return result

def _glob_files_internal(pattern: str, directory: str = ".") -> str:
    """Internal implementation for glob file matching."""

    import glob
    
    try:
        # Change to the specified directory if provided
        original_cwd = os.getcwd()
        if directory and directory != ".":
            os.chdir(directory)
        
        # Use glob to find matching files
        matches = glob.glob(pattern, recursive=True)
        
        # Filter out ignored items
        filtered_matches = []
        for match in matches:
            # Check if any part of the path contains ignored items
            path_parts = match.split(os.sep)
            if not any(part in IGNORE_FILE_LIST for part in path_parts):
                if directory and directory != ".":
                    # Make paths relative to original directory
                    filtered_matches.append(os.path.join(directory, match))
                else:
                    filtered_matches.append(match)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        # Sort results for consistent output
        filtered_matches.sort()
        
        return "\n".join(filtered_matches) if filtered_matches else "No files found matching pattern"
    except Exception as e:
        # Make sure to change back to original directory on error
        try:
            os.chdir(original_cwd)
        except:
            pass
        return f"Error in glob search: {str(e)}"


def grep_search(pattern: str, path: str = ".", glob: str = None, output_mode: str = "files_with_matches", 
                B: int = None, A: int = None, C: int = None, n: bool = False, i: bool = False,
                type: str = None, head_limit: int = None, multiline: bool = False) -> str:
    """Grep: A powerful search tool built on ripgrep
    
    - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
    - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
    - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
    - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
    - Use Task tool for open-ended searches requiring multiple rounds
    - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
    - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`
    """
    
    actual_result = _grep_search_internal(pattern, path, glob, output_mode, B, A, C, n, i, type, head_limit, multiline)
    result = f"Grep search result for pattern '{pattern}':\n<result>\n{actual_result}\n</result>"
    return result


def _grep_search_internal(pattern: str, path: str = ".", glob: str = None, output_mode: str = "files_with_matches",
                         B: int = None, A: int = None, C: int = None, n: bool = False, i: bool = False,
                         type: str = None, head_limit: int = None, multiline: bool = False) -> str:
    """Internal implementation for ripgrep search."""
    import subprocess
    import shlex
    
    try:
        # Build ripgrep command
        cmd = ["rg"]
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path if specified
        if path and path != ".":
            cmd.append(path)
        
        # Add output mode flags
        if output_mode == "files_with_matches":
            cmd.append("-l")  # files with matches
        elif output_mode == "count":
            cmd.append("-c")  # count matches
        # content mode is default, no special flag needed
        
        # Add context flags (only for content mode)
        if output_mode == "content":
            if C is not None:
                cmd.extend(["-C", str(C)])
            elif A is not None or B is not None:
                if A is not None:
                    cmd.extend(["-A", str(A)])
                if B is not None:
                    cmd.extend(["-B", str(B)])
            
            if n:
                cmd.append("-n")  # line numbers
        
        # Add case insensitive flag
        if i:
            cmd.append("-i")
        
        # Add multiline mode
        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])
        
        # Add file type filtering
        if type:
            cmd.extend(["--type", type])
        
        # Add glob filtering
        if glob:
            cmd.extend(["--glob", glob])
        
        # Add ignore patterns for common directories
        for ignore_item in IGNORE_FILE_LIST:
            cmd.extend(["--glob", f"!{ignore_item}/**"])
        
        # Execute ripgrep command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        output = result.stdout.strip()
        
        # Apply head limit if specified
        if head_limit and output:
            lines = output.split('\n')
            output = '\n'.join(lines[:head_limit])
        
        # Handle different exit codes
        if result.returncode == 0:
            return output if output else "No matches found"
        elif result.returncode == 1:
            return "No matches found"
        else:
            error_msg = result.stderr.strip()
            return f"Error running ripgrep: {error_msg if error_msg else 'Unknown error'}"
            
    except subprocess.TimeoutExpired:
        return "Error: Ripgrep search timed out (30s limit)"
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Please install ripgrep to use this tool."
    except Exception as e:
        return f"Error in grep search: {str(e)}"







