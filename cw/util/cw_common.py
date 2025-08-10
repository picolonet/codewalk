from typing import List, Dict, Any



import os
import platform
from datetime import datetime

def get_env() -> str:
    """Get current environment information as a formatted string."""
    cwd = os.getcwd()
    is_git = os.path.exists(os.path.join(cwd, '.git'))
    plat = platform.system().lower()
    os_ver = platform.version()
    today = datetime.now().strftime('%Y-%m-%d')

    env_info = f"""Working directory: {cwd}
Is directory a git repo: {'Yes' if is_git else 'No'}
Platform: {plat}
OS Version: {os_ver}
Today's date: {today}
"""
    return env_info
