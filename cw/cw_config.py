import os
from typing import Dict, Optional, Any
from pathlib import Path


class CwConfig:
    """Configuration loader for codewalk.conf files with x=y format."""

    KB_DIR_KEY = "kb_dir"
    KB_DIR_DEFAULT = ".cw_kb"
    NUM_PARALLEL_KEY = "num_parallel"
    NUM_PARALLEL_DEFAULT = 4
    LLM_MODEL_KEY = "llm_model"
    LLM_MODEL_DEFAULT = "litellm"
    OUTPUT_FILE_KEY = "output_file"
    OUTPUT_FILE_DEFAULT = "codewalk_out.txt"
    GROQ_MODEL_KEY = "groq_model"
    GROQ_MODEL_DEFAULT = "groq/llama-3.1-70b-versatile"
    VALID_MODELS = ["azure_oai", "oai", "claude", "llama", "litellm", "codex"]
    KB_ENABLED_KEY = "kb_enabled"
    KB_ENABLED_DEFAULT = "false"
    KB_ENABLED_VALUE = "true"
    KB_ENABLED_VALUES =  ["true", "false"]
    
    def __init__(self, config_file_path: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_file_path: Path to the config file. If None, looks for codewalk.conf
                             in current directory, then parent directories.
        """
        self.config: Dict[str, str] = {}
        self.config_file: Optional[Path] = None
        
        if config_file_path:
            self.config_file = Path(config_file_path)
        else:
            self.config_file = self._find_config_file()
        
        if self.config_file and self.config_file.exists():
            self._load_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find codewalk.conf file in current or parent directories."""
        current_dir = Path.cwd()
        
        # Check current directory and walk up parent directories
        for parent in [current_dir] + list(current_dir.parents):
            config_file = parent / "codewalk.conf"
            if config_file.exists():
                return config_file
        
        return None
    
    def _load_config(self) -> None:
        """Load configuration from the config file."""
        if not self.config_file or not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse x=y format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        self.config[key] = value
                    else:
                        # Invalid format, skip line
                        continue
                        
        except Exception as e:
            raise IOError(f"Error reading config file {self.config_file}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Boolean value
        """
        value = self.config.get(key)
        if value is None:
            return default
        
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer.
        
        Args:
            key: Configuration key
            default: Default value if key not found or conversion fails
            
        Returns:
            Integer value
        """
        value = self.config.get(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float.
        
        Args:
            key: Configuration key
            default: Default value if key not found or conversion fails
            
        Returns:
            Float value
        """
        value = self.config.get(key)
        if value is None:
            return default
        
        try:
            return float(value)
        except ValueError:
            return default
    
    def set(self, key: str, value: str) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = str(value)
    
    def has_key(self, key: str) -> bool:
        """Check if configuration key exists.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.config
    
    def keys(self):
        """Get all configuration keys."""
        return self.config.keys()
    
    def items(self):
        """Get all configuration key-value pairs."""
        return self.config.items()
    
    def save(self, file_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            file_path: Path to save config. If None, uses original config file path.
        """
        target_file = Path(file_path) if file_path else self.config_file
        
        if not target_file:
            raise ValueError("No config file path specified")
        
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write("# Codewalk configuration file\n")
                f.write("# Format: key=value\n\n")
                
                for key, value in sorted(self.config.items()):
                    # Quote values that contain spaces or special characters
                    if ' ' in value or any(char in value for char in ['#', '"', "'"]):
                        value = f'"{value}"'
                    f.write(f"{key}={value}\n")
                    
        except Exception as e:
            raise IOError(f"Error writing config file {target_file}: {str(e)}")
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.config.clear()
        if self.config_file and self.config_file.exists():
            self._load_config()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"CwConfig({len(self.config)} keys)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"CwConfig(file={self.config_file}, keys={list(self.config.keys())})"


# Global singleton instance
_cw_config_instance = None


def get_cw_config() -> CwConfig:
    """Get the singleton CwConfig instance that loads codewalk.conf from current directory."""
    global _cw_config_instance
    if _cw_config_instance is None:
        _cw_config_instance = CwConfig()
    return _cw_config_instance


# Expose singleton instance directly
cw_config = get_cw_config()
