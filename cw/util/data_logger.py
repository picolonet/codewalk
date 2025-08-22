import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from llm.llm_model import LlmResponse


@dataclass
class StatEntry:
    """Data structure for logging statistics."""
    timestamp: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    operation: str  # e.g., "completion", "codewalk", "query"
    session_id: Optional[str] = None
    error: Optional[str] = None
    stats_type: str = "SINGLE"
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DataLogger:
    """Logger for stats and general messages with date-based file naming."""
    
    def __init__(self, base_dir: str = ".", logs_subdir: str = "logs"):
        """
        Initialize the DataLogger.
        
        Args:
            base_dir: Base directory for the logs folder
            logs_subdir: Subdirectory name for logs (default: "logs")
        """
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / logs_subdir
        
        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Current date for file naming
        self.current_date = datetime.now().strftime("%Y%m%d")
        
        # File paths
        self.stats_file = self.logs_dir / f"stats_{self.current_date}.log"
        self.general_file = self.logs_dir / f"general_{self.current_date}.log"
        
        # Initialize loggers
        self._setup_loggers()
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.inmemory_stats = {}
        
    def _setup_loggers(self):
        """Setup separate loggers for stats and general messages."""
        # Stats logger
        self.stats_logger = logging.getLogger(f"data_logger_stats_{id(self)}")
        self.stats_logger.setLevel(logging.INFO)
        self.stats_logger.handlers.clear()  # Clear any existing handlers
        
        stats_handler = logging.FileHandler(self.stats_file, mode='a', encoding='utf-8')
        stats_formatter = logging.Formatter('%(message)s')  # JSON format, no extra formatting
        stats_handler.setFormatter(stats_formatter)
        self.stats_logger.addHandler(stats_handler)
        self.stats_logger.propagate = False
        
        # General logger
        self.general_logger = logging.getLogger(f"data_logger_general_{id(self)}")
        self.general_logger.setLevel(logging.DEBUG)
        self.general_logger.handlers.clear()  # Clear any existing handlers
        
        general_handler = logging.FileHandler(self.general_file, mode='a', encoding='utf-8')
        general_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        general_handler.setFormatter(general_formatter)
        self.general_logger.addHandler(general_handler)
        self.general_logger.propagate = False
        
    def _check_date_rollover(self):
        """Check if we need to roll over to a new date's log files."""
        current_date = datetime.now().strftime("%Y%m%d")
        if current_date != self.current_date:
            self.current_date = current_date
            self.stats_file = self.logs_dir / f"stats_{self.current_date}.log"
            self.general_file = self.logs_dir / f"general_{self.current_date}.log"
            self._setup_loggers()
    
    def update_inmemory_stats(self, stats_key: str,model_name: str, prompt_tokens: int, completion_tokens: int,
                  latency_seconds: float, operation: str = "completion",
                  error: Optional[str] = None, **kwargs):

        if stats_key not in self.inmemory_stats:
            self.inmemory_stats[stats_key] = {
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_seconds": latency_seconds,
                "operation": operation
            }
        else:
            self.inmemory_stats[stats_key]["prompt_tokens"] += prompt_tokens
            self.inmemory_stats[stats_key]["completion_tokens"] += completion_tokens
            self.inmemory_stats[stats_key]["latency_seconds"] += latency_seconds

    def update_inmemory_stats_with_tool_calls(self, stats_key: str, tool_name: str):
        if stats_key not in self.inmemory_stats:
            return
        
        if tool_name not in self.inmemory_stats[stats_key]:
            self.inmemory_stats[stats_key][tool_name] = 1
        else:
            self.inmemory_stats[stats_key][tool_name] += 1
        

    def log_inmemory_stats(self, stats_key: str):
        if stats_key not in self.inmemory_stats:
            return
        
        stat_entry = self.inmemory_stats[stats_key]
        get_file_summary_count = stat_entry.get("get_file_summary", 0)
        get_module_summary_count = stat_entry.get("get_module_summary", 0)
        get_module_metadata_count = stat_entry.get("get_module_metadata", 0)
        list_modules_count = stat_entry.get("list_modules", 0)
        list_files_count = stat_entry.get("list_files", 0)
        module_exists_count = stat_entry.get("module_exists", 0)
        file_has_summary_count = stat_entry.get("file_has_summary", 0)
        get_file_contents_count = stat_entry.get("get_file_contents", 0)
        list_directory_count = stat_entry.get("list_directory", 0)
        search_files_count = stat_entry.get("search_files", 0)
        todo_write_count = stat_entry.get("todo_write", 0)

        self.log_stats(
            model_name=stat_entry["model_name"],
            prompt_tokens=stat_entry["prompt_tokens"],
            completion_tokens=stat_entry["completion_tokens"],
            latency_seconds=stat_entry["latency_seconds"],
            operation=stat_entry["operation"],
            stats_type="AGGREGATED",
            get_file_summary_count=get_file_summary_count,
            get_module_summary_count=get_module_summary_count,
            get_module_metadata_count=get_module_metadata_count,
            list_modules_count=list_modules_count,
            list_files_count=list_files_count,
            module_exists_count=module_exists_count,
            file_has_summary_count=file_has_summary_count,
            get_file_contents_count=get_file_contents_count,
            list_directory_count=list_directory_count,
            search_files_count=search_files_count,
            todo_write_count=todo_write_count
        )
        del self.inmemory_stats[stats_key]

    def log_stats(self, model_name: str, prompt_tokens: int, completion_tokens: int,
                  latency_seconds: float, operation: str = "completion",
                  error: Optional[str] = None, stats_type: str = "SINGLE", **kwargs):
        """
        Log statistics entry.
        
        Args:
            model_name: Name of the LLM model used
            prompt_tokens: Number of input tokens
            completion_tokens: Number of completion tokens
            latency_seconds: Response latency in seconds
            operation: Type of operation (e.g., "completion", "codewalk", "query")
            error: Optional error message
            **kwargs: Additional fields to include in the stats
        """
        self._check_date_rollover()
        
        total_tokens = prompt_tokens + completion_tokens
        timestamp = datetime.now().isoformat()
        
        stat_entry = StatEntry(
            timestamp=timestamp,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            operation=operation,
            session_id=self.session_id,
            error=error,
            stats_type=stats_type
        )
        
        # Add any additional kwargs to the entry
        stat_dict = stat_entry.to_dict()
        stat_dict.update(kwargs)
        
        # Log as JSON
        json_entry = json.dumps(stat_dict, separators=(',', ':'))
        self.stats_logger.info(json_entry)
    
    def log_llm_response(self, llm_response, model_name: str, operation: str = "completion", **kwargs):
        """
        Log statistics from an LlmResponse object.
        
        Args:
            llm_response: LlmResponse object containing usage and latency info
            model_name: Name of the LLM model used
            operation: Type of operation
            **kwargs: Additional fields to include
        """
        usage = llm_response.usage or {}
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        latency_seconds = llm_response.latency_seconds or 0.0
        
        # Check for errors in the response
        error = None
        if llm_response.finish_reason and llm_response.finish_reason != 'stop':
            error = f"finish_reason: {llm_response.finish_reason}"
        
        self.log_stats(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency_seconds,
            operation=operation,
            error=error,
            **kwargs
        )
    
    def log_info(self, message: str, **kwargs):
        """Log an info message."""
        self._check_date_rollover()
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.general_logger.info(full_message)
    
    def log_debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._check_date_rollover()
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.general_logger.debug(full_message)
    
    def log_warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._check_date_rollover()
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.general_logger.warning(full_message)
    
    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log an error message."""
        self._check_date_rollover()
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        full_message = f"{message} | {extra_info}" if extra_info else message
        
        if exception:
            full_message += f" | Exception: {str(exception)}"
        
        self.general_logger.error(full_message)
    
    def get_stats_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of stats from today's log file.
        
        Args:
            operation: Optional filter by operation type
            
        Returns:
            Dictionary with summary statistics
        """
        if not self.stats_file.exists():
            return {"message": "No stats file found for today"}
        
        total_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        models_used = set()
        operations = set()
        errors = 0
        
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            
                            # Filter by operation if specified
                            if operation and entry.get('operation') != operation:
                                continue
                            
                            total_calls += 1
                            total_prompt_tokens += entry.get('prompt_tokens', 0)
                            total_completion_tokens += entry.get('completion_tokens', 0)
                            total_latency += entry.get('latency_seconds', 0.0)
                            models_used.add(entry.get('model_name', 'unknown'))
                            operations.add(entry.get('operation', 'unknown'))
                            
                            if entry.get('error'):
                                errors += 1
                                
                        except json.JSONDecodeError:
                            continue
            
            avg_latency = total_latency / total_calls if total_calls > 0 else 0.0
            
            return {
                "date": self.current_date,
                "total_calls": total_calls,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "total_latency_seconds": round(total_latency, 2),
                "average_latency_seconds": round(avg_latency, 2),
                "models_used": list(models_used),
                "operations": list(operations),
                "errors": errors,
                "filtered_by_operation": operation
            }
            
        except Exception as e:
            return {"error": f"Failed to read stats file: {str(e)}"}
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Remove log files older than specified days.
        
        Args:
            days_to_keep: Number of days of logs to retain
        """
        if not self.logs_dir.exists():
            return
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_date_str = cutoff_date.strftime("%Y%m%d")
        
        for log_file in self.logs_dir.glob("*.log"):
            # Extract date from filename (assumes format: prefix_YYYYMMDD.log)
            try:
                filename = log_file.stem
                if '_' in filename:
                    date_part = filename.split('_')[-1]
                    if len(date_part) == 8 and date_part.isdigit():
                        if date_part < cutoff_date_str:
                            log_file.unlink()
                            self.log_info(f"Cleaned up old log file: {log_file.name}")
            except Exception as e:
                self.log_error(f"Error cleaning up log file {log_file.name}", exception=e)


# Global instance
_global_data_logger = None

def get_data_logger(base_dir: str = ".") -> DataLogger:
    """Get or create a global DataLogger instance."""
    global _global_data_logger
    if _global_data_logger is None:
        _global_data_logger = DataLogger(base_dir)
    return _global_data_logger


# Example usage and testing
if __name__ == "__main__":
    # Test the DataLogger
    logger = DataLogger()
    
    # Test general logging
    logger.log_info("DataLogger initialized")
    logger.log_debug("This is a debug message", user_id="test_user")
    logger.log_warning("This is a warning", component="test")
    logger.log_error("This is an error", operation="test")
    
    # Test stats logging
    logger.log_stats(
        model_name="gpt-4o",
        prompt_tokens=150,
        completion_tokens=75,
        latency_seconds=1.234,
        operation="test_completion"
    )
    
    logger.log_stats(
        model_name="claude-3-sonnet",
        prompt_tokens=200,
        completion_tokens=100,
        latency_seconds=0.987,
        operation="codewalk",
        file_processed="test.py"
    )
    
    # Test stats summary
    summary = logger.get_stats_summary()
    print("Stats Summary:")
    print(json.dumps(summary, indent=2))
    
    # Test operation-specific summary
    completion_summary = logger.get_stats_summary(operation="test_completion")
    print("\nCompletion Stats:")
    print(json.dumps(completion_summary, indent=2))