from typing import Optional
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
import json

class ConsoleLogger:
    def __init__(self):
        # Initialize Rich Console
        self.console = Console()

    def log_json_panel(self, message: dict, level: str = "INFO", title: Optional[str] = None, type = "to_llm"):
        """
        Prints a JSON message in a log format using Rich.
        
        Args:
            message (dict): The JSON message to log.
            level (str): The log level (e.g., INFO, DEBUG, ERROR).
            type (str): to_llm or from_llm or user 
        """
        # Format the JSON
        json_renderable = JSON(json.dumps(message, indent=2))
        self.log_text_panel(message=json_renderable, level=level, title=title, type=type)

    def log_text_panel(self, message: str, level: str = "INFO", title: Optional[str] = None, type = "to_llm"):
        """
        Prints a text message in a log format using Rich.
        """
             # Time stamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Construct the log header
        header = Text(f"[{timestamp}] [{level.upper()}] {title}", style="bold blue" if level == "INFO" else "bold red" if level == "ERROR" else "bold yellow")

         # Optional series/group title
        #if title:
        #   self.console.rule(f"[bold white]{title}")

        border_color = {
            "to_llm": "green",
            "from_llm": "blue",
            "user": "yellow"
        }.get(level, "white")

        # Combine header and JSON into a panel
        panel = Panel(
            message,
            title=header,
            border_style=border_color,
            expand=True
        )

        # Print the panel to the console
        self.console.print(panel)

    def log_text(self, message:str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.print(f"[{timestamp}] [{level.upper()}] {message}")

    def set_status(self, status):
        self.console_status = status

    def update_status(self, status_msg):
        if self.console_status:
            self.console_status.update(f"[bold green]Walking through the code ... {status_msg}", spinner="dots")

    def reset_token_usage(self):
        self.current_token_usage = 0

    def update_token_usage(self, token_usage: int):
        self.current_token_usage += token_usage
        


    # Example usage
if __name__ == "__main__":
    sample_json = {
        "event": "user_login",
        "user_id": 12345,
        "success": True,
        "metadata": {"ip": "192.168.1.10", "browser": "Firefox"}
    }
    cl = ConsoleLogger()
    cl.log_json_panel(sample_json, level="INFO")
    cl.log_json_panel(sample_json, level="ERROR")
    cl.log_json_panel(sample_json, level="DEBUG")

console_logger = ConsoleLogger()