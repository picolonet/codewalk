from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import print as rprint
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
import sys
import logging
from code_walker import code_walker
from tool_caller import CompletionResult
import json
from util.livepanel import LivePanel
from llm_model import Message, format_messages


class CodeWalkCli:
    """Command line interface for CodeWalk using rich library."""
    
    def __init__(self):
        self.console = Console()
        self.running = True
        self.config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        # Single panel approach - no complex layout needed
        
        # Set up debug logger
        self.debug_messages = []
        self.setup_debug_logging()
    
    def setup_debug_logging(self):
        """Set up debug logging to capture messages for the debug panel."""
        # Create a custom handler that captures log messages
        class DebugHandler(logging.Handler):
            def __init__(self, cli_instance):
                super().__init__()
                self.cli = cli_instance
            
            def emit(self, record):
                # Add log message to debug messages list
                log_msg = self.format(record)
                self.cli.add_debug_message(log_msg, record.levelname)
        
        # Set up the handler
        self.debug_handler = DebugHandler(self)
        self.debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Add handler to root logger
        # logging.getLogger().addHandler(self.debug_handler)
        # logging.getLogger().setLevel(logging.DEBUG)
    
    def add_debug_message(self, message: str, level: str = "INFO"):
        """Add a debug message to the debug panel."""
        color_map = {
            "DEBUG": "dim white",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red"
        }
        color = color_map.get(level, "white")
        
        # Add timestamp and color formatting
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{color}]{timestamp} [{level}][/{color}] {message}"
        
        self.debug_messages.append(formatted_msg)
        
        # Keep only last 50 messages to prevent memory issues
        if len(self.debug_messages) > 50:
            self.debug_messages = self.debug_messages[-50:]
        
    def start_debugpanel(self):
        self.debug_panel = LivePanel()
        self.debug_panel.start()

    def post_debugpanel(self, title: str, messages: List[Message]):
        formatted_message = format_messages(messages)
        self.post_debugpanel(title, formatted_message)

    def post_debugpanel(self, title: str, message: str):
        self.debug_panel.add_message(f"{title}\n {message}")

    def stop_debugpanel(self):
        self.debug_panel.stop()
    
    def show_content(self, content: str, title: str = "CodeWalk"):
        """Display content in a panel."""
        self.console.print(Panel(content, title=title, border_style="blue", padding=(1, 2)))
    
    def show_debug_if_needed(self):
        """Show debug messages if any exist."""
        if self.debug_messages:
            debug_text = "\n".join(self.debug_messages[-10:])  # Show last 10 messages
            self.console.print(Panel(debug_text, title="Debug", border_style="green", padding=(1, 1)))
        
    def show_banner(self):
        """Display the application banner."""
        banner = """
╔═══════════════════════════════════════╗
║              CodeWalk CLI             ║
║          Code Analysis Tool           ║
╚═══════════════════════════════════════╝

Type /help for available commands or enter a query to analyze code.
        """
        self.show_content(banner, "CodeWalk CLI")
    
    def show_help(self):
        """Display help information."""
        help_text = """
**Available Commands:**

• `/help` - Show this help message
• `/config` - Show current configuration
• `/set <param> <value>` - Set configuration parameter
• `/exit` or `/quit` - Exit the application
• `/clear` - Clear the screen

**Usage:**
Type your query or question about the codebase, and CodeWalk will analyze it.
Use slash commands (/) for special operations.
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
    
    def show_config(self):
        """Display current configuration."""
        config_text = "**Current Configuration:**\n\n"
        for key, value in self.config.items():
            config_text += f"• {key}: {value}\n"
        
        self.show_content(config_text, "Configuration")
    
    def set_config(self, param: str, value: str):
        """Set a configuration parameter."""
        if param not in self.config:
            error_msg = f"Unknown parameter: {param}\nAvailable parameters: {', '.join(self.config.keys())}"
            self.show_content(error_msg, "Error")
            return
        
        # Type conversion based on current value type
        current_value = self.config[param]
        try:
            if isinstance(current_value, bool):
                self.config[param] = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(current_value, int):
                self.config[param] = int(value)
            elif isinstance(current_value, float):
                self.config[param] = float(value)
            else:
                self.config[param] = value
            
            success_msg = f"Successfully set {param} = {self.config[param]}"
            self.show_content(success_msg, "Configuration Updated")
        except ValueError:
            error_msg = f"Invalid value for {param}: {value}"
            self.show_content(error_msg, "Error")
    
    def parse_command(self, user_input: str) -> bool:
        """Parse and execute slash commands. Returns True if command was processed."""
        if not user_input.startswith('/'):
            return False
        
        parts = user_input[1:].split()
        if not parts:
            return False
        
        command = parts[0].lower()
        args = parts[1:]
        
        if command in ['help', 'h']:
            self.show_help()
        elif command == 'config':
            self.show_config()
        elif command == 'set' and len(args) >= 2:
            self.set_config(args[0], ' '.join(args[1:]))
        elif command == 'set':
            self.show_content("Usage: /set <parameter> <value>", "Error")
        elif command in ['exit', 'quit', 'q']:
            self.show_content("Goodbye!", "Exit")
            self.running = False
        elif command == 'clear':
            self.console.clear()
            self.show_banner()
        else:
            error_msg = f"Unknown command: /{command}\nType /help for available commands."
            self.show_content(error_msg, "Error")
        
        return True


    
    def process_query(self, query: str):
        """Process a user query through the CodeWalker."""
        # First show the user's query
        self.show_content(f"**User Query:**\n{query}\n\n_Processing..._", "CodeWalk - Query")
        self.add_debug_message(f"Processing query: {query}", "INFO")
        
        try:
            response = code_walker.run_query(query)
            
            # Display the final result
            if hasattr(response, 'content') and response.content:
                self.show_content(response.content, "Response")
            else:
                self.show_content("Query processed successfully but no content returned.", "Response")
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.show_content(error_msg, "Error")
            self.add_debug_message(error_msg, "ERROR")
        
        # Show debug messages if any
        self.show_debug_if_needed()

    def display_completion_result(self, result: CompletionResult):
        """Display a CompletionResult to the console."""
        for message in result.current_result:
            if message.role == "assistant":
                # Display assistant message in a panel
                self.console.print(Panel(
                    message.content if message.content else "No content",
                    title="Assistant",
                    border_style="blue",
                    padding=(1, 2)
                ))
                
                # Display any tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        self.console.print(Panel(
                            json.dumps(tool_call.function, indent=2),
                            title=f"Tool Call: {tool_call.function.get('name', 'unknown')}",
                            border_style="yellow",
                            padding=(1, 2)
                        ))
            
            elif message.role == "tool":
                # Display tool response
                self.console.print(Panel(
                    message.content if message.content else "No content",
                    title=f"Tool Response (ID: {message.tool_call_id})",
                    border_style="green", 
                    padding=(1, 2)
                ))
    
    def run(self):
        """Main interaction loop."""
        self.show_banner()
        
        while self.running:
            try:
                user_input = Prompt.ask(
                    "[bold cyan]CodeWalk[/bold cyan] ",
                    default="",
                    show_default=False,
                    console=self.console
                ).strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if self.parse_command(user_input):
                    continue
                
                # Process as a query
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                self.show_content("Next time use /exit to quit properly.", "Exit")
                break
            except EOFError:
                self.show_content("Goodbye!", "Exit")
                break
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.show_content(error_msg, "Error")
                self.add_debug_message(error_msg, "ERROR")


cli = CodeWalkCli()

if __name__ == "__main__":
    cli.run()