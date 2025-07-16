from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import print as rprint
import sys


class CodeWalker:
    """Stub class for code walking functionality - to be implemented later."""
    
    def __init__(self):
        pass
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a response."""
        return f"CodeWalker processed: {query}"


class CodeWalkCli:
    """Command line interface for CodeWalk using rich library."""
    
    def __init__(self):
        self.console = Console()
        self.code_walker = CodeWalker()
        self.running = True
        self.config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
    def show_banner(self):
        """Display the application banner."""
        banner = """
╔═══════════════════════════════════════╗
║              CodeWalk CLI             ║
║          Code Analysis Tool           ║
╚═══════════════════════════════════════╝
        """
        self.console.print(Panel(banner, style="bold blue"))
    
    def show_help(self):
        """Display help information."""
        help_text = """
**Available Commands:**

• `/help` - Show this help message\n
• `/config` - Show current configuration\n
• `/set <param> <value>` - Set configuration parameter\n
• `/exit` or `/quit` - Exit the application\n
• `/clear` - Clear the screen\n

**Usage:**
Type your query or question about the codebase, and CodeWalk will analyze it.
Use slash commands (/) for special operations.
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
    
    def show_config(self):
        """Display current configuration."""
        table = Table(title="Current Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in self.config.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
    
    def set_config(self, param: str, value: str):
        """Set a configuration parameter."""
        if param not in self.config:
            self.console.print(f"[red]Unknown parameter: {param}[/red]")
            self.console.print(f"Available parameters: {', '.join(self.config.keys())}")
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
            
            self.console.print(f"[green]Set {param} = {self.config[param]}[/green]")
        except ValueError:
            self.console.print(f"[red]Invalid value for {param}: {value}[/red]")
    
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
            self.console.print("[red]Usage: /set <parameter> <value>[/red]")
        elif command in ['exit', 'quit', 'q']:
            self.console.print("[yellow]Goodbye![/yellow]")
            self.running = False
        elif command == 'clear':
            self.console.clear()
            self.show_banner()
        else:
            self.console.print(f"[red]Unknown command: /{command}[/red]")
            self.console.print("Type /help for available commands.")
        
        return True
    
    def process_query(self, query: str):
        """Process a user query through the CodeWalker."""
        with self.console.status("[bold green]Processing query...", spinner="dots"):
            response = self.code_walker.process_query(query)
        
        # Display the response in a panel
        self.console.print(Panel(
            response,
            title="Response",
            border_style="blue",
            padding=(1, 2)
        ))
    
    def run(self):
        """Main interaction loop."""
        self.show_banner()
        self.console.print("[dim]Type /help for available commands or enter a query to analyze code.[/dim]")
        self.console.print()
        
        while self.running:
            try:
                user_input = Prompt.ask(
                    "[bold cyan]CodeWalk[/bold cyan] ",
                    default="",
                    show_default=False
                ).strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if self.parse_command(user_input):
                    continue
                
                # Process as a query
                self.process_query(user_input)
                self.console.print()
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Next time use /exit to quit properly.[/yellow]")
                break
            except EOFError:
                self.console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    cli = CodeWalkCli()
    cli.run()