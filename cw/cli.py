

import sys

import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "cw"))

from cw.code_walker2 import CodeWalker2
from cw.kb.knowledge_base import build_knowledge_base_with_summary


from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import print as rprint
from code_walker import code_walker
from console_logger import console_logger
from llm.llm_router import llm_router
from cw.cw_task import CwTask
from kb.kb_builder import KBBuilder
import os
from cw.cw_config import get_cw_config



class CodeWalkCli:
    """Command line interface for CodeWalk using rich library."""
    
    def __init__(self):
        self.console = console_logger.console
        self.running = True
        self.query_count = 0
        self.config = {
           # "model": "gpt-3.5-turbo",
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
• `/model <type>` - Switch LLM model (oai, claude, llama)\n
• `/codewalk [subdirectory]` - Run codewalk analysis (optional subdirectory path)\n
• `/build [pll] [workers]` - Build knowledge base for current directory (optional parallel mode)\n
• `/exit` or `/quit` - Exit the application\n
• `/clear` - Clear the screen\n

**Usage:**
Type your query or question about the codebase, and CodeWalk will analyze it.
Use slash commands (/) for special operations.

**Examples:**
• `/model oai` - Switch to OpenAI GPT-4o
• `/model claude` - Switch to Anthropic Claude
• `/model llama` - Switch to Llama4 Scout via Groq
• `/codewalk` - Analyze current directory
• `/codewalk src` - Analyze the 'src' subdirectory
• `/codewalk /path/to/project` - Analyze specific directory path
• `/build` - Build knowledge base for current directory
• `/build pll 8` - Build knowledge base with 8 parallel workers
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
    
    def show_config(self):
        """Display current configuration."""
        table = Table(title="Current Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Add current model to configuration display
        try:
            router = llm_router()
            current_model = router.get_current_model_name()
            table.add_row("current_model", current_model)
        except Exception:
            table.add_row("current_model", "unknown")
        
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
        elif command == 'model':
            if len(args) == 1:
                self.set_model(args[0])
            else:
                self.console.print("[red]Usage: /model <type>[/red]")
                self.console.print("Available types: oai, claude, llama, litellm")
        elif command == 'build':
            if args and args[0] == 'pll':
                workers = int(args[1]) if len(args) > 1 and args[1].isdigit() else 4
                self.build_knowledge_base(max_workers=workers)
            else:
                self.build_knowledge_base()
        elif command == 'codewalk':
            if args:
                subdirectory = args[0]
                # Check if third argument is 'pll' for parallel mode
                if len(args) >= 2 and args[1] == 'pll':
                    parallel_count = 3  # default
                    if len(args) >= 3 and args[2].isdigit():
                        parallel_count = int(args[2])
                    self.run_codewalk_parallel(subdirectory, parallel_count)
                else:
                    # Use the provided subdirectory as the starting point
                    self.run_codewalk(subdirectory)
            else:
                # Use current directory as the starting point
                self.run_codewalk()
        else:
            self.console.print(f"[red]Unknown command: /{command}[/red]")
            self.console.print("Type /help for available commands.")
        
        return True
    
    def set_model(self, model_type: str):
        """Set the LLM model type."""
        valid_models = get_cw_config().VALID_MODELS
        
        if model_type.lower() not in valid_models:
            self.console.print(f"[red]Error: Invalid model type '{model_type}'[/red]")
            self.console.print(f"Valid models: {', '.join(valid_models)}")
            return
        
        try:
            router = llm_router()
            current_model = router.get_current_model_name()
            
            # Set the new model using the router
            router.set_model(model_type)
            new_model = router.get_current_model_name()
            
            # Show success message
            if current_model != new_model:
                self.console.print(f"[green]✓ Switched from {current_model} to {new_model}[/green]")
            else:
                self.console.print(f"[yellow]Already using {new_model}[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Error switching to {model_type}: {str(e)}[/red]")
            self.console.print("Make sure you have the required API keys configured.")
    
    def run_codewalk(self, subdirectory: str = None):
        """Run codewalk on the specified directory or current directory if none specified."""
        import os
        
        if subdirectory:
            # Use the full path including the subdirectory as the starting point
            code_base_path = os.path.abspath(subdirectory)
            if not os.path.exists(code_base_path):
                self.console.print(f"[red]Directory not found: {code_base_path}[/red]")
                return
            if not os.path.isdir(code_base_path):
                self.console.print(f"[red]Path is not a directory: {code_base_path}[/red]")
                return
            
            self.console.print(f"[dim]Starting codewalk from: {code_base_path}[/dim]")
        else:
            # Use current directory as the starting point
            code_base_path = os.getcwd()
            self.console.print(f"[dim]Starting codewalk from current directory: {code_base_path}[/dim]")
        
        with self.console.status("[bold green]Walking through the code ...", spinner="dots") as status:
            console_logger.set_status(status)
            code_walker.run_codewalk(code_base_path=code_base_path)

    def run_codewalk_parallel(self, subdirectory: str = None, parallel_count: int = 3):
        """Run codewalk with parallel processing."""
        import os
        
        if subdirectory:
            # Use the full path including the subdirectory as the starting point
            code_base_path = os.path.abspath(subdirectory)
            if not os.path.exists(code_base_path):
                self.console.print(f"[red]Directory not found: {code_base_path}[/red]")
                return
            if not os.path.isdir(code_base_path):
                self.console.print(f"[red]Path is not a directory: {code_base_path}[/red]")
                return
            
            self.console.print(f"[dim]Starting parallel codewalk ({parallel_count} workers) from: {code_base_path}[/dim]")
        else:
            # Use current directory as the starting point
            code_base_path = os.getcwd()
            self.console.print(f"[dim]Starting parallel codewalk ({parallel_count} workers) from current directory: {code_base_path}[/dim]")
        
        with self.console.status(f"[bold green]Walking through the code with {parallel_count} parallel workers...", spinner="dots") as status:
            console_logger.set_status(status)
            code_walker.run_codewalk_parallel(parallel_count=parallel_count, code_base_path=code_base_path)

    def build_knowledge_base(self, subdirectory: str = None, max_workers: int = 4):
        """Build knowledge base for the entire repository using parallel file processing."""
        if subdirectory:
            # Use the full path including the subdirectory as the starting point
            code_base_path = os.path.abspath(subdirectory)
            if not os.path.exists(code_base_path):
                self.console.print(f"[red]Directory not found: {code_base_path}[/red]")
                return
            if not os.path.isdir(code_base_path):
                self.console.print(f"[red]Path is not a directory: {code_base_path}[/red]")
                return
            
            self.console.print(f"[dim]Starting codewalk from: {code_base_path}[/dim]")
        else:
            # Use current directory as the starting point
            code_base_path = os.getcwd()
            self.console.print(f"[dim]Starting codewalk from current directory: {code_base_path}[/dim]")
        
        with self.console.status("[bold green]Building knowledge base ...", spinner="dots") as status:
            kb_root = os.path.join(code_base_path, ".cw_kb")
            kb = build_knowledge_base_with_summary(kb_root, max_workers=max_workers)
            if kb:
                self.console.print(f"[green]Knowledge base built successfully in: {kb_root}[/green]")
            else:
                self.console.print(f"[red]Failed to build knowledge base[/red]")
        
    
    def process_query(self, query: str):
        """Process a user query through the CodeWalker."""
        self.query_count += 1
        with self.console.status("[bold green]Processing query...", spinner="dots"):
            #full_response = code_walker.run_query(query)
            # cw_task = CwTask(user_query=query, code_base_path=os.getcwd())
            #full_response = cw_task.run(query)
            code_walker2 = CodeWalker2(code_base_path=os.getcwd())
            full_response = code_walker2.run_query(query, operation_tag=f"user_query_{self.query_count}")
            response = full_response.user_facing_result
             # Display the response in a panel
            console_logger.log_text_panel(response, title="Response", type="from_llm")

        last_response = full_response.last_response
        if hasattr(last_response, 'content') and last_response.content:
            self.console.print(Panel(
                last_response.content if last_response.content else "No response",
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
                raise


def main():
    parser = argparse.ArgumentParser(description='CodeWalk CLI - Code Analysis Tool')
    parser.add_argument('-q', '--query', type=str, help='Process query directly without entering REPL mode')
    
    args = parser.parse_args()
    
    cli = CodeWalkCli()
    
    if args.query:
        # Direct query mode - process and exit
        cli.process_query(args.query)
    else:
        # Interactive REPL mode
        cli.run()

if __name__ == "__main__":
    main()
