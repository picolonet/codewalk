#!/usr/bin/env python3
"""
Stats Analyzer for CodeWalk - Analyzes and visualizes LLM usage statistics.

This script reads the stats log files created by DataLogger and creates various
visualizations including token vs latency plots.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import glob

# Add the parent directory to the path so we can import from cw
sys.path.append(str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    print("Install with: pip install matplotlib seaborn pandas")
    PLOTTING_AVAILABLE = False


class StatsAnalyzer:
    """Analyzer for CodeWalk stats logs."""
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize the stats analyzer.
        
        Args:
            logs_dir: Directory containing the log files
        """
        self.logs_dir = Path(logs_dir)
        if not self.logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
        
        # Set up plotting style
        if PLOTTING_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")
    
    def load_stats_data(self, date: Optional[str] = None, days: int = 7, 
                       model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load stats data from log files.
        
        Args:
            date: Specific date in YYYYMMDD format, or None for recent days
            days: Number of recent days to load if date is None
            model_filter: Optional model name to filter by
            
        Returns:
            List of stat entries as dictionaries
        """
        stats_data = []
        
        if date:
            # Load specific date
            stats_file = self.logs_dir / f"stats_{date}.log"
            if stats_file.exists():
                stats_data.extend(self._load_single_stats_file(stats_file))
            else:
                print(f"Warning: Stats file not found for date {date}")
        else:
            # Load recent days
            today = datetime.now()
            for i in range(days):
                target_date = today - timedelta(days=i)
                date_str = target_date.strftime("%Y%m%d")
                stats_file = self.logs_dir / f"stats_{date_str}.log"
                if stats_file.exists():
                    stats_data.extend(self._load_single_stats_file(stats_file))
        
        # Filter by model if specified
        if model_filter:
            stats_data = [entry for entry in stats_data 
                         if entry.get('model_name', '').lower() == model_filter.lower()]
            print(f"Filtered to {len(stats_data)} entries for model: {model_filter}")
        
        return stats_data
    
    def _load_single_stats_file(self, stats_file: Path) -> List[Dict[str, Any]]:
        """Load and parse a single stats file."""
        entries = []
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num} in {stats_file}: {e}")
        except Exception as e:
            print(f"Error reading {stats_file}: {e}")
        
        return entries
    
    def get_available_models(self, stats_data: List[Dict[str, Any]]) -> List[str]:
        """Get list of available models in the stats data."""
        models = set()
        for entry in stats_data:
            if 'model_name' in entry and entry['model_name']:
                models.add(entry['model_name'])
        return sorted(list(models))
    
    def create_tokens_vs_latency_plot(self, stats_data: List[Dict[str, Any]], 
                                    output_file: Optional[str] = None,
                                    model_filter: Optional[str] = None) -> bool:
        """
        Create a plot showing the relationship between token counts and latency.
        
        Args:
            stats_data: List of stat entries
            output_file: Optional file path to save the plot
            model_filter: Optional model name to filter by and include in title
            
        Returns:
            True if plot was created successfully, False otherwise
        """
        if not PLOTTING_AVAILABLE:
            print("Error: Plotting libraries not available")
            return False
        
        if not stats_data:
            print("Error: No stats data available for plotting")
            return False
        
        # Filter by model if specified
        filtered_data = stats_data
        if model_filter:
            filtered_data = [entry for entry in stats_data 
                           if entry.get('model_name', '').lower() == model_filter.lower()]
            if not filtered_data:
                print(f"Error: No data found for model '{model_filter}'")
                return False
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(filtered_data)
        
        # Filter out entries with missing or invalid data
        required_cols = ['prompt_tokens', 'completion_tokens', 'total_tokens', 'latency_seconds']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Missing required column '{col}' in stats data")
                return False
        
        # Remove entries with null or negative values
        df = df.dropna(subset=required_cols)
        df = df[(df[required_cols] >= 0).all(axis=1)]
        
        if df.empty:
            print("Error: No valid data points after filtering")
            return False
        
        # Determine title based on model filter
        if model_filter:
            title = f"Token Count vs Latency - {model_filter}"
        else:
            # If no filter but data contains models, show which models are included
            available_models = self.get_available_models(filtered_data)
            if len(available_models) == 1:
                title = f"Token Count vs Latency - {available_models[0]}"
            elif len(available_models) <= 3:
                title = f"Token Count vs Latency - {', '.join(available_models)}"
            else:
                title = f"Token Count vs Latency - {len(available_models)} Models"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot three series: prompt_tokens, completion_tokens, total_tokens vs latency
        ax.scatter(df['prompt_tokens'], df['latency_seconds'], 
                  alpha=0.6, s=50, label='Prompt Tokens', color='blue')
        ax.scatter(df['completion_tokens'], df['latency_seconds'], 
                  alpha=0.6, s=50, label='Completion Tokens', color='red')
        ax.scatter(df['total_tokens'], df['latency_seconds'], 
                  alpha=0.6, s=50, label='Total Tokens', color='green')
        
        # Add trend lines
        self._add_trend_line(ax, df['prompt_tokens'], df['latency_seconds'], 'blue')
        self._add_trend_line(ax, df['completion_tokens'], df['latency_seconds'], 'red')
        self._add_trend_line(ax, df['total_tokens'], df['latency_seconds'], 'green')
        
        # Customize the plot
        ax.set_xlabel('Token Count', fontsize=12)
        ax.set_ylabel('Latency (seconds)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = self._generate_stats_text(df, model_filter)
        ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def _add_trend_line(self, ax, x_data, y_data, color):
        """Add a trend line to the plot."""
        try:
            # Calculate linear regression
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            
            # Plot trend line
            x_trend = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.8, linewidth=1)
        except Exception as e:
            print(f"Warning: Could not add trend line: {e}")
    
    def _generate_stats_text(self, df: "pd.DataFrame", model_filter: Optional[str] = None) -> str:
        """Generate statistics text for the plot."""
        total_requests = len(df)
        avg_latency = df['latency_seconds'].mean()
        max_latency = df['latency_seconds'].max()
        avg_total_tokens = df['total_tokens'].mean()
        
        # Calculate correlation coefficients
        corr_prompt = df['prompt_tokens'].corr(df['latency_seconds'])
        corr_completion = df['completion_tokens'].corr(df['latency_seconds'])
        corr_total = df['total_tokens'].corr(df['latency_seconds'])
        
        stats_text = f"""Stats Summary:
Total Requests: {total_requests}
Avg Latency: {avg_latency:.2f}s
Max Latency: {max_latency:.2f}s
Avg Total Tokens: {avg_total_tokens:.0f}"""
        
        if model_filter:
            stats_text = f"Model: {model_filter}\n" + stats_text
        
        stats_text += f"""

Correlations with Latency:
Prompt Tokens: {corr_prompt:.3f}
Completion Tokens: {corr_completion:.3f}
Total Tokens: {corr_total:.3f}"""
        
        return stats_text
    
    def create_model_comparison_plot(self, stats_data: List[Dict[str, Any]], 
                                   output_file: Optional[str] = None) -> bool:
        """Create a plot comparing performance across different models."""
        if not PLOTTING_AVAILABLE:
            print("Error: Plotting libraries not available")
            return False
        
        df = pd.DataFrame(stats_data)
        if 'model_name' not in df.columns:
            print("Error: No model information available")
            return False
        
        # Group by model and calculate metrics
        model_stats = df.groupby('model_name').agg({
            'latency_seconds': ['mean', 'std', 'count'],
            'total_tokens': 'mean',
            'prompt_tokens': 'mean',
            'completion_tokens': 'mean'
        }).round(3)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        models = model_stats.index
        
        # Average latency by model
        latencies = model_stats[('latency_seconds', 'mean')]
        ax1.bar(models, latencies)
        ax1.set_title('Average Latency by Model')
        ax1.set_ylabel('Latency (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Request count by model
        counts = model_stats[('latency_seconds', 'count')]
        ax2.bar(models, counts)
        ax2.set_title('Request Count by Model')
        ax2.set_ylabel('Number of Requests')
        ax2.tick_params(axis='x', rotation=45)
        
        # Average tokens by model
        avg_tokens = model_stats[('total_tokens', 'mean')]
        ax3.bar(models, avg_tokens)
        ax3.set_title('Average Total Tokens by Model')
        ax3.set_ylabel('Average Tokens')
        ax3.tick_params(axis='x', rotation=45)
        
        # Latency distribution by model (box plot)
        model_data = [df[df['model_name'] == model]['latency_seconds'] for model in models]
        ax4.boxplot(model_data, labels=models)
        ax4.set_title('Latency Distribution by Model')
        ax4.set_ylabel('Latency (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def create_time_series_plot(self, stats_data: List[Dict[str, Any]], 
                              output_file: Optional[str] = None,
                              model_filter: Optional[str] = None) -> bool:
        """Create a time series plot of latency and token usage over time."""
        if not PLOTTING_AVAILABLE:
            print("Error: Plotting libraries not available")
            return False
        
        # Filter by model if specified
        filtered_data = stats_data
        if model_filter:
            filtered_data = [entry for entry in stats_data 
                           if entry.get('model_name', '').lower() == model_filter.lower()]
            if not filtered_data:
                print(f"Error: No data found for model '{model_filter}'")
                return False
        
        df = pd.DataFrame(filtered_data)
        if 'timestamp' not in df.columns:
            print("Error: No timestamp information available")
            return False
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime')
        
        # Determine title
        if model_filter:
            title_suffix = f" - {model_filter}"
        else:
            available_models = self.get_available_models(filtered_data)
            if len(available_models) == 1:
                title_suffix = f" - {available_models[0]}"
            else:
                title_suffix = ""
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot latency over time
        ax1.plot(df['datetime'], df['latency_seconds'], marker='o', markersize=3, alpha=0.7)
        ax1.set_title(f'Latency Over Time{title_suffix}')
        ax1.set_ylabel('Latency (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Plot token usage over time
        ax2.plot(df['datetime'], df['prompt_tokens'], label='Prompt Tokens', alpha=0.7)
        ax2.plot(df['datetime'], df['completion_tokens'], label='Completion Tokens', alpha=0.7)
        ax2.plot(df['datetime'], df['total_tokens'], label='Total Tokens', alpha=0.7)
        ax2.set_title(f'Token Usage Over Time{title_suffix}')
        ax2.set_ylabel('Token Count')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Time series plot saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def print_summary(self, stats_data: List[Dict[str, Any]], model_filter: Optional[str] = None):
        """Print a summary of the stats data."""
        if not stats_data:
            print("No stats data available.")
            return
        
        df = pd.DataFrame(stats_data)
        
        print("\n" + "="*60)
        print("CODEWALK STATS SUMMARY")
        if model_filter:
            print(f"Model Filter: {model_filter}")
        print("="*60)
        
        print(f"Total Requests: {len(df)}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        if 'model_name' in df.columns:
            models = df['model_name'].unique()
            print(f"Models Used: {', '.join(models)}")
        
        if 'operation' in df.columns:
            print(f"Operations: {', '.join(df['operation'].unique())}")
        
        print("\nToken Statistics:")
        print(f"  Average Prompt Tokens: {df['prompt_tokens'].mean():.1f}")
        print(f"  Average Completion Tokens: {df['completion_tokens'].mean():.1f}")
        print(f"  Average Total Tokens: {df['total_tokens'].mean():.1f}")
        print(f"  Total Tokens Used: {df['total_tokens'].sum():,}")
        
        print("\nLatency Statistics:")
        print(f"  Average Latency: {df['latency_seconds'].mean():.2f}s")
        print(f"  Median Latency: {df['latency_seconds'].median():.2f}s")
        print(f"  Max Latency: {df['latency_seconds'].max():.2f}s")
        print(f"  Min Latency: {df['latency_seconds'].min():.2f}s")
        
        if 'error' in df.columns:
            error_count = df['error'].notna().sum()
            print(f"\nErrors: {error_count} ({error_count/len(df)*100:.1f}%)")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Analyze CodeWalk stats logs')
    parser.add_argument('--logs-dir', default='logs', 
                       help='Directory containing log files (default: logs)')
    parser.add_argument('--date', 
                       help='Specific date to analyze (YYYYMMDD format)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of recent days to analyze (default: 7)')
    parser.add_argument('--model', 
                       help='Filter by specific model name')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    parser.add_argument('--output-dir', default='.',
                       help='Directory to save plots (default: current directory)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots, only show summary')
    parser.add_argument('--tokens-latency', action='store_true',
                       help='Generate tokens vs latency plot')
    parser.add_argument('--model-comparison', action='store_true',
                       help='Generate model comparison plots')
    parser.add_argument('--time-series', action='store_true',
                       help='Generate time series plots')
    parser.add_argument('--all-plots', action='store_true',
                       help='Generate all available plots')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = StatsAnalyzer(args.logs_dir)
        
        # Load stats data (without model filter for listing models)
        print(f"Loading stats data from {args.logs_dir}...")
        all_stats_data = analyzer.load_stats_data(args.date, args.days)
        
        if not all_stats_data:
            print("No stats data found.")
            return
        
        # List available models if requested
        if args.list_models:
            available_models = analyzer.get_available_models(all_stats_data)
            print(f"\nAvailable models ({len(available_models)}):")
            for model in available_models:
                count = sum(1 for entry in all_stats_data if entry.get('model_name') == model)
                print(f"  - {model} ({count} requests)")
            return
        
        # Apply model filter
        stats_data = analyzer.load_stats_data(args.date, args.days, args.model)
        
        # Print summary
        analyzer.print_summary(stats_data, args.model)
        
        if args.no_plots:
            return
        
        if not PLOTTING_AVAILABLE:
            print("\nPlotting libraries not available. Install with:")
            print("pip install matplotlib seaborn pandas")
            return
        
        # Generate plots
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"_{args.model}" if args.model else ""
        
        if args.tokens_latency or args.all_plots:
            print("\nGenerating tokens vs latency plot...")
            output_file = output_dir / f"tokens_vs_latency{model_suffix}_{timestamp}.png"
            analyzer.create_tokens_vs_latency_plot(stats_data, str(output_file), args.model)
        
        if args.model_comparison or args.all_plots:
            if args.model:
                print("Note: Model comparison plot shows all models, ignoring --model filter")
            print("\nGenerating model comparison plots...")
            output_file = output_dir / f"model_comparison_{timestamp}.png"
            analyzer.create_model_comparison_plot(all_stats_data, str(output_file))
        
        if args.time_series or args.all_plots:
            print("\nGenerating time series plots...")
            output_file = output_dir / f"time_series{model_suffix}_{timestamp}.png"
            analyzer.create_time_series_plot(stats_data, str(output_file), args.model)
        
        if not any([args.tokens_latency, args.model_comparison, args.time_series, args.all_plots]):
            # Default: generate tokens vs latency plot
            print("\nGenerating default tokens vs latency plot...")
            output_file = output_dir / f"tokens_vs_latency{model_suffix}_{timestamp}.png"
            analyzer.create_tokens_vs_latency_plot(stats_data, str(output_file), args.model)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())