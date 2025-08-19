#!/usr/bin/env python3
"""
Usage Analyzer for CodeWalk - Analyzes usage statistics from stats logs.

This script reads stats log files and computes total input/output tokens,
total latency, and cost analysis with filtering capabilities.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


class UsageAnalyzer:
    """Analyzer for CodeWalk usage statistics."""
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize the usage analyzer.
        
        Args:
            logs_dir: Directory containing the log files
        """
        self.logs_dir = Path(logs_dir)
        if not self.logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    
    def load_stats_data(self, log_file: Optional[str] = None, 
                       filter_string: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load stats data from log files.
        
        Args:
            log_file: Specific log file path, or None to use latest
            filter_string: Optional string to filter entries by model_name
            
        Returns:
            List of stat entries as dictionaries
        """
        stats_data = []
        
        if log_file:
            # Use specified log file
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = self.logs_dir / log_file
            if log_path.exists():
                stats_data.extend(self._load_single_stats_file(log_path))
            else:
                raise FileNotFoundError(f"Log file not found: {log_file}")
        else:
            # Use the latest stats log file
            stats_files = list(self.logs_dir.glob("stats_*.log"))
            if not stats_files:
                raise FileNotFoundError(f"No stats log files found in {self.logs_dir}")
            
            # Get the most recent file
            latest_file = max(stats_files, key=lambda f: f.stat().st_mtime)
            print(f"Using latest log file: {latest_file.name}")
            stats_data.extend(self._load_single_stats_file(latest_file))
        
        # Filter by string if specified
        if filter_string:
            original_count = len(stats_data)
            stats_data = [entry for entry in stats_data 
                         if filter_string.lower() in json.dumps(entry).lower()]
            print(f"Filtered from {original_count} to {len(stats_data)} entries matching '{filter_string}'")
        
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
    
    def compute_usage_stats(self, stats_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute usage statistics from the stats data.
        
        Args:
            stats_data: List of stat entries
            
        Returns:
            Dictionary with computed statistics
        """
        if not stats_data:
            return {
                'total_requests': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_latency_seconds': 0,
                'average_latency_seconds': 0,
                'error_count': 0
            }
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_latency = 0
        error_count = 0
        
        for entry in stats_data:
            # Count tokens
            prompt_tokens = entry.get('prompt_tokens', 0) or 0
            completion_tokens = entry.get('completion_tokens', 0) or 0
            entry_total_tokens = entry.get('total_tokens', 0) or 0
            
            total_input_tokens += prompt_tokens
            total_output_tokens += completion_tokens
            total_tokens += entry_total_tokens
            
            # Count latency
            latency = entry.get('latency_seconds', 0) or 0
            total_latency += latency
            
            # Count errors
            if entry.get('error') is not None:
                error_count += 1
        
        return {
            'total_requests': len(stats_data),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'total_latency_seconds': total_latency,
            'average_latency_seconds': total_latency / len(stats_data) if stats_data else 0,
            'error_count': error_count,
            'error_rate_percent': (error_count / len(stats_data) * 100) if stats_data else 0
        }
    
    def compute_costs(self, stats: Dict[str, Any], 
                     input_cost_per_mil: float, 
                     output_cost_per_mil: float) -> Dict[str, float]:
        """
        Compute costs based on token usage and per-million-token rates.
        
        Args:
            stats: Statistics dictionary from compute_usage_stats
            input_cost_per_mil: Cost per million input tokens
            output_cost_per_mil: Cost per million output tokens
            
        Returns:
            Dictionary with cost calculations
        """
        input_cost = (stats['total_input_tokens'] / 1_000_000) * input_cost_per_mil
        output_cost = (stats['total_output_tokens'] / 1_000_000) * output_cost_per_mil
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'cost_per_request': total_cost / stats['total_requests'] if stats['total_requests'] > 0 else 0
        }
    
    def print_usage_report(self, stats_data: List[Dict[str, Any]], 
                          input_cost_per_mil: Optional[float] = None,
                          output_cost_per_mil: Optional[float] = None,
                          filter_string: Optional[str] = None):
        """Print a comprehensive usage report."""
        if not stats_data:
            print("No data available for analysis.")
            return
        
        stats = self.compute_usage_stats(stats_data)
        
        print("\n" + "="*70)
        print("CODEWALK USAGE ANALYSIS")
        if filter_string:
            print(f"Filter: {filter_string}")
        print("="*70)
        
        # Basic statistics
        print(f"\nRequest Statistics:")
        print(f"  Total Requests: {stats['total_requests']:,}")
        print(f"  Successful Requests: {stats['total_requests'] - stats['error_count']:,}")
        print(f"  Failed Requests: {stats['error_count']:,} ({stats['error_rate_percent']:.1f}%)")
        
        # Token statistics
        print(f"\nToken Usage:")
        print(f"  Total Input Tokens: {stats['total_input_tokens']:,}")
        print(f"  Total Output Tokens: {stats['total_output_tokens']:,}")
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        print(f"  Average Input Tokens/Request: {stats['total_input_tokens'] / stats['total_requests']:.1f}")
        print(f"  Average Output Tokens/Request: {stats['total_output_tokens'] / stats['total_requests']:.1f}")
        print(f"  Average Total Tokens/Request: {stats['total_tokens'] / stats['total_requests']:.1f}")
        
        # Latency statistics
        print(f"\nLatency Statistics:")
        print(f"  Total Latency: {stats['total_latency_seconds']:.2f} seconds")
        print(f"  Average Latency: {stats['average_latency_seconds']:.2f} seconds")
        print(f"  Total Processing Time: {stats['total_latency_seconds'] / 60:.1f} minutes")
        
        # Cost analysis if provided
        if input_cost_per_mil is not None and output_cost_per_mil is not None:
            costs = self.compute_costs(stats, input_cost_per_mil, output_cost_per_mil)
            
            print(f"\nCost Analysis:")
            print(f"  Input Token Cost (${input_cost_per_mil:.2f}/1M): ${costs['input_cost']:.4f}")
            print(f"  Output Token Cost (${output_cost_per_mil:.2f}/1M): ${costs['output_cost']:.4f}")
            print(f"  Total Cost: ${costs['total_cost']:.4f}")
            print(f"  Average Cost per Request: ${costs['cost_per_request']:.6f}")
        
        # Model breakdown if available
        model_breakdown = {}
        for entry in stats_data:
            model = entry.get('model_name', 'unknown')
            if model not in model_breakdown:
                model_breakdown[model] = {'count': 0, 'tokens': 0, 'latency': 0}
            model_breakdown[model]['count'] += 1
            model_breakdown[model]['tokens'] += entry.get('total_tokens', 0) or 0
            model_breakdown[model]['latency'] += entry.get('latency_seconds', 0) or 0
        
        if len(model_breakdown) > 1:
            print(f"\nModel Breakdown:")
            for model, data in sorted(model_breakdown.items()):
                print(f"  {model}:")
                print(f"    Requests: {data['count']:,}")
                print(f"    Tokens: {data['tokens']:,}")
                print(f"    Avg Latency: {data['latency'] / data['count']:.2f}s")
        
        print("="*70)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Analyze CodeWalk usage statistics')
    parser.add_argument('--log-file', 
                       help='Specific log file to analyze (default: latest in logs/)')
    parser.add_argument('--logs-dir', default='logs',
                       help='Directory containing log files (default: logs)')
    parser.add_argument('--filter', dest='filter_string',
                       help='Filter entries by model name (case-insensitive substring match)')
    parser.add_argument('--input-cost', type=float,
                       help='Cost per million input tokens (e.g., 3.00 for $3/1M tokens)')
    parser.add_argument('--output-cost', type=float,
                       help='Cost per million output tokens (e.g., 15.00 for $15/1M tokens)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = UsageAnalyzer(args.logs_dir)
        
        # Load stats data
        print(f"Loading stats data...")
        stats_data = analyzer.load_stats_data(args.log_file, args.filter_string)
        
        if not stats_data:
            print("No stats data found.")
            return 1
        
        # Print report
        analyzer.print_usage_report(
            stats_data, 
            args.input_cost, 
            args.output_cost, 
            args.filter_string
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())