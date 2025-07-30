#!/usr/bin/env python3
"""
Convert stats logs to CSV files for analysis.
Creates three CSV files: prompt_tokens vs latency, completion_tokens vs latency, and total_tokens vs latency.
"""

import json
import csv
import argparse
import os
import glob
from datetime import datetime


def parse_log_file(log_file_path, model_filter=None):
    """Parse a single log file and extract relevant data."""
    data_points = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Skip if model filter is specified and doesn't match
                    if model_filter and data.get('model_name') != model_filter:
                        continue
                    
                    # Extract required fields
                    if all(field in data for field in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'latency_seconds']):
                        data_points.append({
                            'prompt_tokens': data['prompt_tokens'],
                            'completion_tokens': data['completion_tokens'],
                            'total_tokens': data['total_tokens'],
                            'latency_seconds': data['latency_seconds'],
                            'model_name': data.get('model_name', 'unknown'),
                            'timestamp': data.get('timestamp', '')
                        })
                        
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                    
    except FileNotFoundError:
        print(f"Warning: Log file {log_file_path} not found")
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    
    return data_points


def write_csv_files(data_points, output_dir="./csv/", model_filter=None):
    """Write three CSV files with token vs latency data."""
    if not data_points:
        print("No data points found to write to CSV files")
        return
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = f"_{model_filter}" if model_filter else "_all"
    
    # Define the three CSV files to create
    csv_files = [
        {
            'filename': f"prompt_tokens_vs_latency{model_suffix}_{timestamp}.csv",
            'token_field': 'prompt_tokens',
            'headers': ['prompt_tokens', 'latency_seconds', 'model_name', 'timestamp']
        },
        {
            'filename': f"completion_tokens_vs_latency{model_suffix}_{timestamp}.csv",
            'token_field': 'completion_tokens',
            'headers': ['completion_tokens', 'latency_seconds', 'model_name', 'timestamp']
        },
        {
            'filename': f"total_tokens_vs_latency{model_suffix}_{timestamp}.csv",
            'token_field': 'total_tokens',
            'headers': ['total_tokens', 'latency_seconds', 'model_name', 'timestamp']
        }
    ]
    
    for csv_config in csv_files:
        output_path = os.path.join(output_dir, csv_config['filename'])
        
        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(csv_config['headers'])
                
                # Write data rows
                for point in data_points:
                    writer.writerow([
                        point[csv_config['token_field']],
                        point['latency_seconds'],
                        point['model_name'],
                        point['timestamp']
                    ])
            
            print(f"Created: {output_path} ({len(data_points)} data points)")
            
        except Exception as e:
            print(f"Error writing {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert stats logs to CSV files for token vs latency analysis"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Filter data points by model name (e.g., "lite_llm", "anthropic"). If not specified, includes all models.'
    )
    parser.add_argument(
        '--logs-dir', '-l',
        type=str,
        default='../logs',
        help='Directory containing log files (default: ../logs)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for CSV files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Find all stats log files
    logs_dir = os.path.abspath(args.logs_dir)
    stats_pattern = os.path.join(logs_dir, "stats_*.log")
    log_files = glob.glob(stats_pattern)
    
    if not log_files:
        print(f"No stats log files found in {logs_dir}")
        print(f"Looking for files matching pattern: {stats_pattern}")
        return
    
    print(f"Found {len(log_files)} log file(s) in {logs_dir}")
    if args.model:
        print(f"Filtering for model: {args.model}")
    
    # Parse all log files
    all_data_points = []
    for log_file in log_files:
        print(f"Processing: {log_file}")
        data_points = parse_log_file(log_file, args.model)
        all_data_points.extend(data_points)
    
    print(f"Total data points collected: {len(all_data_points)}")
    
    # Write CSV files
    if all_data_points:
        write_csv_files(all_data_points, args.output_dir, args.model)
    else:
        print("No matching data points found")


if __name__ == "__main__":
    main()