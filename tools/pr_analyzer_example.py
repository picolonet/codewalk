#!/usr/bin/env python3
"""
Example script showing how to use the PR Analyzer programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.pr_analyzer import PRAnalyzer


def example_basic_analysis():
    """Basic example: Analyze a repository and save results."""
    print("Example 1: Basic Analysis")
    print("-" * 50)

    # Initialize analyzer
    analyzer = PRAnalyzer(
        repo_url="owner/repo",  # Replace with actual repo
        llm_model_type="oai",   # Use OpenAI GPT-4
        min_score=7.0,          # Only include PRs scored 7.0 or higher
        max_prs=10,             # Analyze first 10 PRs
        state="merged"          # Only merged PRs
    )

    # Fetch PRs
    prs = analyzer.fetch_prs()
    print(f"Fetched {len(prs)} PRs")

    # Analyze PRs
    analyzed_prs = analyzer.analyze_all_prs(prs)
    print(f"Found {len(analyzed_prs)} significant PRs")

    # Save to YAML
    analyzer.save_to_yaml(analyzed_prs, "example_output.yaml")

    # Print summary
    analyzer.generate_summary_report(analyzed_prs)


def example_custom_filtering():
    """Advanced example: Custom filtering and processing."""
    print("\nExample 2: Custom Filtering")
    print("-" * 50)

    analyzer = PRAnalyzer(
        repo_url="https://github.com/owner/repo",
        llm_model_type="claude",  # Use Anthropic Claude
        min_score=8.0,            # Higher threshold
        max_prs=50,
        state="closed"
    )

    prs = analyzer.fetch_prs()
    analyzed_prs = analyzer.analyze_all_prs(prs)

    # Custom filtering: Only PRs with specific architectural elements
    architectural_prs = [
        pr for pr in analyzed_prs
        if any('database' in elem.lower() or 'api' in elem.lower()
               for elem in pr.get('architectural_elements', []))
    ]

    print(f"\nFound {len(architectural_prs)} PRs with database/API architectural elements")

    # Save filtered results
    analyzer.save_to_yaml(architectural_prs, "architectural_prs.yaml")


def example_batch_analysis():
    """Example: Analyze multiple repositories."""
    print("\nExample 3: Batch Analysis")
    print("-" * 50)

    repos = [
        "owner/repo1",
        "owner/repo2",
        "owner/repo3"
    ]

    for repo in repos:
        print(f"\nAnalyzing {repo}...")

        try:
            analyzer = PRAnalyzer(
                repo_url=repo,
                llm_model_type="oai",
                min_score=7.0,
                max_prs=20,
                state="merged"
            )

            prs = analyzer.fetch_prs()
            analyzed_prs = analyzer.analyze_all_prs(prs)

            # Save with repo-specific filename
            output_file = f"prs_{repo.replace('/', '_')}.yaml"
            analyzer.save_to_yaml(analyzed_prs, output_file)

            print(f"✓ Completed analysis for {repo}: {len(analyzed_prs)} significant PRs")

        except Exception as e:
            print(f"✗ Error analyzing {repo}: {e}")


def example_analyze_single_pr():
    """Example: Analyze a single specific PR."""
    print("\nExample 4: Single PR Analysis")
    print("-" * 50)

    analyzer = PRAnalyzer(
        repo_url="owner/repo",
        llm_model_type="oai"
    )

    # Fetch details for a specific PR
    pr_details = analyzer.fetch_pr_details(pr_number=123)  # Replace with actual PR number

    # Analyze it
    analysis = analyzer.analyze_pr_with_llm(pr_details)

    print(f"PR #{pr_details['number']}: {pr_details['title']}")
    print(f"Significance Score: {analysis['significance_score']}/10")
    print(f"Feature: {analysis['feature_summary']}")
    print(f"Solution: {analysis['solution_summary']}")
    print(f"Architectural Elements: {', '.join(analysis['architectural_elements'])}")


if __name__ == "__main__":
    # Run examples (comment out as needed)

    # example_basic_analysis()
    # example_custom_filtering()
    # example_batch_analysis()
    # example_analyze_single_pr()

    print("\n" + "="*50)
    print("Example script - uncomment functions to run")
    print("="*50)
    print("\nAvailable examples:")
    print("  1. example_basic_analysis() - Basic repository analysis")
    print("  2. example_custom_filtering() - Custom filtering by architectural elements")
    print("  3. example_batch_analysis() - Analyze multiple repositories")
    print("  4. example_analyze_single_pr() - Analyze a specific PR")
    print("\nEdit this file and uncomment the example you want to run.")
