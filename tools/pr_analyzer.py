#!/usr/bin/env python3
"""
PR Analyzer for GitHub Repositories - Analyzes and filters significant PRs.

This script fetches PRs from a GitHub repository (including private repos),
uses an LLM to analyze and score them for significance, design, and architectural
elements, then outputs a dataset of significant PRs as a YAML file.
"""

import json
import os
import sys
import argparse
import subprocess
import yaml
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cw.llm.llm_router import llm_router
from cw.llm.llm_model import Message


class PRAnalyzer:
    """Analyzer for GitHub Pull Requests using LLM-based filtering."""

    def __init__(self, repo_url: str, llm_model_type: str = "oai",
                 min_score: float = 7.0, max_prs: Optional[int] = None,
                 state: str = "closed", sort_by: str = "created",
                 cache_dir: str = ".pr_cache", use_cache: bool = True,
                 resume: bool = False, use_llm_cache: bool = True):
        """
        Initialize the PR analyzer.

        Args:
            repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)
            llm_model_type: LLM model to use ('oai', 'claude', 'litellm', etc.)
            min_score: Minimum score (0-10) for PR to be included
            max_prs: Maximum number of PRs to fetch (None = all)
            state: PR state to filter ('open', 'closed', 'merged', 'all')
            sort_by: Sort PRs by ('created', 'updated', 'merged', default: 'created')
            cache_dir: Directory to store cached PR data (default: '.pr_cache')
            use_cache: Whether to use caching (default: True)
            resume: Whether to resume from previous analysis (default: False)
            use_llm_cache: Whether to cache LLM analysis results per PR (default: True)
        """
        self.repo_url = repo_url
        self.repo_owner, self.repo_name = self._parse_repo_url(repo_url)
        self.min_score = min_score
        self.max_prs = max_prs
        self.state = state
        self.sort_by = sort_by
        self.use_cache = use_cache
        self.resume = resume
        self.use_llm_cache = use_llm_cache

        # Setup cache directory and file paths
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file naming: {owner}_{repo}_{state}_{sort_by}.json
        cache_filename = f"{self.repo_owner}_{self.repo_name}_{self.state}_{self.sort_by}.json"
        self.cache_file = self.cache_dir / cache_filename

        # Setup LLM cache directory (per-PR analysis results)
        self.llm_cache_dir = self.cache_dir / "llm_results" / f"{self.repo_owner}_{self.repo_name}"
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM
        self.llm = llm_router.set_model(llm_model_type)
        print(f"Using LLM model: {llm_router.get_current_model_name()}")

    def _parse_repo_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub URL to extract owner and repo name."""
        # Handle various GitHub URL formats
        url = url.rstrip('/')

        if url.startswith('https://github.com/') or url.startswith('http://github.com/'):
            parts = urlparse(url).path.strip('/').split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]
        elif '/' in url and not url.startswith('http'):
            # Assume format: owner/repo
            parts = url.split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]

        raise ValueError(f"Invalid GitHub URL format: {url}. Expected format: https://github.com/owner/repo or owner/repo")

    def _run_gh_command(self, args: List[str]) -> str:
        """Run a GitHub CLI command and return output."""
        try:
            result = subprocess.run(
                ['gh'] + args,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running gh command: {e}")
            print(f"stderr: {e.stderr}")
            raise

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cache from file if it exists."""
        if not self.use_cache or not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            print(f"✓ Loaded cache from {self.cache_file}")
            print(f"  Cache contains: {len(cache_data.get('fetched_prs', []))} fetched PRs, "
                  f"{len(cache_data.get('analyzed_prs', []))} analyzed PRs")
            return cache_data
        except Exception as e:
            print(f"Warning: Could not load cache from {self.cache_file}: {e}")
            return None

    def _save_cache(self, cache_data: Dict[str, Any]):
        """Save cache to file."""
        if not self.use_cache:
            return

        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            print(f"✓ Saved cache to {self.cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")

    def clear_cache(self):
        """Clear the cache file for this repository."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"✓ Cleared cache: {self.cache_file}")
        else:
            print(f"No cache file found: {self.cache_file}")

    def _get_llm_cache_file(self, pr_number: int) -> Path:
        """Get the cache file path for a specific PR's LLM analysis."""
        return self.llm_cache_dir / f"pr_{pr_number}.json"

    def _load_llm_cache(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """Load cached LLM analysis for a specific PR."""
        if not self.use_llm_cache:
            return None

        cache_file = self._get_llm_cache_file(pr_number)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_analysis = json.load(f)
            return cached_analysis
        except Exception as e:
            print(f"Warning: Could not load LLM cache for PR #{pr_number}: {e}")
            return None

    def _save_llm_cache(self, pr_number: int, analysis: Dict[str, Any]):
        """Save LLM analysis results for a specific PR."""
        if not self.use_llm_cache:
            return

        cache_file = self._get_llm_cache_file(pr_number)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'pr_number': pr_number,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save LLM cache for PR #{pr_number}: {e}")

    def clear_llm_cache(self):
        """Clear all LLM cache files for this repository."""
        if self.llm_cache_dir.exists():
            import shutil
            shutil.rmtree(self.llm_cache_dir)
            self.llm_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared LLM cache: {self.llm_cache_dir}")
        else:
            print(f"No LLM cache directory found: {self.llm_cache_dir}")

    def fetch_prs(self) -> List[Dict[str, Any]]:
        """Fetch PRs from the repository using GitHub CLI.

        PRs are sorted based on the sort_by parameter (newest first) to ensure
        we prioritize the most recent PRs when using --max-prs.

        Uses cache if available to avoid re-fetching.
        """
        # Try to load from cache first
        cache_data = self._load_cache()
        if cache_data and 'fetched_prs' in cache_data:
            print(f"Using cached PRs (from {cache_data.get('fetch_timestamp', 'unknown time')})")
            return cache_data['fetched_prs']

        # Not in cache, fetch from GitHub
        print(f"Fetching PRs from {self.repo_owner}/{self.repo_name}...")

        # Build gh pr list command
        fetch_limit = self.max_prs if self.max_prs else 1000
        args = [
            'pr', 'list',
            '--repo', f"{self.repo_owner}/{self.repo_name}",
            '--state', self.state,
            '--json', 'number,title,author,createdAt,mergedAt,closedAt,updatedAt,state,body,labels,files,additions,deletions,changedFiles',
            '--limit', str(fetch_limit)
        ]

        output = self._run_gh_command(args)
        prs = json.loads(output)

        # Sort PRs based on specified criteria (newest first)
        sort_field_map = {
            'created': 'createdAt',
            'updated': 'updatedAt',
            'merged': 'mergedAt',
            'closed': 'closedAt'
        }

        sort_field = sort_field_map.get(self.sort_by, 'createdAt')

        # Sort by the specified field, handling None values for merged/closed dates
        prs.sort(
            key=lambda pr: pr.get(sort_field) or '',
            reverse=True
        )

        print(f"Fetched {len(prs)} PRs (sorted by {self.sort_by} date, newest first)")

        # Save fetched PRs to cache
        if self.use_cache:
            cache_data = {
                'fetched_prs': prs,
                'analyzed_prs': [],
                'fetch_timestamp': datetime.now().isoformat(),
                'repo_owner': self.repo_owner,
                'repo_name': self.repo_name,
                'state': self.state,
                'sort_by': self.sort_by,
                'max_prs': self.max_prs
            }
            self._save_cache(cache_data)

        return prs

    def fetch_pr_details(self, pr_number: int) -> Dict[str, Any]:
        """Fetch detailed information about a specific PR."""
        print(f"Fetching details for PR #{pr_number}...")

        # Get PR details
        args = [
            'pr', 'view', str(pr_number),
            '--repo', f"{self.repo_owner}/{self.repo_name}",
            '--json', 'number,title,body,author,createdAt,mergedAt,closedAt,state,labels,files,additions,deletions,changedFiles,commits,reviews'
        ]

        output = self._run_gh_command(args)
        pr_details = json.loads(output)

        # Get diff (sample first 50KB to avoid huge diffs)
        try:
            diff_args = [
                'pr', 'diff', str(pr_number),
                '--repo', f"{self.repo_owner}/{self.repo_name}"
            ]
            diff_output = self._run_gh_command(diff_args)

            # Limit diff size
            max_diff_size = 50000  # 50KB
            if len(diff_output) > max_diff_size:
                diff_output = diff_output[:max_diff_size] + f"\n\n... (diff truncated, original size: {len(diff_output)} bytes)"

            pr_details['diff'] = diff_output
        except Exception as e:
            print(f"Warning: Could not fetch diff for PR #{pr_number}: {e}")
            pr_details['diff'] = ""

        return pr_details

    def analyze_pr_with_llm(self, pr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze a PR and score it for significance.

        Checks LLM cache first to avoid redundant API calls.

        Returns a dict with:
            - significance_score (0-10)
            - feature_summary
            - solution_summary
            - architectural_elements
            - reasoning
        """
        pr_number = pr['number']

        # Check LLM cache first
        cached_result = self._load_llm_cache(pr_number)
        if cached_result:
            print(f"Using cached LLM analysis for PR #{pr_number}: {pr['title']}")
            return cached_result['analysis']

        print(f"Analyzing PR #{pr_number}: {pr['title']}")

        # Prepare context for LLM
        pr_context = self._prepare_pr_context(pr)

        # Create analysis prompt
        prompt = f"""Analyze this GitHub Pull Request and provide a detailed assessment.

PR Information:
{pr_context}

Please provide a JSON response with the following fields:

1. significance_score (0-10): Rate the significance of this PR based on:
   - Feature complexity and scope
   - Architectural/design elements introduced or modified
   - Impact on codebase structure
   - Innovation and problem-solving approach
   Score 0-3: Minor changes, bug fixes, simple features
   Score 4-6: Moderate features, some design considerations
   Score 7-8: Significant features with architectural elements
   Score 9-10: Major features with substantial design/architectural work

2. feature_summary (string): 2-3 sentence summary of what feature/capability this PR adds

3. solution_summary (string): 2-3 sentence summary of how the solution was implemented

4. architectural_elements (list of strings): Key architectural or design elements involved (e.g., "new service layer", "database schema migration", "API redesign", "state management refactoring")

5. technical_highlights (list of strings): Notable technical aspects (e.g., "uses lazy loading", "implements caching strategy", "adds type safety")

6. reasoning (string): Brief explanation of the significance score

Respond ONLY with valid JSON, no additional text."""

        # Call LLM
        messages = [Message(role="user", content=prompt)]

        try:
            response = self.llm.complete(messages, temperature=0.3)
            response_text = response.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if response_text.startswith('```'):
                # Remove markdown code block markers
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            analysis = json.loads(response_text)

            # Validate required fields
            required_fields = ['significance_score', 'feature_summary', 'solution_summary',
                             'architectural_elements', 'reasoning']
            for field in required_fields:
                if field not in analysis:
                    print(f"Warning: Missing field '{field}' in LLM response for PR #{pr_number}")
                    analysis[field] = "" if field != 'significance_score' else 0

            # Save to LLM cache
            self._save_llm_cache(pr_number, analysis)

            return analysis

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response for PR #{pr_number}: {e}")
            print(f"Response was: {response_text[:500]}")
            return {
                'significance_score': 0,
                'feature_summary': "Error analyzing PR",
                'solution_summary': "",
                'architectural_elements': [],
                'technical_highlights': [],
                'reasoning': f"JSON parse error: {str(e)}"
            }
        except Exception as e:
            print(f"Error analyzing PR #{pr_number}: {e}")
            traceback.print_exc()
            raise e
            return {
                'significance_score': 0,
                'feature_summary': "Error analyzing PR",
                'solution_summary': "",
                'architectural_elements': [],
                'technical_highlights': [],
                'reasoning': f"Error: {str(e)}"
            }

    def _prepare_pr_context(self, pr: Dict[str, Any]) -> str:
        """Prepare PR information as context for LLM."""
        context_parts = [
            f"Title: {pr.get('title', 'N/A')}",
            f"Number: #{pr.get('number', 'N/A')}",
            f"Author: {pr.get('author', {}).get('login', 'N/A')}",
            f"State: {pr.get('state', 'N/A')}",
            f"",
            f"Description:",
            pr.get('body', 'No description provided'),
            f"",
            f"Files Changed: {pr.get('changedFiles', 0)}",
            f"Additions: +{pr.get('additions', 0)}",
            f"Deletions: -{pr.get('deletions', 0)}",
        ]

        # Add labels if present
        labels = pr.get('labels', [])
        if labels:
            label_names = [label.get('name', '') for label in labels]
            context_parts.append(f"Labels: {', '.join(label_names)}")

        # Add file list
        files = pr.get('files', [])
        if files:
            context_parts.append(f"\nFiles Changed ({len(files)}):")
            for i, file in enumerate(files[:30]):  # Limit to first 30 files
                path = file.get('path', 'unknown')
                additions = file.get('additions', 0)
                deletions = file.get('deletions', 0)
                context_parts.append(f"  - {path} (+{additions} -{deletions})")
            if len(files) > 30:
                context_parts.append(f"  ... and {len(files) - 30} more files")

        # Add diff sample (first 5000 chars)
        diff = pr.get('diff', '')
        if diff:
            diff_sample = diff[:5000]
            context_parts.append(f"\nDiff Sample (first 5000 chars):")
            context_parts.append(diff_sample)
            if len(diff) > 5000:
                context_parts.append(f"... (diff continues, total size: {len(diff)} chars)")

        return '\n'.join(context_parts)

    def analyze_all_prs(self, prs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze all PRs and return filtered, scored results.

        Supports resume mode to continue from previously analyzed PRs.
        """
        # Load existing analyzed PRs from cache if resuming
        cache_data = self._load_cache() if self.resume else None
        analyzed_prs = []
        analyzed_pr_numbers = set()

        if cache_data and 'analyzed_prs' in cache_data:
            analyzed_prs = cache_data['analyzed_prs']
            analyzed_pr_numbers = {pr['pr_number'] for pr in analyzed_prs}
            print(f"Resuming from cache: {len(analyzed_prs)} PRs already analyzed")

        # Track PRs to analyze (skip already analyzed ones)
        prs_to_analyze = [pr for pr in prs if pr['number'] not in analyzed_pr_numbers]

        if not prs_to_analyze:
            print("All PRs have already been analyzed!")
            # Apply min_score filter to cached results
            filtered_prs = [pr for pr in analyzed_prs if pr['significance_score'] >= self.min_score]
            filtered_prs.sort(key=lambda x: x['significance_score'], reverse=True)
            return filtered_prs

        print(f"Analyzing {len(prs_to_analyze)} PRs ({len(analyzed_pr_numbers)} already in cache)")

        for i, pr in enumerate(prs_to_analyze, 1):
            print(f"\n[{i}/{len(prs_to_analyze)}] Processing PR #{pr['number']}")

            try:
                # Fetch detailed PR info including diff
                detailed_pr = self.fetch_pr_details(pr['number'])

                # Analyze with LLM
                analysis = self.analyze_pr_with_llm(detailed_pr)

                # Combine PR data with analysis
                result = {
                    'pr_number': pr['number'],
                    'title': pr['title'],
                    'url': f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{pr['number']}",
                    'author': pr.get('author', {}).get('login', 'N/A'),
                    'state': pr.get('state', 'N/A'),
                    'created_at': pr.get('createdAt', 'N/A'),
                    'merged_at': pr.get('mergedAt'),
                    'closed_at': pr.get('closedAt'),
                    'files_changed': pr.get('changedFiles', 0),
                    'additions': pr.get('additions', 0),
                    'deletions': pr.get('deletions', 0),
                    'labels': [label.get('name', '') for label in pr.get('labels', [])],
                    'significance_score': analysis.get('significance_score', 0),
                    'feature_summary': analysis.get('feature_summary', ''),
                    'solution_summary': analysis.get('solution_summary', ''),
                    'architectural_elements': analysis.get('architectural_elements', []),
                    'technical_highlights': analysis.get('technical_highlights', []),
                    'analysis_reasoning': analysis.get('reasoning', ''),
                    'description': pr.get('body', ''),
                    'files': [
                        {
                            'path': f.get('path', ''),
                            'additions': f.get('additions', 0),
                            'deletions': f.get('deletions', 0)
                        }
                        for f in pr.get('files', [])
                    ]
                }

                # Add to analyzed list (regardless of score for cache)
                analyzed_prs.append(result)
                print(f"  ✓ Score: {result['significance_score']}/10")

                # Save to cache after each analysis (incremental save for resume)
                if self.use_cache:
                    cache_update = {
                        'fetched_prs': prs,
                        'analyzed_prs': analyzed_prs,
                        'fetch_timestamp': cache_data.get('fetch_timestamp') if cache_data else datetime.now().isoformat(),
                        'last_analysis_timestamp': datetime.now().isoformat(),
                        'repo_owner': self.repo_owner,
                        'repo_name': self.repo_name,
                        'state': self.state,
                        'sort_by': self.sort_by,
                        'max_prs': self.max_prs
                    }
                    self._save_cache(cache_update)

            except Exception as e:
                print(f"  Error processing PR #{pr['number']}: {e}")
                continue

        # Filter by minimum score and sort
        filtered_prs = [pr for pr in analyzed_prs if pr['significance_score'] >= self.min_score]
        filtered_prs.sort(key=lambda x: x['significance_score'], reverse=True)

        print(f"\n✓ Analysis complete: {len(analyzed_prs)} total analyzed, {len(filtered_prs)} meet threshold (>= {self.min_score})")

        return filtered_prs

    def save_to_yaml(self, analyzed_prs: List[Dict[str, Any]], output_file: str):
        """Save analyzed PRs to a YAML file."""
        output_path = Path(output_file)

        # Prepare output structure
        output_data = {
            'repository': f"{self.repo_owner}/{self.repo_name}",
            'repository_url': self.repo_url,
            'analysis_date': datetime.now().isoformat(),
            'filter_criteria': {
                'min_significance_score': self.min_score,
                'pr_state': self.state,
                'total_prs_analyzed': len(analyzed_prs)
            },
            'significant_prs': analyzed_prs
        }

        # Write to YAML
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"\n✓ Saved {len(analyzed_prs)} significant PRs to {output_file}")

    def generate_summary_report(self, analyzed_prs: List[Dict[str, Any]]):
        """Print a summary report of the analysis."""
        if not analyzed_prs:
            print("\nNo PRs met the significance threshold.")
            return

        print("\n" + "="*70)
        print("PR ANALYSIS SUMMARY")
        print("="*70)
        print(f"Repository: {self.repo_owner}/{self.repo_name}")
        print(f"Total Significant PRs: {len(analyzed_prs)}")
        print(f"Minimum Score Threshold: {self.min_score}/10")
        print()

        # Score distribution
        score_ranges = {'9-10': 0, '7-8': 0, '5-6': 0, '3-4': 0, '0-2': 0}
        for pr in analyzed_prs:
            score = pr['significance_score']
            if score >= 9:
                score_ranges['9-10'] += 1
            elif score >= 7:
                score_ranges['7-8'] += 1
            elif score >= 5:
                score_ranges['5-6'] += 1
            elif score >= 3:
                score_ranges['3-4'] += 1
            else:
                score_ranges['0-2'] += 1

        print("Score Distribution:")
        for range_label, count in score_ranges.items():
            if count > 0:
                print(f"  {range_label}: {count} PRs")
        print()

        # Top PRs
        print("Top 10 Most Significant PRs:")
        for i, pr in enumerate(analyzed_prs[:10], 1):
            print(f"\n{i}. PR #{pr['pr_number']}: {pr['title']}")
            print(f"   Score: {pr['significance_score']}/10")
            print(f"   Feature: {pr['feature_summary'][:100]}...")
            print(f"   URL: {pr['url']}")

        print("="*70)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Analyze GitHub PRs for significant feature development',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a public repository (most recently created PRs first)
  python pr_analyzer.py https://github.com/owner/repo

  # Analyze with custom settings
  python pr_analyzer.py owner/repo --model claude --min-score 8 --max-prs 50

  # Resume from previous analysis (skip already analyzed PRs, reuse LLM cache)
  python pr_analyzer.py owner/repo --resume

  # Start fresh (clear all caches and re-analyze)
  python pr_analyzer.py owner/repo --clear-cache --clear-llm-cache

  # Re-analyze with different threshold (uses LLM cache, no new API calls!)
  python pr_analyzer.py owner/repo --min-score 8.0

  # Force fresh LLM analysis (ignore LLM cache)
  python pr_analyzer.py owner/repo --no-llm-cache

  # Disable all caching (always fetch fresh data)
  python pr_analyzer.py owner/repo --no-cache --no-llm-cache

  # Analyze merged PRs only, sorted by merge date
  python pr_analyzer.py owner/repo --state merged --sort-by merged --output merged_prs.yaml

  # Get most recently updated PRs
  python pr_analyzer.py owner/repo --sort-by updated --max-prs 20
        """
    )

    parser.add_argument('repo_url',
                       help='GitHub repository URL (https://github.com/owner/repo) or owner/repo')
    parser.add_argument('--model', default='oai',
                       choices=['oai', 'claude', 'litellm', 'llama', 'azure_oai', 'codex'],
                       help='LLM model to use for analysis (default: oai)')
    parser.add_argument('--min-score', type=float, default=7.0,
                       help='Minimum significance score (0-10) to include PR (default: 7.0)')
    parser.add_argument('--max-prs', type=int,
                       help='Maximum number of PRs to fetch (default: all)')
    parser.add_argument('--state', default='closed',
                       choices=['open', 'closed', 'merged', 'all'],
                       help='PR state to filter (default: closed)')
    parser.add_argument('--sort-by', default='created',
                       choices=['created', 'updated', 'merged', 'closed'],
                       help='Sort PRs by date field (default: created). Newest first.')
    parser.add_argument('--cache-dir', default='.pr_cache',
                       help='Directory to store cache files (default: .pr_cache)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (always fetch fresh data)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous analysis (skip already analyzed PRs)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache for this repository before running')
    parser.add_argument('--no-llm-cache', action='store_true',
                       help='Disable LLM result caching (always call LLM for analysis)')
    parser.add_argument('--clear-llm-cache', action='store_true',
                       help='Clear LLM cache for this repository before running')
    parser.add_argument('--output', '-o',
                       help='Output YAML file path (default: prs_{repo_name}_{timestamp}.yaml)')

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = PRAnalyzer(
            repo_url=args.repo_url,
            llm_model_type=args.model,
            min_score=args.min_score,
            max_prs=args.max_prs,
            state=args.state,
            sort_by=args.sort_by,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            resume=args.resume,
            use_llm_cache=not args.no_llm_cache
        )

        # Clear cache if requested
        if args.clear_cache:
            analyzer.clear_cache()

        # Clear LLM cache if requested
        if args.clear_llm_cache:
            analyzer.clear_llm_cache()

        # Fetch PRs
        prs = analyzer.fetch_prs()

        if not prs:
            print("No PRs found.")
            return 1

        # Analyze PRs
        print(f"\nStarting LLM analysis of {len(prs)} PRs...")
        analyzed_prs = analyzer.analyze_all_prs(prs)

        # Generate output filename if not specified
        if not args.output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"prs_{analyzer.repo_name}_{timestamp}.yaml"

        # Save results
        analyzer.save_to_yaml(analyzed_prs, args.output)

        # Print summary
        analyzer.generate_summary_report(analyzed_prs)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
