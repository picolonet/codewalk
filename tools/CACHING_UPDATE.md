# PR Analyzer - Caching & Resume Feature

## Overview

Added intelligent caching to the PR analyzer tool to save time, reduce API calls, and enable resuming long-running analyses.

## Key Features

### 1. **Local PR Storage**
- Fetched PRs are stored locally in `.pr_cache/` directory
- Cache files are named: `{owner}_{repo}_{state}_{sort_by}.json`
- Avoids redundant GitHub API calls

### 2. **Resume Functionality**
- Continue analysis from where you left off
- Already-analyzed PRs are skipped automatically
- Saves LLM costs on interrupted runs

### 3. **Incremental Saving**
- Cache is updated after each PR analysis
- Safe to interrupt and resume at any time
- No progress is lost

### 4. **Flexible Cache Control**
- `--resume`: Continue from previous analysis
- `--clear-cache`: Start fresh
- `--no-cache`: Disable caching entirely
- `--cache-dir`: Specify custom cache location

## Usage Examples

### Basic Workflow

```bash
# First run: Analyze 100 PRs (fetches from GitHub and creates cache)
python tools/pr_analyzer.py owner/repo --max-prs 100

# Second run: Uses cached PRs (no GitHub API calls)
python tools/pr_analyzer.py owner/repo --max-prs 100
# Output: "Using cached PRs (from 2025-12-02T18:00:00)"
```

### Resume After Interruption

```bash
# Start analyzing 200 PRs
python tools/pr_analyzer.py owner/repo --max-prs 200

# ... Process interrupted after 50 PRs ...
# Press Ctrl+C

# Resume from where you left off (skips first 50 PRs)
python tools/pr_analyzer.py owner/repo --max-prs 200 --resume
# Output: "Resuming from cache: 50 PRs already analyzed"
# Output: "Analyzing 150 PRs (50 already in cache)"
```

### Fetch Latest PRs

```bash
# Clear cache and fetch latest PRs from GitHub
python tools/pr_analyzer.py owner/repo --clear-cache
```

### Disable Caching

```bash
# Always fetch fresh data (no caching)
python tools/pr_analyzer.py owner/repo --no-cache
```

## Cache Structure

### Cache File Location
```
.pr_cache/
  ├── owner_repo_closed_created.json       # Closed PRs, sorted by created date
  ├── owner_repo_merged_merged.json        # Merged PRs, sorted by merge date
  └── owner_repo_all_updated.json          # All PRs, sorted by updated date
```

### Cache File Contents

```json
{
  "fetched_prs": [
    {
      "number": 123,
      "title": "Add new feature",
      "state": "merged",
      // ... complete PR data from GitHub
    }
  ],
  "analyzed_prs": [
    {
      "pr_number": 123,
      "significance_score": 8.5,
      "feature_summary": "...",
      "solution_summary": "...",
      "architectural_elements": ["..."],
      // ... complete analysis results
    }
  ],
  "fetch_timestamp": "2025-12-02T18:00:00",
  "last_analysis_timestamp": "2025-12-02T18:30:00",
  "repo_owner": "owner",
  "repo_name": "repo",
  "state": "closed",
  "sort_by": "created",
  "max_prs": 100
}
```

## Technical Implementation

### Modified Functions

1. **`__init__`**: Added cache parameters
   - `cache_dir`: Cache directory path
   - `use_cache`: Enable/disable caching
   - `resume`: Resume from previous analysis

2. **`_load_cache()`**: Load cache from disk
   - Returns cached data if available
   - Displays cache stats

3. **`_save_cache()`**: Save cache to disk
   - Saves after each PR analysis (incremental)
   - Stores both fetched PRs and analysis results

4. **`clear_cache()`**: Clear cache file
   - Removes cache file for current repo/state/sort

5. **`fetch_prs()`**: Modified to use cache
   - Checks cache first
   - Falls back to GitHub API if cache missing
   - Saves fetched PRs to cache

6. **`analyze_all_prs()`**: Modified to support resume
   - Loads analyzed PRs from cache when `--resume` is used
   - Skips already-analyzed PRs
   - Updates cache after each analysis
   - Returns combined results (cached + new)

### Cache Key Strategy

Cache files are keyed by: `{owner}_{repo}_{state}_{sort_by}`

**Why?**
- Different `--state` values need separate caches (open vs closed PRs)
- Different `--sort-by` values may return different PR sets
- This ensures cache correctness across different query parameters

**Example:**
```bash
# These use different cache files:
--state closed --sort-by created  → owner_repo_closed_created.json
--state merged --sort-by merged   → owner_repo_merged_merged.json
--state all --sort-by updated     → owner_repo_all_updated.json
```

## Benefits

### 1. **Save Time**
- No need to re-fetch PRs from GitHub
- Typical fetch time: 5-10 seconds → 0 seconds (cached)

### 2. **Save API Calls**
- GitHub API has rate limits
- Cached data doesn't count against limits

### 3. **Save Money**
- LLM analysis is the expensive part
- Resume skips already-analyzed PRs
- Example: 100 PRs @ $0.01/PR = $1.00 saved on resume

### 4. **Reliability**
- Interrupt and resume safely
- No lost progress
- Incremental saves ensure data integrity

### 5. **Reproducibility**
- Cached PRs ensure consistent analysis
- Same data across multiple runs
- Useful for comparing different `--min-score` thresholds

## Best Practices

### When to Use Cache (Default)
✅ Re-running analysis with different `--min-score`
✅ Resuming interrupted analysis
✅ Analyzing same PRs with different output formats
✅ Testing and development

### When to Clear Cache
🔄 Fetching latest PRs after new ones merged
🔄 Changing analysis scope (`--state`, `--max-prs`)
🔄 Re-analyzing with different LLM or prompt

### When to Disable Cache
🚫 One-time analysis of dynamic data
🚫 CI/CD pipelines requiring fresh data
🚫 Testing fetch logic

## Migration Notes

### Backward Compatibility
✅ **Fully backward compatible**
- Caching is enabled by default but transparent
- Existing commands work without changes
- Old workflows continue to function

### Breaking Changes
❌ **None**

### New Files Created
- `.pr_cache/` directory (added to `.gitignore`)
- Cache JSON files per repository

## Example Scenarios

### Scenario 1: Large Repository Analysis
```bash
# Day 1: Start analyzing 500 PRs
python tools/pr_analyzer.py bigorg/bigrepo --max-prs 500

# ... Analyzed 150 PRs, then system crashes ...

# Day 2: Resume from PR #151
python tools/pr_analyzer.py bigorg/bigrepo --max-prs 500 --resume
# Only analyzes remaining 350 PRs!
```

### Scenario 2: Experiment with Thresholds
```bash
# Analyze with default threshold
python tools/pr_analyzer.py owner/repo --min-score 7.0

# Try stricter threshold (uses cached data, no re-analysis!)
python tools/pr_analyzer.py owner/repo --min-score 8.0

# Try looser threshold (still uses cache)
python tools/pr_analyzer.py owner/repo --min-score 6.0
```

### Scenario 3: Daily Updates
```bash
# Monday: Initial analysis
python tools/pr_analyzer.py owner/repo --state merged --sort-by merged

# Tuesday: Fetch latest merged PRs
python tools/pr_analyzer.py owner/repo --state merged --sort-by merged --clear-cache

# Or use --no-cache for one-time fresh fetch
python tools/pr_analyzer.py owner/repo --state merged --no-cache
```

## Files Modified

1. **`tools/pr_analyzer.py`**
   - Added cache management methods
   - Modified `fetch_prs()` to use cache
   - Modified `analyze_all_prs()` to support resume
   - Added command-line arguments

2. **`tools/PR_ANALYZER_README.md`**
   - Added caching documentation
   - Updated examples with cache usage
   - Added "Caching & Resume" section

3. **`.gitignore`**
   - Added `.pr_cache/` directory

4. **`tools/CACHING_UPDATE.md`** (this file)
   - Documentation of caching feature

## Testing

### Test Cache Creation
```bash
# Run once to create cache
python tools/pr_analyzer.py owner/repo --max-prs 10

# Verify cache file exists
ls -la .pr_cache/
# Should show: owner_repo_closed_created.json
```

### Test Cache Usage
```bash
# Second run should use cache
python tools/pr_analyzer.py owner/repo --max-prs 10
# Look for: "Using cached PRs (from ...)"
```

### Test Resume
```bash
# Start analysis
python tools/pr_analyzer.py owner/repo --max-prs 20

# Interrupt with Ctrl+C after a few PRs

# Resume
python tools/pr_analyzer.py owner/repo --max-prs 20 --resume
# Look for: "Resuming from cache: X PRs already analyzed"
```

### Test Clear Cache
```bash
# Clear cache
python tools/pr_analyzer.py owner/repo --clear-cache
# Look for: "✓ Cleared cache: .pr_cache/..."
```

## Summary

The caching feature transforms the PR analyzer into a robust, resumable tool suitable for analyzing large repositories. It saves time, reduces costs, and makes the tool more reliable for real-world use cases.

**Key takeaway:** Run once, analyze many times with different parameters! 🚀
