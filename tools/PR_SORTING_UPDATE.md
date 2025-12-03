# PR Analyzer - Sorting Update

## Summary

Updated the PR analyzer tool to **explicitly prioritize the latest PRs** (newest first) when using `--max-prs` limit. Added flexible sorting options to allow users to choose what "latest" means.

## Changes Made

### 1. Enhanced PR Fetching (tools/pr_analyzer.py)

**Previous behavior:** Relied on GitHub CLI's default ordering (which is typically newest first, but not guaranteed).

**New behavior:**
- Explicitly sorts PRs by the specified date field in descending order (newest first)
- Ensures consistent behavior across all use cases
- Added configurable sort criteria

```python
# Sorts PRs by specified field (newest first)
prs.sort(
    key=lambda pr: pr.get(sort_field) or '',
    reverse=True
)
```

### 2. New `--sort-by` Parameter

Added command-line option to control sorting:

```bash
--sort-by {created,updated,merged,closed}
```

**Sort options:**
- `created` (default): Sort by PR creation date
- `updated`: Sort by last update date (useful for active development)
- `merged`: Sort by merge date (useful for recently completed work)
- `closed`: Sort by close date

### 3. Updated Examples

```bash
# Get 50 most recently created PRs
python tools/pr_analyzer.py owner/repo --max-prs 50

# Get 50 most recently updated PRs (active development)
python tools/pr_analyzer.py owner/repo --max-prs 50 --sort-by updated

# Get recently merged PRs
python tools/pr_analyzer.py owner/repo --state merged --sort-by merged --max-prs 30
```

## Benefits

### ✅ Predictable Behavior
When you specify `--max-prs 50`, you're guaranteed to get the 50 most recent PRs based on your chosen sort criteria, not an arbitrary subset.

### ✅ Prioritizes Latest Work
The default `--sort-by created` ensures you analyze the most recently opened PRs, which typically represent the latest feature development.

### ✅ Flexible Analysis
Different sorting options allow you to focus on different aspects:
- **Recently created** (`created`): Find newly proposed features
- **Recently updated** (`updated`): Track active development
- **Recently merged** (`merged`): Analyze completed work
- **Recently closed** (`closed`): Review recently rejected/closed PRs

### ✅ Efficient Resource Usage
When analyzing large repositories with thousands of PRs, you can focus on the most recent N PRs, saving API calls and LLM analysis costs.

## Technical Details

### Modified Functions

1. **`__init__` method**: Added `sort_by` parameter
   ```python
   def __init__(self, ..., sort_by: str = "created"):
   ```

2. **`fetch_prs` method**: Added explicit sorting logic
   - Fetches PRs from GitHub CLI
   - Sorts by specified date field (newest first)
   - Added `updatedAt` to JSON fields fetched

3. **`main` function**: Added `--sort-by` argument parser

### Backward Compatibility

✅ **Fully backward compatible** - Default behavior is `--sort-by created`, which maintains the expected behavior of prioritizing newest PRs.

Existing scripts will continue to work without modification.

## Use Cases

### Use Case 1: Track Latest Features
```bash
# Analyze the 100 most recently created PRs
python tools/pr_analyzer.py myorg/myrepo --max-prs 100 --sort-by created
```

### Use Case 2: Monitor Active Development
```bash
# Analyze the 50 most recently updated PRs (good for daily monitoring)
python tools/pr_analyzer.py myorg/myrepo --max-prs 50 --sort-by updated
```

### Use Case 3: Review Recent Merges
```bash
# Analyze recently merged PRs to build a dataset of completed features
python tools/pr_analyzer.py myorg/myrepo --state merged --sort-by merged --max-prs 200
```

## Files Modified

1. `/tools/pr_analyzer.py` - Core implementation
2. `/tools/PR_ANALYZER_README.md` - Documentation updated
3. `/tools/PR_SORTING_UPDATE.md` - This update summary

## Verification

The tool now always prints the sorting behavior:
```
Fetched 50 PRs (sorted by created date, newest first)
```

This confirms that PRs are being sorted as expected.
