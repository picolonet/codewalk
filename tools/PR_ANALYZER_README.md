# PR Analyzer Tool

A tool for analyzing GitHub Pull Requests to identify significant feature development with architectural and design elements. Uses LLM analysis to score and filter PRs, then generates a comprehensive YAML dataset.

## Features

- **Automated PR Fetching**: Uses GitHub CLI (`gh`) to fetch PRs from public or private repositories
- **Intelligent Caching**: Stores fetched PRs locally to avoid re-fetching and enable resume functionality
- **Resume Support**: Continue analysis from where you left off without losing progress
- **LLM-Powered Analysis**: Analyzes each PR using configurable LLM models (OpenAI GPT-4, Claude, etc.)
- **Smart Filtering**: Scores PRs on a 0-10 scale based on significance, architectural elements, and complexity
- **Comprehensive Output**: Generates detailed YAML dataset with:
  - Feature summaries
  - Solution summaries
  - Architectural elements identified
  - Technical highlights
  - Complete file change details
  - PR metadata and statistics

## Prerequisites

1. **GitHub CLI**: Must be installed and authenticated
   ```bash
   # Install gh (if not already installed)
   brew install gh  # macOS
   # or download from https://cli.github.com/

   # Authenticate
   gh auth login
   ```

2. **Python Dependencies**: Ensure PyYAML is installed
   ```bash
   pip install pyyaml
   ```

3. **LLM API Keys**: Configure appropriate environment variables based on the model you want to use:
   - OpenAI: `OPENAI_API_KEY`
   - Anthropic Claude: `ANTHROPIC_API_KEY`
   - Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_MODEL_API_VERSION`
   - Groq/Llama: `GROQ_API_KEY`

## Usage

### Basic Usage

```bash
# Analyze a public repository
python tools/pr_analyzer.py https://github.com/owner/repo

# Or use shorthand format
python tools/pr_analyzer.py owner/repo
```

### Advanced Options

```bash
# Use Claude for analysis with custom score threshold
python tools/pr_analyzer.py owner/repo --model claude --min-score 8.0

# Resume from previous analysis (saves LLM costs!)
python tools/pr_analyzer.py owner/repo --resume

# Analyze only merged PRs, sorted by merge date (most recent first)
python tools/pr_analyzer.py owner/repo --state merged --sort-by merged

# Get most recently updated PRs (useful for active development)
python tools/pr_analyzer.py owner/repo --sort-by updated --max-prs 50

# Limit number of PRs to analyze (will get the 50 most recent based on sort)
python tools/pr_analyzer.py owner/repo --max-prs 50

# Start fresh - clear cache and re-fetch
python tools/pr_analyzer.py owner/repo --clear-cache

# Specify custom output file
python tools/pr_analyzer.py owner/repo --output my_analysis.yaml

# Combine options
python tools/pr_analyzer.py owner/repo \
  --model oai \
  --min-score 7.5 \
  --max-prs 100 \
  --state closed \
  --sort-by created \
  --resume \
  --output significant_prs.yaml
```

### Command Line Options

- `repo_url`: GitHub repository URL or `owner/repo` format (required)
- `--model`: LLM model to use (choices: `oai`, `claude`, `litellm`, `llama`, `azure_oai`, `codex`, default: `oai`)
- `--min-score`: Minimum significance score 0-10 to include PR (default: `7.0`)
- `--max-prs`: Maximum number of PRs to fetch (default: all available). **Important:** Combined with `--sort-by`, this fetches the N most recent PRs based on the sort criteria
- `--state`: PR state to filter (choices: `open`, `closed`, `merged`, `all`, default: `closed`)
- `--sort-by`: Sort PRs by date field (choices: `created`, `updated`, `merged`, `closed`, default: `created`). Always sorts newest first to prioritize recent PRs
- `--cache-dir`: Directory to store cache files (default: `.pr_cache`)
- `--no-cache`: Disable PR data caching and always fetch fresh data from GitHub
- `--resume`: Resume from previous analysis, skipping already analyzed PRs
- `--clear-cache`: Clear the PR data cache for this repository before running
- `--no-llm-cache`: Disable LLM result caching (always call LLM for analysis, even if cached)
- `--clear-llm-cache`: Clear the LLM analysis cache for this repository before running
- `--output`, `-o`: Output YAML file path (default: auto-generated with timestamp)

### Sorting Behavior

**By default, the tool prioritizes the latest PRs** (newest first). This ensures that when using `--max-prs`, you analyze the most recent activity.

Sort options:
- `--sort-by created` (default): Sort by PR creation date - useful for finding recently opened PRs
- `--sort-by updated`: Sort by last update date - useful for finding recently active PRs
- `--sort-by merged`: Sort by merge date - useful for finding recently completed work
- `--sort-by closed`: Sort by close date - useful for finding recently closed/rejected PRs

**Example:** `--max-prs 50 --sort-by updated` will analyze the 50 most recently updated PRs, which is ideal for tracking active development.

## Caching & Resume

The tool includes **two-tier intelligent caching** to maximize efficiency and minimize costs:

### How Caching Works

**Tier 1: PR Data Cache**
1. **Fetched PRs Cache**: When PRs are fetched from GitHub, they're stored in `.pr_cache/{owner}_{repo}_{state}_{sort_by}.json`
2. **Analysis Cache**: As each PR is analyzed, the results are incrementally saved to the cache
3. **Resume Support**: Use `--resume` to continue from where you left off if analysis was interrupted

**Tier 2: LLM Results Cache (NEW!)**
1. **Per-PR LLM Cache**: Each PR's LLM analysis is cached separately in `.pr_cache/llm_results/{owner}_{repo}/pr_{number}.json`
2. **Smart Reuse**: Cached LLM results are reused across different analysis runs, even with different `--min-score` thresholds
3. **Maximum Cost Savings**: Zero LLM API calls when analyzing the same PRs with different parameters

### Cache Benefits

- **Avoid Re-fetching**: GitHub API calls are saved by reusing cached PR data
- **Resume Long Operations**: Interrupt and resume analysis without losing progress
- **Save LLM Costs**: Already-analyzed PRs are skipped when using `--resume`
- **Reuse LLM Results**: Change `--min-score` without re-analyzing (zero LLM cost!)
- **Incremental Updates**: Both caches are updated after each PR analysis

### Cache Usage Examples

```bash
# First run: Fetch and analyze 100 PRs (creates both caches)
python tools/pr_analyzer.py owner/repo --max-prs 100

# Second run: Use cached PRs and LLM results (NO API calls!)
python tools/pr_analyzer.py owner/repo --max-prs 100

# Try different threshold: Uses LLM cache (zero LLM cost!)
python tools/pr_analyzer.py owner/repo --max-prs 100 --min-score 8.0
# Output: "Using cached LLM analysis for PR #123..."

# Resume after interruption (skip already analyzed PRs, reuse LLM cache)
python tools/pr_analyzer.py owner/repo --max-prs 100 --resume

# Force fresh LLM analysis (keep PR cache)
python tools/pr_analyzer.py owner/repo --clear-llm-cache

# Start completely fresh: Clear both caches
python tools/pr_analyzer.py owner/repo --clear-cache --clear-llm-cache

# Disable all caching (always fetch and analyze fresh)
python tools/pr_analyzer.py owner/repo --no-cache --no-llm-cache
```

### Cache File Structure

Cache files are stored in `.pr_cache/` directory with **two-tier structure**:

```
.pr_cache/
  ├── owner_repo_closed_created.json        # PR data cache (Tier 1)
  └── llm_results/                          # LLM results cache (Tier 2)
      └── owner_repo/
          ├── pr_123.json                   # LLM analysis for PR #123
          ├── pr_124.json                   # LLM analysis for PR #124
          └── pr_125.json                   # LLM analysis for PR #125
```

**Tier 1 Cache File** (`owner_repo_closed_created.json`) contains:
- `fetched_prs`: Raw PR data from GitHub
- `analyzed_prs`: PRs with LLM analysis results
- `fetch_timestamp`: When PRs were fetched
- `last_analysis_timestamp`: Last analysis time
- Repository and filter metadata

**Tier 2 Cache Files** (`llm_results/owner_repo/pr_*.json`) each contain:
- `pr_number`: The PR number
- `analysis`: Complete LLM analysis (score, summaries, elements)
- `timestamp`: When analysis was performed

### When to Clear Cache

**Clear PR cache (`--clear-cache`)** when:
- You want to fetch the latest PRs from GitHub (new PRs may have been added)
- Repository state changed significantly

**Clear LLM cache (`--clear-llm-cache`)** when:
- You want to re-analyze PRs with a different LLM model
- You've updated the analysis prompt
- PRs have been updated and you want fresh analysis

**Clear both caches** when:
- Starting completely fresh analysis
- Switching to different repository branch or fork

**Tip**: Cache files are keyed by `{owner}_{repo}_{state}_{sort_by}`, so changing `--state` or `--sort-by` creates a new cache file automatically.

## Scoring Criteria

The LLM analyzes PRs and assigns a significance score (0-10) based on:

- **0-3**: Minor changes, bug fixes, simple features, documentation updates
- **4-6**: Moderate features with some design considerations
- **7-8**: Significant features with architectural elements and design work
- **9-10**: Major features with substantial architectural/design work, system-wide impact

The LLM considers:
- Feature complexity and scope
- Architectural/design elements introduced or modified
- Impact on codebase structure
- Code organization and patterns
- Innovation and problem-solving approach

## Output Format

The tool generates a YAML file with the following structure:

```yaml
repository: owner/repo
repository_url: https://github.com/owner/repo
analysis_date: '2025-12-02T12:34:56'
filter_criteria:
  min_significance_score: 7.0
  pr_state: closed
  total_prs_analyzed: 25

significant_prs:
  - pr_number: 123
    title: "Add real-time collaboration feature"
    url: https://github.com/owner/repo/pull/123
    author: username
    state: merged
    created_at: '2024-01-15T10:30:00Z'
    merged_at: '2024-01-20T14:45:00Z'
    files_changed: 42
    additions: 1250
    deletions: 340
    labels:
      - feature
      - enhancement
    significance_score: 9.5
    feature_summary: "Implements real-time collaborative editing using WebSockets with operational transform algorithm for conflict resolution"
    solution_summary: "Built a new WebSocket service layer with Redis pub/sub for multi-server support, integrated CRDT-based conflict resolution, and added presence indicators"
    architectural_elements:
      - "New WebSocket service layer"
      - "Redis pub/sub integration for horizontal scaling"
      - "CRDT implementation for conflict-free editing"
      - "State management refactoring"
    technical_highlights:
      - "Operational transform algorithm"
      - "Optimistic UI updates"
      - "Automatic reconnection with exponential backoff"
    analysis_reasoning: "Major architectural addition with new service layer, complex algorithm implementation, and system-wide state management changes"
    description: "Full PR description..."
    files:
      - path: "src/services/websocket/CollaborationService.ts"
        additions: 450
        deletions: 0
      - path: "src/state/collaboration/reducer.ts"
        additions: 200
        deletions: 50
      # ... more files
```

## Examples

### Example 1: Analyze React Repository

```bash
python tools/pr_analyzer.py facebook/react \
  --model claude \
  --min-score 8.0 \
  --max-prs 30 \
  --state merged
```

### Example 2: Analyze Private Repository

```bash
# Ensure gh is authenticated with access to private repo
gh auth status

python tools/pr_analyzer.py myorg/private-repo \
  --model oai \
  --min-score 7.0 \
  --output private_repo_analysis.yaml
```

### Example 3: Quick Analysis of Recent PRs

```bash
python tools/pr_analyzer.py owner/repo \
  --max-prs 20 \
  --state all \
  --min-score 6.0
```

## Tips

1. **Start with a smaller batch**: Use `--max-prs 20` initially to test before analyzing hundreds of PRs
2. **Adjust min-score**: Lower threshold (5.0-6.0) for more PRs, higher (8.0-9.0) for only the most significant
3. **State filtering**: Use `--state merged` to focus on completed work
4. **Cost considerations**: Each PR requires an LLM API call. Monitor usage especially with large repositories
5. **Private repos**: Ensure `gh auth login` is configured with appropriate permissions

## Troubleshooting

### GitHub CLI Not Found
```bash
which gh
# If not found, install: brew install gh
```

### Authentication Issues
```bash
gh auth status
gh auth login
```

### LLM API Errors
- Check that your API key is properly set in environment variables
- Verify the `.env` file in the project root or export keys directly:
  ```bash
  export OPENAI_API_KEY="your-key-here"
  ```

### Rate Limiting
- GitHub CLI respects GitHub API rate limits
- Add delays between requests if needed
- For large repositories, consider running analysis in batches

## Output Analysis

After generating the YAML file, you can:

1. **Filter and sort**: Use `yq` or Python to filter PRs by score, author, date, etc.
2. **Generate reports**: Parse the YAML to create summaries or visualizations
3. **Build training datasets**: Use significant PRs as examples for code generation or analysis models
4. **Identify patterns**: Analyze architectural_elements across PRs to understand project evolution

## Architecture

The tool follows this workflow:

1. **Fetch**: Uses `gh pr list` to retrieve PRs based on filters
2. **Details**: For each PR, fetches full details including diff via `gh pr view` and `gh pr diff`
3. **Analyze**: Sends PR context (title, description, files, diff sample) to LLM
4. **Score**: LLM returns structured JSON with scores and analysis
5. **Filter**: Keeps only PRs meeting minimum score threshold
6. **Output**: Generates comprehensive YAML dataset sorted by significance

## License

Part of the CodeWalk project.
