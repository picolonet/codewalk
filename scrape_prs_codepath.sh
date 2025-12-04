python tools/pr_analyzer.py codepath/codepath-backend \
  --model oai \
  --min-score 7.5 \
  --max-prs 100 \
  --state closed \
  --model azure_oai \
  --output cp_significant_prs.yaml
