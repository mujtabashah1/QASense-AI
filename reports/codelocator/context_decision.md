# Source context decision

{
  "alternate_rows": 16899,
  "schemas": [
    {
      "file": "train-00000-of-00002.parquet",
      "columns": [
        "identifier",
        "condition",
        "bug",
        "fix",
        "post_conditional",
        "bugType",
        "buggy_commit",
        "repository_url",
        "bug_timestamp",
        "num_commits_til_fix",
        "originating_commit"
      ],
      "rows": 8450
    },
    {
      "file": "train-00001-of-00002.parquet",
      "columns": [
        "identifier",
        "condition",
        "bug",
        "fix",
        "post_conditional",
        "bugType",
        "buggy_commit",
        "repository_url",
        "bug_timestamp",
        "num_commits_til_fix",
        "originating_commit"
      ],
      "rows": 8449
    }
  ],
  "index_entries": 12620,
  "ambiguous_source_keys": 481,
  "outcomes": {
    "verified_full_source_join": 12505,
    "no_unique_context_join": 48060,
    "line_number_mismatch": 1821,
    "character_offset_mismatch": 8
  },
  "verified_samples": 12505,
  "genuine_context_available": true,
  "localization_scientifically_supported": true,
  "decision": "Only complete pre-fix sources reconstructed from condition + bug + post_conditional, joined by repository, parent commit, category and exact buggy/fixed fragments, with matching line and character offset. Patches are never predictive inputs. No fabricated context."
}

Primary source fields are fragments. Alternate prefix + original bug + suffix reconstruct real pre-fix files. Only exact joins with verified line and character offset are accepted. Candidate inputs center every candidate using the same context rule; neither original bug index nor a patch is exposed. Other context lines are weak negatives: they are not independently proven bug free. Ranking evaluation is conditional on verified Java source context and does not establish full-repository localization.
