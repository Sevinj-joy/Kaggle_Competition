---
name: kaggle-notebook-review-and-submission
description: Review a Kaggle notebook for syntax/data bugs, fix the run path, and validate the final submission schema before producing the competition output.
---

# Kaggle Notebook Review and Submission Skill

## Goal
Use this workflow when a Kaggle notebook is broken, partially duplicated, or close to submission but needs validation before it is ready for the competition.

## Workflow
1. Inspect the notebook in stages: imports, dataset loading, feature extraction, model training, validation, inference, and final submission.
2. Check for malformed code, duplicate experiments, undefined variables, broken file paths, bad quoting, and mismatched schema assumptions.
3. Validate the exact submission contract against the sample submission: row order, StudyInstanceUID coverage, required columns, and probability ranges.
4. Fix the root cause rather than hiding symptoms.
5. Produce a clean submission script or notebook cell that writes the final CSV to the required Kaggle output path.

## Decision Points
- If a notebook cell contains invalid Python syntax, correct that cell immediately and re-run parse checks.
- If the sample submission schema is the source of truth, align every prediction row to that schema before writing the CSV.
- If the data is not available locally, keep the pipeline in a scripted generator and do not fabricate a fake submission file.

## Completion Checks
- Notebook Python parses successfully.
- All required label columns exist in the submission file.
- Every StudyInstanceUID in the sample submission has a prediction.
- Predictions are finite and within the $[0, 1]$ range.
- The CSV is written to the expected Kaggle output path and has the correct shape and column order.

## Example prompts
- Review this Kaggle notebook for syntax and submission bugs.
- Fix the broken notebook and generate the competition-ready CSV.
- Validate the final output schema against the sample submission.
