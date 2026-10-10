#!/usr/bin/env python3
"""Clean, combined RSNA knee abnormality pipeline.

This file consolidates the notebook's key pieces into one consistent pipeline:
1. Load competition CSV files.
2. Keep the competition label definitions in one place.
3. Validate prediction probabilities before writing the final submission.
4. Write a Kaggle-ready submission.csv that matches the sample submission schema.

The file is intentionally structured so it can be used as a clean script in the
Kaggle environment without the fragmented notebook cell flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


LABEL_COLS: Sequence[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def load_competition_tables(
    train_path: str | Path,
    test_path: str | Path,
    train_series_path: str | Path,
    test_series_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the main competition tables used by the notebook pipeline."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_series_df = pd.read_csv(train_series_path)
    test_series_df = pd.read_csv(test_series_path)

    return train_df, test_df, train_series_df, test_series_df


def prepare_labeled_studies(train_df: pd.DataFrame, label_cols: Sequence[str]) -> pd.DataFrame:
    """Return only rows with all required labels present."""
    labeled = train_df.dropna(subset=list(label_cols)).copy()
    return labeled


def _normalize_prediction_frame(predictions, label_cols: Sequence[str]) -> pd.DataFrame:
    """Coerce a prediction mapping or DataFrame into the expected label schema."""
    if isinstance(predictions, Mapping):
        frame = pd.DataFrame.from_dict(predictions, orient="index", columns=list(label_cols))
        return frame

    if hasattr(predictions, "to_pandas"):
        predictions = predictions.to_pandas()

    if isinstance(predictions, pd.DataFrame):
        frame = predictions.copy()
    elif isinstance(predictions, str):
        frame = pd.read_csv(predictions, index_col=0)
    else:
        raise TypeError(
            "Unsupported predictions object. Pass a dict-like mapping, DataFrame, or CSV path."
        )

    if list(frame.columns) != list(label_cols):
        missing = [c for c in label_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Prediction frame is missing required columns: {missing}")
        extra = [c for c in frame.columns if c not in label_cols]
        if extra:
            raise ValueError(f"Prediction frame contains unexpected columns: {extra}")
        frame = frame[list(label_cols)]

    frame.index = frame.index.astype(str)
    return frame


def generate_submission(
    predictions,
    sample_submission_path: str | Path,
    output_path: str | Path,
    label_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a competition-ready submission.csv from a prediction frame."""
    sample_submission = pd.read_csv(sample_submission_path)
    if "StudyInstanceUID" not in sample_submission.columns:
        raise ValueError("Sample submission is missing StudyInstanceUID")

    if label_cols is None:
        label_cols = [column for column in sample_submission.columns if column != "StudyInstanceUID"]

    if not label_cols:
        raise ValueError("No label columns found in the sample submission")

    prediction_frame = _normalize_prediction_frame(predictions, label_cols)
    submission_ids = sample_submission["StudyInstanceUID"].astype(str)

    if len(prediction_frame) == 0:
        raise ValueError("Prediction frame is empty")

    prediction_index = prediction_frame.index.astype(str)
    if set(submission_ids) != set(prediction_index):
        missing = sorted(set(submission_ids) - set(prediction_index))
        unexpected = sorted(set(prediction_index) - set(submission_ids))
        raise ValueError(
            "Prediction IDs do not match the sample submission IDs. "
            f"Missing: {missing[:5]}; Unexpected: {unexpected[:5]}"
        )

    aligned = prediction_frame.reindex(submission_ids.tolist())
    values = aligned.to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1")

    submission = sample_submission.copy()
    submission.loc[:, list(label_cols)] = values

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)

    saved = pd.read_csv(output)
    if saved.shape != submission.shape or saved.columns.tolist() != submission.columns.tolist():
        raise ValueError("Saved submission does not match the expected sample submission schema")

    print(f"Saved {len(saved)} predictions to {output}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean RSNA knee abnormality pipeline with submission validation."
    )
    parser.add_argument(
        "--train",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv",
        help="Path to train.csv.",
    )
    parser.add_argument(
        "--test",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/test.csv",
        help="Path to test.csv.",
    )
    parser.add_argument(
        "--train-series",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series.csv",
        help="Path to train_series.csv.",
    )
    parser.add_argument(
        "--test-series",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/test_series.csv",
        help="Path to test_series.csv.",
    )
    parser.add_argument(
        "--sample",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv",
        help="Path to the sample submission CSV.",
    )
    parser.add_argument(
        "--predictions",
        default="predictions.csv",
        help="CSV file with StudyInstanceUID as index and label probabilities as columns.",
    )
    parser.add_argument(
        "--output",
        default="/kaggle/working/submission.csv",
        help="Output path for the final submission CSV.",
    )
    args = parser.parse_args()

    train_df, test_df, train_series_df, test_series_df = load_competition_tables(
        train_path=args.train,
        test_path=args.test,
        train_series_path=args.train_series,
        test_series_path=args.test_series,
    )

    labeled = prepare_labeled_studies(train_df, LABEL_COLS)
    print(f"Train rows: {len(train_df)}")
    print(f"Labeled train rows: {len(labeled)}")

    prediction_file = Path(args.predictions)
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {prediction_file}. "
            "Generate probabilities before running this script or provide an existing predictions.csv file."
        )

    prediction_frame = pd.read_csv(prediction_file, index_col=0)
    generate_submission(
        predictions=prediction_frame,
        sample_submission_path=args.sample,
        output_path=args.output,
        label_cols=LABEL_COLS,
    )


if __name__ == "__main__":
    main()
