"""Framework-agnostic Kaggle submission utilities.

Any detector works as long as its test predictions are COCO-format records
({"image_id", "category_id" (1-based), "bbox" [x, y, w, h] in original pixels, "score"}),
e.g. from src.ultralytics_utils.predict_coco or src.evaluation.format_coco_predictions.
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.setup_utils import COMPETITION_NAME

MAX_DETS_SCORED = 500  # detections per image beyond this are ignored by the scorer
NUM_CLASSES = 8


def load_test_frames(dataset_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Reads test_images.csv into records with "id" (submission image_id) and "file_name"
    (relative to the dataset root), plus clip, frame, width and height."""
    df = pd.read_csv(Path(dataset_dir) / "test_images.csv")
    return [{**row, "id": row["image_id"]} for row in df.to_dict("records")]


def build_submission(
    records: List[Dict[str, Any]],
    test_frames: List[Dict[str, Any]],
    output_path: Union[str, Path],
    max_dets: int = MAX_DETS_SCORED,
) -> pd.DataFrame:
    """Writes the submission CSV (one row per test image, in test_images.csv order).

    PredictionString = space-separated "class score x y w h" groups with 0-based class ids
    and top-left pixel coordinates; empty for images without detections.
    """
    by_image: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_image[r["image_id"]].append(r)

    rows = []
    for frame in test_frames:
        W, H = float(frame["width"]), float(frame["height"])
        dets = sorted(by_image.get(frame["id"], []), key=lambda r: -r["score"])[:max_dets]
        tokens = []
        for d in dets:
            x, y, w, h = d["bbox"]
            x1, y1 = min(max(x, 0.0), W), min(max(y, 0.0), H)
            x2, y2 = min(max(x + w, 0.0), W), min(max(y + h, 0.0), H)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            tokens.append(
                f"{int(d['category_id']) - 1} {d['score']:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}"
            )
        rows.append({"image_id": frame["id"], "PredictionString": " ".join(tokens)})

    sub = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    return sub


def validate_submission(
    submission_path: Union[str, Path],
    test_frames: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Checks columns, one row per test image, and "class score x y w h" token groups.
    Empty PredictionStrings are valid (images without detections)."""
    df = pd.read_csv(submission_path, keep_default_na=False, dtype={"PredictionString": str})
    expected_ids = {f["id"] for f in test_frames}

    errors: List[str] = []
    if list(df.columns) != ["image_id", "PredictionString"]:
        errors.append(f"columns are {list(df.columns)}")
    if df["image_id"].duplicated().any():
        errors.append(f"{int(df['image_id'].duplicated().sum())} duplicated image_id")
    if set(df["image_id"]) != expected_ids:
        errors.append(
            f"{len(expected_ids - set(df['image_id']))} missing / {len(set(df['image_id']) - expected_ids)} unknown image_id"
        )

    n_dets, n_empty = 0, 0
    for image_id, pred in zip(df["image_id"], df["PredictionString"]):
        tokens = pred.split()
        if not tokens:
            n_empty += 1
            continue
        if len(tokens) % 6:
            errors.append(f"{image_id}: token count {len(tokens)} is not a multiple of 6")
            break
        groups = [tokens[i : i + 6] for i in range(0, len(tokens), 6)]
        if len(groups) > MAX_DETS_SCORED:
            errors.append(f"{image_id}: {len(groups)} detections > {MAX_DETS_SCORED}")
            break
        if any(not g[0].isdigit() or not 0 <= int(g[0]) < NUM_CLASSES for g in groups):
            errors.append(f"{image_id}: class id outside 0..{NUM_CLASSES - 1}")
            break
        n_dets += len(groups)

    return {
        "is_valid": not errors,
        "errors": errors,
        "rows": len(df),
        "expected_rows": len(expected_ids),
        "empty_rows": n_empty,
        "detections": n_dets,
        "size_MB": round(Path(submission_path).stat().st_size / 1e6, 2),
    }


def submit_to_kaggle(submission_path: Union[str, Path], message: str) -> None:
    """Uploads a submission with the Kaggle CLI (needs ~/.kaggle/kaggle.json)."""
    subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "submit",
         "-c", COMPETITION_NAME, "-f", str(submission_path), "-m", message],
        check=True,
    )
