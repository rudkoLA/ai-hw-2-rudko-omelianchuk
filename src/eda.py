"""Exploratory Data Analysis (EDA) functions for VisDrone/HW2 dataset."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import cv2
import numpy as np
import pandas as pd


def load_coco_dataset(coco_path: Path) -> Dict[str, Any]:
    """Loads COCO format annotations JSON."""
    with open(coco_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
        return data


def compute_class_distribution(
    coco_data: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Computes target class frequencies, proportions, and imbalance metrics."""
    cat_names: Dict[int, str] = {c["id"]: c["name"] for c in coco_data["categories"]}
    target_anns = [a for a in coco_data["annotations"] if a.get("iscrowd", 0) == 0]
    total_targets = len(target_anns)

    counts = Counter(int(a["category_id"]) for a in target_anns)
    records: List[Dict[str, Any]] = []
    for cid in sorted(cat_names.keys()):
        cnt = counts.get(cid, 0)
        pct = (cnt / total_targets * 100.0) if total_targets > 0 else 0.0
        records.append({
            "class_id": cid,
            "class_name": cat_names[cid],
            "count": cnt,
            "percentage": pct,
        })
    df = pd.DataFrame(records)
    
    count_values = [r["count"] for r in records]
    max_cnt = max(count_values) if count_values else 0
    non_zero = [c for c in count_values if c > 0]
    min_cnt = min(non_zero) if non_zero else 1
    imbalance_ratio = float(max_cnt / min_cnt)
    
    max_idx = int(np.argmax(count_values)) if count_values else 0
    min_idx = int(np.argmin(count_values)) if count_values else 0

    metrics: Dict[str, Any] = {
        "total_instances": float(total_targets),
        "majority_class": records[max_idx]["class_name"],
        "minority_class": records[min_idx]["class_name"],
        "imbalance_ratio": imbalance_ratio,
    }
    return df, metrics


def compute_size_metrics(
    coco_data: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Computes bbox scale distributions according to COCO definitions."""
    cat_names: Dict[int, str] = {c["id"]: c["name"] for c in coco_data["categories"]}
    records: List[Dict[str, Any]] = []
    areas: List[float] = []
    aspect_ratios: List[float] = []

    for a in coco_data["annotations"]:
        if a.get("iscrowd", 0) == 0:
            w = float(a["bbox"][2])
            h = float(a["bbox"][3])
            area = w * h
            ar = (w / h) if h > 0 else 0.0
            scale_cat = "small" if area < 1024.0 else ("medium" if area <= 9216.0 else "large")
            areas.append(area)
            aspect_ratios.append(ar)
            records.append({
                "category_id": a["category_id"],
                "class_name": cat_names.get(a["category_id"], str(a["category_id"])),
                "width": w,
                "height": h,
                "area": area,
                "aspect_ratio": ar,
                "scale_category": scale_cat,
            })
    df = pd.DataFrame(records)
    
    areas_arr = np.array(areas) if areas else np.array([0.0])
    ar_arr = np.array(aspect_ratios) if aspect_ratios else np.array([0.0])
    total = len(areas)

    small_cnt = int(np.sum(areas_arr < 1024.0))
    med_cnt = int(np.sum((areas_arr >= 1024.0) & (areas_arr <= 9216.0)))
    large_cnt = int(np.sum(areas_arr > 9216.0))

    summary: Dict[str, float] = {
        "small_percentage": float(small_cnt / total * 100.0) if total > 0 else 0.0,
        "medium_percentage": float(med_cnt / total * 100.0) if total > 0 else 0.0,
        "large_percentage": float(large_cnt / total * 100.0) if total > 0 else 0.0,
        "median_area": float(np.median(areas_arr)),
        "mean_area": float(np.mean(areas_arr)),
        "median_aspect_ratio": float(np.median(ar_arr)),
    }
    return df, summary


def compute_stride_iou_drift(
    coco_data: Dict[str, Any],
    strides: List[int] = [1, 2, 3, 5, 10],
    max_tracks: int = 500,
) -> Dict[int, Dict[str, float]]:
    """Calculates temporal bounding box overlap (IoU) drift across frame strides."""
    track_boxes: Dict[Tuple[str, int], Dict[int, List[float]]] = defaultdict(dict)
    img_map: Dict[int, Tuple[str, int]] = {
        img["id"]: (img["video"], img["frame_id"]) for img in coco_data["images"]
    }

    for a in coco_data["annotations"]:
        if a.get("iscrowd", 0) == 0 and "track_id" in a:
            vid, frame_id = img_map[a["image_id"]]
            track_boxes[(vid, a["track_id"])][frame_id] = [float(x) for x in a["bbox"]]

    def box_iou(b1: List[float], b2: List[float]) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return (inter / union) if union > 0 else 0.0

    stride_results: Dict[int, Dict[str, float]] = {}
    sampled_tracks = list(track_boxes.items())[:max_tracks]

    for s in strides:
        ious: List[float] = []
        for (_, _), frames in sampled_tracks:
            sorted_f = sorted(frames.keys())
            for f in sorted_f:
                if f + s in frames:
                    ious.append(box_iou(frames[f], frames[f + s]))
        stride_results[s] = {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "median_iou": float(np.median(ious)) if ious else 0.0,
            "sample_count": float(len(ious)),
        }
    return stride_results


def compute_ignored_spatial_density(
    dataset_dir: Path,
    sample_targets: int = 150000,
) -> Dict[str, Any]:
    """Computes normalized 2D spatial density distributions for ignored regions vs target objects."""
    ign_dir = dataset_dir / "train" / "ignore"
    ign_xc: List[float] = []
    ign_yc: List[float] = []
    ign_w: List[float] = []
    ign_h: List[float] = []
    ign_areas: List[float] = []

    for p in ign_dir.glob("*/*.txt"):
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    xc, yc, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    ign_xc.append(xc)
                    ign_yc.append(yc)
                    ign_w.append(bw)
                    ign_h.append(bh)
                    ign_areas.append(bw * bh)

    with open(dataset_dir / "train" / "annotations_coco.json") as f:
        coco: Dict[str, Any] = json.load(f)

    img_wh: Dict[int, Tuple[int, int]] = {
        img["id"]: (img["width"], img["height"]) for img in coco["images"]
    }
    tgt_xc: List[float] = []
    tgt_yc: List[float] = []
    tgt_w: List[float] = []
    tgt_h: List[float] = []
    tgt_areas: List[float] = []

    step = max(1, len(coco["annotations"]) // sample_targets)
    for a in coco["annotations"][::step]:
        if a.get("iscrowd", 0) == 0:
            w_img, h_img = img_wh[a["image_id"]]
            bx, by, bw, bh = a["bbox"]
            n_xc = float(bx + bw / 2.0) / float(w_img)
            n_yc = float(by + bh / 2.0) / float(h_img)
            n_w = float(bw) / float(w_img)
            n_h = float(bh) / float(h_img)
            tgt_xc.append(n_xc)
            tgt_yc.append(n_yc)
            tgt_w.append(n_w)
            tgt_h.append(n_h)
            tgt_areas.append(n_w * n_h)

    ign_yc_arr = np.array(ign_yc)
    tgt_yc_arr = np.array(tgt_yc)

    return {
        "total_ignored": len(ign_xc),
        "total_targets_sampled": len(tgt_xc),
        "ign_xc": ign_xc,
        "ign_yc": ign_yc,
        "tgt_xc": tgt_xc,
        "tgt_yc": tgt_yc,
        "ign_median_y": float(np.median(ign_yc_arr)),
        "tgt_median_y": float(np.median(tgt_yc_arr)),
        "ign_top_quarter_pct": float((ign_yc_arr < 0.25).mean() * 100.0) if len(ign_yc_arr) else 0.0,
        "tgt_top_quarter_pct": float((tgt_yc_arr < 0.25).mean() * 100.0) if len(tgt_yc_arr) else 0.0,
        "ign_median_area_norm": float(np.median(ign_areas)) if ign_areas else 0.0,
        "tgt_median_area_norm": float(np.median(tgt_areas)) if tgt_areas else 0.0,
    }


def compute_optical_flow_box_progression(
    dataset_dir: Path,
    sequence_name: str,
    num_pairs: int = 10,
    step: int = 1,
    flow_scale: float = 0.5,
) -> Dict[str, Any]:
    """Progression of bounding boxes between sequential frames using forward and backward optical flow.
    Warped predictions are compared against ground truth in destination frames.
    """
    with open(dataset_dir / "train" / "annotations_coco.json") as f:
        coco: Dict[str, Any] = json.load(f)

    img_map: Dict[int, Tuple[str, int]] = {
        img["id"]: (img["video"], img["frame_id"]) for img in coco["images"]
    }
    frame_boxes: Dict[Tuple[str, int], Dict[int, List[float]]] = defaultdict(dict)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 0 and "track_id" in a:
            vid, fid = img_map[a["image_id"]]
            if vid == sequence_name:
                frame_boxes[(vid, fid)][a["track_id"]] = [float(x) for x in a["bbox"]]

    fids = sorted([f for (v, f) in frame_boxes if v == sequence_name])
    if len(fids) < step + 1:
        return {}

    pairs: List[Tuple[int, int]] = []
    stride = max(1, (len(fids) - step) // num_pairs)
    for i in range(0, len(fids) - step, stride):
        pairs.append((fids[i], fids[i + step]))
        if len(pairs) >= num_pairs:
            break

    def box_iou(b1: List[float], b2: List[float]) -> float:
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return (inter / union) if union > 0 else 0.0

    fwd_ious: List[float] = []
    bwd_ious: List[float] = []
    static_ious: List[float] = []
    fwd_errs: List[float] = []
    bwd_errs: List[float] = []
    static_dists: List[float] = []

    img_dir = dataset_dir / "train" / "images" / sequence_name

    for f1, f2 in pairs:
        p1 = img_dir / f"{f1:07d}.jpg"
        p2 = img_dir / f"{f2:07d}.jpg"
        if not p1.exists() or not p2.exists():
            continue

        im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            continue

        h, w = im1.shape
        s1 = cv2.resize(im1, (0, 0), fx=flow_scale, fy=flow_scale)
        s2 = cv2.resize(im2, (0, 0), fx=flow_scale, fy=flow_scale)

        flow_fwd = (
            cv2.resize(
                cv2.calcOpticalFlowFarneback(s1, s2, None, 0.5, 3, 15, 3, 5, 1.2, 0),  # pyright: ignore[reportCallIssue,reportArgumentType]
                (w, h),
            )
            * (1.0 / flow_scale)
        )
        flow_bwd = (
            cv2.resize(
                cv2.calcOpticalFlowFarneback(s2, s1, None, 0.5, 3, 15, 3, 5, 1.2, 0),  # pyright: ignore[reportCallIssue,reportArgumentType]
                (w, h),
            )
            * (1.0 / flow_scale)
        )

        t1_boxes = frame_boxes[(sequence_name, f1)]
        t2_boxes = frame_boxes[(sequence_name, f2)]
        common = set(t1_boxes.keys()) & set(t2_boxes.keys())

        for tid in common:
            b1 = t1_boxes[tid]
            b2 = t2_boxes[tid]
            static_ious.append(box_iou(b1, b2))

            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
            static_dists.append(float(np.hypot(c1[0] - c2[0], c1[1] - c2[1])))

            # Forward progression (t -> t+step)
            x1, y1, bw1, bh1 = map(int, b1)
            patch_f = flow_fwd[max(0, y1) : min(h, y1 + bh1), max(0, x1) : min(w, x1 + bw1)]
            dx = float(np.median(patch_f[:, :, 0])) if patch_f.size > 0 else 0.0
            dy = float(np.median(patch_f[:, :, 1])) if patch_f.size > 0 else 0.0
            pred_b2 = [b1[0] + dx, b1[1] + dy, b1[2], b1[3]]
            fwd_ious.append(box_iou(pred_b2, b2))
            pred_c2 = (pred_b2[0] + pred_b2[2] / 2.0, pred_b2[1] + pred_b2[3] / 2.0)
            fwd_errs.append(float(np.hypot(pred_c2[0] - c2[0], pred_c2[1] - c2[1])))

            # Backward progression (t+step -> t)
            x2, y2, bw2, bh2 = map(int, b2)
            patch_b = flow_bwd[max(0, y2) : min(h, y2 + bh2), max(0, x2) : min(w, x2 + bw2)]
            bdx = float(np.median(patch_b[:, :, 0])) if patch_b.size > 0 else 0.0
            bdy = float(np.median(patch_b[:, :, 1])) if patch_b.size > 0 else 0.0
            pred_b1 = [b2[0] + bdx, b2[1] + bdy, b2[2], b2[3]]
            bwd_ious.append(box_iou(pred_b1, b1))
            pred_c1 = (pred_b1[0] + pred_b1[2] / 2.0, pred_b1[1] + pred_b1[3] / 2.0)
            bwd_errs.append(float(np.hypot(pred_c1[0] - c1[0], pred_c1[1] - c1[1])))

    return {
        "pair_count": len(pairs),
        "evaluated_boxes": len(fwd_ious),
        "mean_static_iou": float(np.mean(static_ious)) if static_ious else 0.0,
        "median_static_iou": float(np.median(static_ious)) if static_ious else 0.0,
        "mean_fwd_iou": float(np.mean(fwd_ious)) if fwd_ious else 0.0,
        "median_fwd_iou": float(np.median(fwd_ious)) if fwd_ious else 0.0,
        "mean_bwd_iou": float(np.mean(bwd_ious)) if bwd_ious else 0.0,
        "median_bwd_iou": float(np.median(bwd_ious)) if bwd_ious else 0.0,
        "mean_static_displacement_px": float(np.mean(static_dists)) if static_dists else 0.0,
        "mean_fwd_error_px": float(np.mean(fwd_errs)) if fwd_errs else 0.0,
        "median_fwd_error_px": float(np.median(fwd_errs)) if fwd_errs else 0.0,
        "mean_bwd_error_px": float(np.mean(bwd_errs)) if bwd_errs else 0.0,
        "median_bwd_error_px": float(np.median(bwd_errs)) if bwd_errs else 0.0,
    }


def detect_postprocessed_blur_regions(
    bgr_img: np.ndarray,
    ksize: int = 21,
    min_area: float = 80.0,
    max_area: float = 25000.0,
    base_width: float = 960.0,
) -> List[Dict[str, Any]]:
    """Detects intentional post-processed blurred regions in video frames using re-blur contrast metric."""
    h, w = bgr_img.shape[:2]
    scale = base_width / float(w)
    small = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    reblur = cv2.GaussianBlur(gray, (ksize, ksize), 4.0)
    diff = cv2.absdiff(gray, reblur).astype(np.float32)

    mean_val = cv2.boxFilter(gray.astype(np.float32), -1, (ksize, ksize))
    sq_val = cv2.boxFilter((gray.astype(np.float32)) ** 2, -1, (ksize, ksize))
    local_std = np.sqrt(np.maximum(0.0, sq_val - mean_val**2))

    smooth_ratio = cv2.boxFilter(diff / (local_std + 1e-4), -1, (ksize, ksize))
    blur_mask = ((local_std > 8.0) & (smooth_ratio < 0.22)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_OPEN, kernel)
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(blur_mask)
    spots: List[Dict[str, Any]] = []
    inv_scale = 1.0 / scale

    for i in range(1, num_labels):
        s_area = float(stats[i, cv2.CC_STAT_AREA])
        orig_area = s_area * (inv_scale**2)
        if min_area <= orig_area <= max_area:
            sx = float(stats[i, cv2.CC_STAT_LEFT]) * inv_scale
            sy = float(stats[i, cv2.CC_STAT_TOP]) * inv_scale
            sw = float(stats[i, cv2.CC_STAT_WIDTH]) * inv_scale
            sh = float(stats[i, cv2.CC_STAT_HEIGHT]) * inv_scale
            scx = float(centroids[i, 0]) * inv_scale
            scy = float(centroids[i, 1]) * inv_scale
            spots.append({
                "area": orig_area,
                "bbox": [sx, sy, sw, sh],
                "centroid": [scx, scy],
                "aspect_ratio": (sw / sh) if sh > 0 else 1.0,
            })
    return spots


def track_blur_dynamics(
    dataset_dir: Path,
    sequence_name: str,
    frame_indices: List[int],
    is_test: bool = False,
) -> Dict[str, Any]:
    """Tracks detected blur spots across sequential frames to measure displacement and persistence."""
    sub = "test" if is_test else "train"
    img_dir = dataset_dir / sub / "images" / sequence_name
    spots_per_frame: List[List[Dict[str, Any]]] = []

    for fid in frame_indices:
        fp = img_dir / f"{fid:07d}.jpg"
        if fp.exists():
            im = cv2.imread(str(fp))
            if im is not None:
                spots_per_frame.append(detect_postprocessed_blur_regions(im))

    displacements: List[float] = []
    tracked_areas: List[float] = []

    for i in range(len(spots_per_frame) - 1):
        f_spots1 = spots_per_frame[i]
        f_spots2 = spots_per_frame[i + 1]
        for s1 in f_spots1:
            tracked_areas.append(s1["area"])
            c1 = s1["centroid"]
            best_d = 1e9
            for s2 in f_spots2:
                c2 = s2["centroid"]
                d = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                if d < best_d:
                    best_d = d
            if best_d < 45.0:
                displacements.append(best_d)

    areas_arr = np.array(tracked_areas) if tracked_areas else np.array([0.0])
    disp_arr = np.array(displacements) if displacements else np.array([0.0])

    return {
        "total_spots_detected": len(tracked_areas),
        "matched_transitions": len(displacements),
        "median_area": float(np.median(areas_arr)),
        "mean_area": float(np.mean(areas_arr)),
        "p25_area": float(np.percentile(areas_arr, 25)),
        "p75_area": float(np.percentile(areas_arr, 75)),
        "median_displacement_px": float(np.median(disp_arr)),
        "mean_displacement_px": float(np.mean(disp_arr)),
    }


def compute_velocity_flow_dynamics(
    dataset_dir: Path,
    sequence_name: str,
    num_pairs: int = 15,
    flow_scale: float = 0.5,
) -> Dict[str, Any]:
    """Calculates pixel displacement vectors (dx, dy) across consecutive frames for vehicles vs. pedestrians using dense optical flow and ground truth annotations.

    Compares:
    - Pedestrians (category 1: person)
    - Vehicles (categories 3, 4, 5, 7, 8: car, van, truck, bus, motor)
    """
    with open(dataset_dir / "train" / "annotations_coco.json") as f:
        coco: Dict[str, Any] = json.load(f)

    img_map: Dict[int, Tuple[str, int]] = {
        img["id"]: (img["video"], img["frame_id"]) for img in coco["images"]
    }
    frame_boxes: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 0 and "track_id" in a:
            vid, fid = img_map[a["image_id"]]
            if vid == sequence_name:
                frame_boxes[(vid, fid)][a["track_id"]] = {
                    "bbox": [float(x) for x in a["bbox"]],
                    "category_id": int(a["category_id"]),
                }

    fids = sorted([f for (v, f) in frame_boxes if v == sequence_name])
    if len(fids) < 2:
        return {}

    pairs: List[Tuple[int, int]] = []
    stride = max(1, (len(fids) - 1) // num_pairs)
    for i in range(0, len(fids) - 1, stride):
        pairs.append((fids[i], fids[i + 1]))
        if len(pairs) >= num_pairs:
            break

    img_dir = dataset_dir / "train" / "images" / sequence_name

    ped_gt_speeds: List[float] = []
    veh_gt_speeds: List[float] = []
    ped_flow_speeds: List[float] = []
    veh_flow_speeds: List[float] = []
    ped_dx: List[float] = []
    ped_dy: List[float] = []
    veh_dx: List[float] = []
    veh_dy: List[float] = []
    ped_errors: List[float] = []
    veh_errors: List[float] = []

    for f1, f2 in pairs:
        p1 = img_dir / f"{f1:07d}.jpg"
        p2 = img_dir / f"{f2:07d}.jpg"
        if not p1.exists() or not p2.exists():
            continue

        im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            continue

        h, w = im1.shape
        s1 = cv2.resize(im1, (0, 0), fx=flow_scale, fy=flow_scale)
        s2 = cv2.resize(im2, (0, 0), fx=flow_scale, fy=flow_scale)

        flow_small = cv2.calcOpticalFlowFarneback(
            s1, s2, None, 0.5, 3, 15, 3, 5, 1.2, 0  # pyright: ignore[reportCallIssue,reportArgumentType]
        )
        flow = cv2.resize(flow_small, (w, h)) * (1.0 / flow_scale)

        t1_dict = frame_boxes[(sequence_name, f1)]
        t2_dict = frame_boxes[(sequence_name, f2)]
        common = set(t1_dict.keys()) & set(t2_dict.keys())

        for tid in common:
            o1 = t1_dict[tid]
            o2 = t2_dict[tid]
            cid = o1["category_id"]
            b1 = o1["bbox"]
            b2 = o2["bbox"]

            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
            gt_dx = float(c2[0] - c1[0])
            gt_dy = float(c2[1] - c1[1])
            gt_spd = float(np.hypot(gt_dx, gt_dy))

            x, y, bw, bh = map(int, b1)
            cw, ch = int(bw * 0.7), int(bh * 0.7)
            cx, cy = int(c1[0]), int(c1[1])
            x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
            y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
            patch = flow[y1:y2, x1:x2]
            if patch.size > 0:
                fl_dx = float(np.median(patch[:, :, 0]))
                fl_dy = float(np.median(patch[:, :, 1]))
            else:
                fl_dx, fl_dy = 0.0, 0.0

            fl_spd = float(np.hypot(fl_dx, fl_dy))
            err = float(np.hypot(fl_dx - gt_dx, fl_dy - gt_dy))

            if cid == 1:
                ped_gt_speeds.append(gt_spd)
                ped_flow_speeds.append(fl_spd)
                ped_dx.append(fl_dx)
                ped_dy.append(fl_dy)
                ped_errors.append(err)
            elif cid in [3, 4, 5, 7, 8]:
                veh_gt_speeds.append(gt_spd)
                veh_flow_speeds.append(fl_spd)
                veh_dx.append(fl_dx)
                veh_dy.append(fl_dy)
                veh_errors.append(err)

    p_gt = np.array(ped_gt_speeds) if ped_gt_speeds else np.array([0.0])
    v_gt = np.array(veh_gt_speeds) if veh_gt_speeds else np.array([0.0])
    p_fl = np.array(ped_flow_speeds) if ped_flow_speeds else np.array([0.0])
    v_fl = np.array(veh_flow_speeds) if veh_flow_speeds else np.array([0.0])
    p_err = np.array(ped_errors) if ped_errors else np.array([0.0])
    v_err = np.array(veh_errors) if veh_errors else np.array([0.0])

    return {
        "pair_count": len(pairs),
        "ped_count": len(ped_gt_speeds),
        "veh_count": len(veh_gt_speeds),
        "ped_gt_speeds": ped_gt_speeds,
        "veh_gt_speeds": veh_gt_speeds,
        "ped_flow_speeds": ped_flow_speeds,
        "veh_flow_speeds": veh_flow_speeds,
        "ped_dx": ped_dx,
        "ped_dy": ped_dy,
        "veh_dx": veh_dx,
        "veh_dy": veh_dy,
        "ped_errors": ped_errors,
        "veh_errors": veh_errors,
        "ped_median_speed": float(np.median(p_gt)),
        "ped_mean_speed": float(np.mean(p_gt)),
        "ped_p90_speed": float(np.percentile(p_gt, 90)),
        "veh_median_speed": float(np.median(v_gt)),
        "veh_mean_speed": float(np.mean(v_gt)),
        "veh_p90_speed": float(np.percentile(v_gt, 90)),
        "ped_flow_median_speed": float(np.median(p_fl)),
        "veh_flow_median_speed": float(np.median(v_fl)),
        "ped_median_error": float(np.median(p_err)),
        "veh_median_error": float(np.median(v_err)),
        "speed_ratio": float(np.median(v_gt) / (np.median(p_gt) + 1e-4)) if len(ped_gt_speeds) and len(veh_gt_speeds) else 1.0,
    }


def run_full_eda(
    coco_train: Dict[str, Any],
    dataset_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Universal single-call EDA suite summarizing VisDrone dataset characteristics.

    Computes:
    1. Class distribution & imbalance metrics
    2. Object scale & size categorization (COCO standards)
    3. Scene density & objects-per-frame variation
    4. Temporal IoU drift across frame strides (justifying k=3 subsampling)
    """
    print("=" * 60)
    print("Running VisDrone Comprehensive Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # 1. Class Distribution
    class_df, class_summary = compute_class_distribution(coco_train)
    print(f"\n[1. Class Distribution]: {class_summary['total_instances']:,} total target instances")
    print(f"   Majority Class: {class_summary['majority_class']} | Minority: {class_summary['minority_class']}")
    print(f"   Imbalance Ratio: {class_summary['imbalance_ratio']:.1f}x (extreme class imbalance)")

    # 2. Object Scale & Size Distribution
    size_df, size_summary = compute_size_metrics(coco_train)
    print(f"\n[2. Object Scale Distribution (COCO Standards)]:")
    print(f"   Small  (< 32x32 px):  {size_summary['small_percentage']:.1f}%")
    print(f"   Medium (32x32 to 96x96): {size_summary['medium_percentage']:.1f}%")
    print(f"   Large  (> 96x96 px):  {size_summary['large_percentage']:.1f}%")
    print(f"   Median Area: {size_summary['median_area']:.0f} px² (~{np.sqrt(size_summary['median_area']):.0f}x{np.sqrt(size_summary['median_area']):.0f} px)")

    # 3. Scene Density
    img_counts = Counter(a["image_id"] for a in coco_train["annotations"] if a.get("iscrowd", 0) == 0)
    counts_list = list(img_counts.values()) if img_counts else [0]
    density_summary = {
        "mean_density": float(np.mean(counts_list)),
        "median_density": float(np.median(counts_list)),
        "max_density": int(np.max(counts_list)),
        "p95_density": float(np.percentile(counts_list, 95)),
    }
    print(f"\n[3. Scene Density (Objects per Frame)]:")
    print(f"   Mean: {density_summary['mean_density']:.1f} | Median: {density_summary['median_density']:.0f} | 95th Percentile: {density_summary['p95_density']:.0f} | Max: {density_summary['max_density']}")

    # 4. Temporal Subsampling Overlap (Stride IoU Drift)
    stride_summary = compute_stride_iou_drift(coco_train, strides=[1, 2, 3, 5, 10], max_tracks=300)
    print(f"\n[4. Temporal Bounding Box IoU Drift (Frame Subsampling)]:")
    for s, res in stride_summary.items():
        print(f"   Stride k={s:2d}: Mean IoU = {res['mean_iou']:.3f} | Median IoU = {res['median_iou']:.3f}")

    # 5. Cross-Sequence Heterogeneity
    seq_df, seq_summary = compute_sequence_heterogeneity(coco_train)
    print(f"\n[5. Cross-Sequence Variation ({seq_summary['total_sequences']} Sequences)]:")
    print(f"   Density Range: {seq_summary['min_mean_density']:.1f} to {seq_summary['max_mean_density']:.1f} objects/frame")
    print(f"   Small Object % Range: {seq_summary['min_small_pct']:.1f}% to {seq_summary['max_small_pct']:.1f}%")
    print(f"   Vehicle % Range: {seq_summary['min_vehicle_pct']:.1f}% to {seq_summary['max_vehicle_pct']:.1f}%")

    return {
        "class_df": class_df,
        "class_summary": class_summary,
        "size_df": size_df,
        "size_summary": size_summary,
        "density_summary": density_summary,
        "stride_summary": stride_summary,
        "sequence_df": seq_df,
        "sequence_summary": seq_summary,
    }


def compute_sequence_heterogeneity(
    coco_data: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyzes cross-sequence variations in object density, scale, and class dominance.

    Returns:
        df: DataFrame summarizing frame count, density (min, mean, max), scale breakdown
            (small, medium, large %), and class dominance (% pedestrian vs % vehicle) per sequence.
        summary: Overall statistics of variation across sequences.
    """
    img_to_seq = {img["id"]: str(img["video"]) for img in coco_data["images"]}
    seq_frames: Dict[str, int] = defaultdict(int)
    for img in coco_data["images"]:
        seq_frames[str(img["video"])] += 1

    seq_anns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in coco_data["annotations"]:
        if a.get("iscrowd", 0) == 0:
            seq = img_to_seq.get(a["image_id"])
            if seq:
                seq_anns[seq].append(a)

    rows = []
    for seq in sorted(seq_frames.keys()):
        anns = seq_anns[seq]
        n_frames = seq_frames[seq]

        f_counts: Dict[int, int] = defaultdict(int)
        for a in anns:
            f_counts[a["image_id"]] += 1
        counts = [f_counts[img["id"]] for img in coco_data["images"] if str(img["video"]) == seq]

        areas = [float(a["bbox"][2] * a["bbox"][3]) for a in anns]
        n_total = len(areas)
        s_pct = (sum(1 for ar in areas if ar < 1024.0) / n_total * 100.0) if n_total else 0.0
        m_pct = (sum(1 for ar in areas if 1024.0 <= ar <= 9216.0) / n_total * 100.0) if n_total else 0.0
        l_pct = (sum(1 for ar in areas if ar > 9216.0) / n_total * 100.0) if n_total else 0.0

        ped_pct = (sum(1 for a in anns if a["category_id"] in [1, 2]) / n_total * 100.0) if n_total else 0.0
        veh_pct = (sum(1 for a in anns if a["category_id"] in [3, 4, 5, 6, 7, 8]) / n_total * 100.0) if n_total else 0.0

        mean_d = float(np.mean(counts)) if counts else 0.0
        min_d = int(np.min(counts)) if counts else 0
        max_d = int(np.max(counts)) if counts else 0

        rows.append({
            "sequence": seq,
            "frames": n_frames,
            "mean_density": round(mean_d, 1),
            "min_density": min_d,
            "max_density": max_d,
            "small_pct": round(s_pct, 1),
            "medium_pct": round(m_pct, 1),
            "large_pct": round(l_pct, 1),
            "pedestrian_pct": round(ped_pct, 1),
            "vehicle_pct": round(veh_pct, 1),
        })

    df = pd.DataFrame(rows)
    summary = {
        "total_sequences": len(df),
        "min_mean_density": float(df["mean_density"].min()) if len(df) else 0.0,
        "max_mean_density": float(df["mean_density"].max()) if len(df) else 0.0,
        "min_small_pct": float(df["small_pct"].min()) if len(df) else 0.0,
        "max_small_pct": float(df["small_pct"].max()) if len(df) else 0.0,
        "min_vehicle_pct": float(df["vehicle_pct"].min()) if len(df) else 0.0,
        "max_vehicle_pct": float(df["vehicle_pct"].max()) if len(df) else 0.0,
    }
    return df, summary


