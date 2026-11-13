"""setup_utils.py
Standardized environment configuration, hardware detection, Google Drive
persistence, and automated Kaggle competition dataset acquisition.
"""

import os
import sys
import json
import random
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import yaml
import torch
import numpy as np

COMPETITION_NAME = "multiple-object-video-detection-hw-2-ucu-ai-2026"

CANDIDATE_DATASET_DIRS = [
    Path("dataset"),
    Path("/content/dataset"),
    Path("/content/ai-lab2/dataset"),
    Path.home() / ".cache" / "kagglehub" / "competitions" / COMPETITION_NAME,
    Path("/kaggle/input") / COMPETITION_NAME,
    Path("/kaggle/input") / "multiple-object-video-detection",
    Path("data"),
]


def use_native_batchnorm() -> None:
    """Runs every nn.BatchNorm* layer with PyTorch's own GPU kernel instead of MIOpen (convolutions keep MIOpen).

    On AMD GPUs (ROCm builds of PyTorch) MIOpen's batch norm can fail with
    `RuntimeError: miopenStatusUnknownError` (seen with D-FINE + AMP on an RX 7800 XT under Windows).
    F.batch_norm only picks MIOpen while torch.backends.cudnn.enabled is True, so the flag is switched off
    for the duration of each BN call; the backward pass uses the kernel chosen in the forward pass. Idempotent.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    if getattr(_BatchNorm.forward, "_native_bn", False):
        return
    original_forward = _BatchNorm.forward

    def forward(self, input):
        if input.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                return original_forward(self, input)
        return original_forward(self, input)

    forward._native_bn = True
    _BatchNorm.forward = forward


def setup_kaggle_credentials(
    kaggle_json_path: Union[str, Path] = "kaggle.json",
    backup_json_str: Optional[str] = None,
) -> bool:
    """Configures Kaggle API authentication.
    
    Tries in order:
    1. backup_json_str if provided (ideal for pasting credentials directly into Colab).
    2. Local kaggle_json_path file (e.g. ./kaggle.json).
    3. Existing ~/.kaggle/kaggle.json.
    
    Sets KAGGLE_USERNAME and KAGGLE_KEY environment variables and creates ~/.kaggle/kaggle.json.
    
    Returns:
        True if credentials were successfully configured, False otherwise.
    """
    creds: Optional[Dict[str, str]] = None
    source_desc = ""

    # 1. Try backup string first if non-empty and not a placeholder
    if backup_json_str and backup_json_str.strip():
        try:
            parsed = json.loads(backup_json_str.strip())
            u = parsed.get("username", "")
            k = parsed.get("key", "")
            if u and k and "USERNAME_HERE" not in u and "LEGACY_API_KEY_HERE" not in k and "YOUR_" not in u:
                creds = parsed
                source_desc = "backup string in notebook"
        except json.JSONDecodeError as e:
            print(f"[Kaggle]: Warning: Failed to parse backup_json_str as JSON: {e}")

    # 2. Try local file if not yet loaded
    if creds is None:
        p = Path(kaggle_json_path)
        if p.exists():
            try:
                with open(p, "r") as f:
                    creds = json.load(f)
                source_desc = f"local file ({p})"
            except Exception as e:
                print(f"[Kaggle]: Warning: Failed to read {p}: {e}")

    # 3. Try existing ~/.kaggle/kaggle.json
    dest_path = Path.home() / ".kaggle" / "kaggle.json"
    if creds is None and dest_path.exists():
        try:
            with open(dest_path, "r") as f:
                creds = json.load(f)
            source_desc = f"existing {dest_path}"
        except Exception as e:
            print(f"[Kaggle]: Warning: Failed to read {dest_path}: {e}")

    if creds and "username" in creds and "key" in creds:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as f:
            json.dump(creds, f)
        dest_path.chmod(0o600)

        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
        print(f"[Kaggle]: Credentials successfully configured from {source_desc} (User: {creds['username']})")
        return True

    print("[Kaggle]: No valid Kaggle credentials found. Please provide 'kaggle.json' or set backup_json_str.")
    return False


def setup_environment(
    seed: int = 42,
    mount_gdrive: bool = True,
    gdrive_folder: str = "ai_lab2_checkpoints"
) -> Dict[str, Any]:
    """
    Initializes deterministic seeds, configures hardware profile,
    and optionally mounts Google Drive if executing within Google Colab.
    On ROCm (AMD) builds of PyTorch, batch norm is moved off MIOpen (see use_native_batchnorm).
    """
    if getattr(torch.version, "hip", None) is not None:
        use_native_batchnorm()

    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Hardware detection
    is_cuda = torch.cuda.is_available()
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if is_cuda:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Environment]: CUDA GPU ({gpu_name}, {vram_gb:.1f} GB VRAM) -> Mixed Precision (AMP Enabled)")
    elif is_mps:
        device = torch.device("mps")
        print("[Environment]: Apple Silicon (Metal Performance Shaders - MPS)")
    else:
        device = torch.device("cpu")
        print("[Environment]: CPU Only")

    # Colab Google Drive persistence
    gdrive_dir: Optional[Path] = None
    if mount_gdrive and Path("/content").exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            gdrive_dir = Path("/content/drive/MyDrive") / gdrive_folder
            gdrive_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Google Drive]: Mounted! Checkpoints will persist in: {gdrive_dir}")
        except Exception as e:
            print(f"[Google Drive]: Mount skipped or unavailable ({type(e).__name__})")

    # Import hardware config from src.dataset if present
    hw_cfg: Dict[str, Any] = {}
    try:
        from src.dataset import get_hardware_config
        hw_cfg = get_hardware_config()
    except ImportError:
        hw_cfg = {
            "device": device,
            "is_cuda": is_cuda,
            "is_mps": is_mps,
            "use_amp": is_cuda,
            "batch_size": 2 if is_cuda else 1,
            "input_size": (1024, 1024) if is_cuda else (640, 640),
            "num_workers": 4 if is_cuda else 0
        }

    in_colab = Path("/content").exists()
    checkpoint_dir = gdrive_dir if gdrive_dir is not None else Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        "device": device,
        "is_cuda": is_cuda,
        "is_mps": is_mps,
        "hw_cfg": hw_cfg,
        "in_colab": in_colab,
        "checkpoint_dir": checkpoint_dir,
        "gdrive_dir": checkpoint_dir,
        "seed": seed,
    }


def ensure_dataset(
    custom_dir: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
    backup_kaggle_json: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Locates or automatically downloads the competition dataset.
    Returns:
        dataset_dir: Path to dataset root containing 'train' and 'val'
        coco_train: Parsed COCO json dictionary for training split
        coco_val: Parsed COCO json dictionary for validation split
        class_names: List of target class name strings
    """
    resolved_dir: Optional[Path] = None

    # Check custom path first
    if custom_dir is not None:
        p = Path(custom_dir)
        if p.exists() and (p / "train" / "annotations_coco.json").exists():
            resolved_dir = p

    # Check known candidate locations
    if resolved_dir is None:
        for c in CANDIDATE_DATASET_DIRS:
            if c.exists() and (c / "train" / "annotations_coco.json").exists():
                resolved_dir = c
                break

    # If still not found, download via Kaggle
    if resolved_dir is None and auto_download:
        print(f"--> Dataset not found locally. Preparing download for '{COMPETITION_NAME}'...")

        setup_kaggle_credentials(backup_json_str=backup_kaggle_json)

        zip_name = f"{COMPETITION_NAME}.zip"
        target_dir = Path("/content/dataset") if Path("/content").exists() else Path("dataset")

        if Path(zip_name).exists():
            print(f"--> Found existing {zip_name}. Extracting into {target_dir}...")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_name, "r") as zf:
                zf.extractall(target_dir)
            resolved_dir = target_dir
            print(f"--> Dataset extracted successfully to: {resolved_dir}")
        else:
            try:
                print("--> Downloading competition archive via Kaggle CLI...")
                subprocess.run(
                    [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION_NAME],
                    check=True
                )
                print(f"--> Extracting archive into {target_dir}...")
                target_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_name, "r") as zf:
                    zf.extractall(target_dir)
                resolved_dir = target_dir
                print(f"--> Dataset extracted successfully to: {resolved_dir}")
            except Exception as e_cli:
                print(f"--> Kaggle CLI returned: {e_cli}. Attempting kagglehub fallback...")
                try:
                    import kagglehub
                    dataset_path_str = kagglehub.competition_download(COMPETITION_NAME)
                    resolved_dir = Path(dataset_path_str)
                    print(f"--> kagglehub download successful: {resolved_dir}")
                except Exception as e_hub:
                    raise RuntimeError(
                        f"Failed to automatically retrieve dataset. Please ensure Kaggle API "
                        f"credentials are set (~/.kaggle/kaggle.json) or run 'kaggle competitions download -c {COMPETITION_NAME}' manually.\n"
                        f"Errors: CLI ({e_cli}), Hub ({e_hub})"
                    )

    if resolved_dir is None:
        raise FileNotFoundError(
            f"Could not locate dataset root directory with train/annotations_coco.json in {CANDIDATE_DATASET_DIRS}"
        )

    print(f"[Dataset]: Resolved dataset root at -> {resolved_dir}")

    # Load annotations
    train_anno_path = resolved_dir / "train" / "annotations_coco.json"
    val_anno_path = resolved_dir / "val" / "annotations_coco.json"

    with open(train_anno_path, "r") as f:
        coco_train = json.load(f)
    with open(val_anno_path, "r") as f:
        coco_val = json.load(f)

    # Load class names from data.yaml if available, else derive from annotations
    yaml_path = resolved_dir / "train" / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)
        class_names = data_yaml.get("names", [])
    else:
        cats = sorted(coco_train.get("categories", []), key=lambda c: c["id"])
        class_names = [c["name"] for c in cats]

    return resolved_dir, coco_train, coco_val, class_names


def resolve_storage(
    runtime: str,
    subdir: str,
    local_root: Union[str, Path] = "runs",
    gdrive_folder: str = "ai_lab2_checkpoints",
) -> Path:
    """Returns a persistent directory for checkpoints and logs of one checkpoint/experiment.

    Args:
        runtime: "colab" (mounts Google Drive, stores under MyDrive/<gdrive_folder>)
            or "local" (stores under <local_root>). Chosen explicitly, never auto-detected.
        subdir: Sub-directory for this notebook, e.g. "cp5_rtdetr".
        local_root: Root directory used when runtime == "local".
        gdrive_folder: Folder inside MyDrive used when runtime == "colab".
    """
    if runtime == "colab":
        from google.colab import drive  # type: ignore[import-not-found]

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive") / gdrive_folder
    elif runtime == "local":
        root = Path(local_root)
    else:
        raise ValueError(f"runtime must be 'colab' or 'local', got {runtime!r}")

    out = (root / subdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Storage]: {runtime} -> {out}")
    return out
