import os
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

_THIS_DIR = Path(__file__).resolve().parent
_SERVICE_ROOT = _THIS_DIR.parent
_DEFAULT_SALAD_MAIN = _SERVICE_ROOT.parent / "salad-main"
_SALAD_MAIN = Path(os.environ.get("SALAD_MAIN_PATH", _DEFAULT_SALAD_MAIN))

if _SALAD_MAIN.is_dir() and str(_SALAD_MAIN) not in sys.path:
    sys.path.insert(0, str(_SALAD_MAIN))

from vpr_model import VPRModel

# Enable cuDNN benchmark for fixed-size inputs
torch.backends.cudnn.benchmark = True


_device = "cuda" if torch.cuda.is_available() else "cpu"
_model: torch.nn.Module | None = None
_autocast_dtype: torch.dtype = torch.float32

_DEFAULT_IMG_SIZE = (322, 322)  # 14 * 23 to align with DINOv2 patch size
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _log(msg: str) -> None:
    print(f"[SALAD compute] {msg}", flush=True)


def _parse_image_size(raw: str | None) -> Tuple[int, int]:
    if not raw:
        return _DEFAULT_IMG_SIZE
    cleaned = raw.lower().replace("x", ",").replace(" ", "")
    parts: Iterable[str] = [p for p in cleaned.split(",") if p]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        _log(f"Failed to parse SALAD_IMAGE_SIZE={raw}, fallback to default {_DEFAULT_IMG_SIZE}")
        return _DEFAULT_IMG_SIZE
    if not nums:
        return _DEFAULT_IMG_SIZE
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (nums[0], nums[1])


_IMAGE_SIZE = _parse_image_size(os.environ.get("SALAD_IMAGE_SIZE"))
_TRANSFORM = T.Compose(
    [
        T.Resize(_IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ]
)


def _maybe_set_autocast_dtype(device: str) -> None:
    global _autocast_dtype
    if device != "cuda":
        _autocast_dtype = torch.float32
        _log("CPU mode: autocast disabled.")
        return
    major, _ = torch.cuda.get_device_capability()
    _autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
    _log(f"CUDA autocast dtype: {_autocast_dtype}.")


def _resolve_ckpt_path(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_ckpt = os.environ.get("SALAD_CKPT")
    if env_ckpt:
        candidates.append(Path(env_ckpt))
    candidates.append(_SALAD_MAIN / "weights" / "dino_salad.ckpt")
    candidates.append(_SERVICE_ROOT / "weights" / "dino_salad.ckpt")

    for path in candidates:
        if path.is_file():
            return path
    return None


def _build_model(ckpt_path: Path | None) -> torch.nn.Module:
    if ckpt_path is not None:
        _log(f"Loading SALAD weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )
        model.load_state_dict(state)
    else:
        if _SALAD_MAIN.is_dir():
            repo_dir = _SALAD_MAIN
            source = "local"
            _log(f"No local ckpt found, loading dinov2_salad from repo: {repo_dir}")
        else:
            repo_dir = "serizba/salad"
            source = "github"
            _log("No local ckpt and SALAD repo missing; falling back to torch.hub (requires network).")
        model = torch.hub.load(repo_or_dir=str(repo_dir), model="dinov2_salad", source=source, pretrained=True)

    model.eval()
    model.to(_device)
    return model


def init_model(ckpt_path: str | None = None) -> torch.nn.Module:
    """
    Initialize SALAD model (prefers GPU if available). Cached after first call.
    """
    global _model, _device
    if _model is not None:
        return _model

    ckpt = _resolve_ckpt_path(ckpt_path)
    _log(f"Init SALAD model, target device: {_device}, image size: {_IMAGE_SIZE}.")

    try:
        _model = _build_model(ckpt)
    except RuntimeError as err:
        if _device == "cuda" and "out of memory" in str(err).lower():
            _log(f"CUDA OOM, falling back to CPU: {err}")
            torch.cuda.empty_cache()
            _device = "cpu"
            _model = _build_model(ckpt)
        else:
            _log(f"Model load failed: {err}")
            raise

    _maybe_set_autocast_dtype(_device)
    _log(f"SALAD model ready on device: {_device}.")
    return _model


def _preprocess(pil_imgs: Sequence[Image.Image]) -> torch.Tensor:
    if not pil_imgs:
        raise ValueError("At least one image is required.")
    tensors = [_TRANSFORM(img.convert("RGB")) for img in pil_imgs]
    batch = torch.stack(tensors, dim=0)
    return batch.to(device=_device, dtype=torch.float32)


def _forward(batch: torch.Tensor) -> torch.Tensor:
    model = init_model()
    with torch.inference_mode():
        if _device == "cuda":
            with torch.cuda.amp.autocast(dtype=_autocast_dtype):
                outputs = model(batch)
        else:
            outputs = model(batch)
    return torch.nn.functional.normalize(outputs, dim=1).to(torch.float32)


def extract_embeddings(pil_imgs: Sequence[Image.Image]) -> np.ndarray:
    """
    Batch compute SALAD embeddings. Returns float32 numpy array, shape [N, D].
    """
    batch = _preprocess(list(pil_imgs))
    outputs = _forward(batch)
    arr = outputs.detach().cpu().numpy().astype(np.float32)
    _log(f"Embeddings ready: count={arr.shape[0]}, dim={arr.shape[1]}, device={_device}.")
    return arr


def extract_embedding(pil_img: Image.Image) -> np.ndarray:
    return extract_embeddings([pil_img])[0]


__all__ = ["init_model", "extract_embedding", "extract_embeddings"]
