import os
from pathlib import Path

import torch
import torch.nn as nn

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = self._load_backbone(model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    @staticmethod
    def _is_dinov2_repo(path: Path) -> bool:
        return (path / "hubconf.py").is_file() and (path / "dinov2").is_dir()

    @staticmethod
    def _find_local_dinov2_repo() -> Path | None:
        candidates: list[Path] = []

        # 1) Explicit override
        env_repo = os.environ.get("DINOV2_REPO_DIR")
        if env_repo:
            candidates.append(Path(env_repo))

        # 2) torch.hub default cache path
        try:
            hub_dir = Path(torch.hub.get_dir())
            candidates.append(hub_dir / "facebookresearch_dinov2_main")
        except Exception:
            pass

        # 3) Service cache path when mounted as:
        #    -v $WS/salad-svc/cache/torch:/workspace/.cache/torch
        this_file = Path(__file__).resolve()
        # .../salad-main/models/backbones/dinov2.py -> .../catkin_ws
        workspace_dir = this_file.parents[3]
        candidates.append(
            workspace_dir / "salad-svc" / "cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
        )

        seen: set[str] = set()
        for c in candidates:
            key = str(c)
            if key in seen:
                continue
            seen.add(key)
            if DINOv2._is_dinov2_repo(c):
                return c
        return None

    @staticmethod
    def _load_backbone(model_name: str):
        local_repo = DINOv2._find_local_dinov2_repo()
        if local_repo is not None:
            print(f"[SALAD DINOv2] Loading backbone from local repo: {local_repo}", flush=True)
            return torch.hub.load(repo_or_dir=str(local_repo), model=model_name, source="local")

        # Last fallback: load from github. Use skip_validation/trust_repo when available
        # to reduce unnecessary API calls.
        try:
            return torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
                model=model_name,
                source="github",
                trust_repo=True,
                skip_validation=True,
            )
        except TypeError:
            # Older torch versions may not support trust_repo / skip_validation
            return torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 backbone. No local dinov2 hub repo found and github "
                "download failed. Set DINOV2_REPO_DIR to a local facebookresearch_dinov2_main "
                "directory or provide torch hub cache before startup."
            ) from exc


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f
