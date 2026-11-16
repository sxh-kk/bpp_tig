"""Frozen BPP-LIBERO inference with a single RGB-only demonstration.

Only prompt tokenization is adapted. The checkpoint's current-observation tokens,
queries, diffusion head, and normalization statistics retain their original shapes.
The returned 10D actions are *robosuite OSC inputs*, not meter/radian deltas.
"""
from __future__ import annotations

import contextlib
import copy
import io
import math
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping

import numpy as np


def select_prompt_frame_indices(
    frame_count: int, fps: float, prompt_hz: float, max_frames: int,
) -> np.ndarray:
    """Sample by time, include both endpoints, and keep the whole clip if capped."""
    if frame_count < 1 or max_frames < 1:
        raise ValueError("A video and its prompt must each contain at least one frame")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("The reference video's actual fps must be positive")
    if not math.isfinite(prompt_hz) or prompt_hz <= 0:
        raise ValueError("prompt_hz must be positive")
    # Frame indices, rather than decoded RGB values, define a shared sampling rule.
    times = np.arange(0.0, frame_count / fps, 1.0 / prompt_hz)
    indices = np.minimum(np.floor(times * fps + 1e-8).astype(np.int64), frame_count - 1)
    indices = np.unique(np.concatenate((indices, [frame_count - 1])))
    if len(indices) > max_frames:
        indices = indices[np.rint(np.linspace(0, len(indices) - 1, max_frames)).astype(np.int64)]
    return indices


def install_video_only_tokenizer(tokenizer: Any, image_key: str = "agentview_rgb") -> None:
    """Select only the external RGB token before the original learned pooling.

    No dummy action or demonstration state is constructed. Current observations use
    the untouched original forward method. No parameter is removed or resized.
    """
    if getattr(tokenizer, "_bpp_tig_video_only", False):
        raise RuntimeError("The video-only tokenizer has already been installed")
    if tokenizer.concat_prompt_and_receding or tokenizer.merge_prompt_tokens != "obs_and_action":
        raise ValueError("Expected the released separate-prompt, pooled BPP-LIBERO architecture")
    if image_key not in tokenizer.prompt_rgb_keys:
        raise ValueError(f"Checkpoint has no prompt RGB key {image_key!r}")
    original_forward = tokenizer.forward
    original_names = tokenizer.prompt_obs_encoder.get_obs_names(
        rgb_keys=tokenizer.prompt_rgb_keys, low_dim_keys=tokenizer.prompt_low_dim_keys,
    )
    modality_index = original_names.index(image_key)

    def video_only_forward(this: Any, obs_dict: Mapping[str, Any]):
        if "prompt" not in obs_dict:
            return original_forward(obs_dict)
        if any(key in obs_dict for key in this.current_obs_keys):
            raise ValueError("Encode video with prepare_video before predicting current observations")
        prompt = obs_dict["prompt"]
        if set(prompt) - {"obs", "metadata"} or set(prompt["obs"]) != {image_key}:
            raise ValueError("The demonstration prompt must contain only external RGB video")
        features, _ = this.prompt_obs_encoder(
            prompt["obs"], rgb_keys=[image_key], low_dim_keys=[], is_prompt=True,
        )
        if features.ndim != 4 or features.shape[2] != 1 or features.shape[-1] != this.n_emb:
            raise RuntimeError(f"Unexpected single-view prompt encoding shape {tuple(features.shape)}")
        batch, length, _, dim = features.shape
        tokens = features.reshape(batch * length, 1, dim)
        if this.pool_modality_pos_embed is not None:
            tokens = tokens + this.pool_modality_pos_embed[modality_index:modality_index + 1]
        tokens = this.prompt_attention_pool(tokens).reshape(batch, length, dim)
        mask = prompt.get("metadata", {}).get("mask")
        if mask is not None and tuple(mask.shape) != (batch, length):
            raise ValueError("Prompt mask must have shape [batch, prompt_frames]")
        return tokens, mask

    tokenizer.forward = MethodType(video_only_forward, tokenizer)
    tokenizer._bpp_tig_video_only = True


def _load_frozen_policy(checkpoint_path: Path, bpp_repo: Path | None):
    """Load official cfg/state strictly; never restore an optimizer or fit data."""
    import dill
    import hydra
    from omegaconf import OmegaConf
    import torch

    if bpp_repo is not None:
        repo = str(bpp_repo.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    # Match the official entry point for older configs retaining eval resolvers.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    # Workspace checkpoints also contain optimizer tensors; map their storages
    # lazily and touch only the model weights required by strict loading.
    payload = torch.load(
        checkpoint_path, map_location="cpu", pickle_module=dill,
        weights_only=False, mmap=True,
    )
    if not isinstance(payload, dict) or "cfg" not in payload or "model" not in payload.get("state_dicts", {}):
        raise ValueError("Expected an official BPP workspace checkpoint containing cfg and state_dicts.model")
    cfg = copy.deepcopy(payload["cfg"])
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    state = copy.copy(payload["state_dicts"]["model"])
    notes: list[str] = []
    tokenizer_cfg = model_cfg.obs_encoder.obs_encoder
    if "PairPromptTransformerTokenizer" not in str(tokenizer_cfg.get("_target_", "")):
        raise ValueError("This adapter requires the official PairPromptTransformerTokenizer checkpoint")
    pool_exists = any(key.endswith(".pool_modality_pos_embed") for key in state)
    if tokenizer_cfg.get("use_pool_modality_pos_embed", None) != pool_exists:
        notes.append(f"Matched legacy use_pool_modality_pos_embed={pool_exists} to saved parameter keys")
    tokenizer_cfg.use_pool_modality_pos_embed = pool_exists
    # Do not download the CLIP initialization: all visual weights are in this ckpt.
    vision_cfg = tokenizer_cfg.obs_encoder
    if not str(vision_cfg.model_name).startswith("vit_"):
        raise ValueError("The released LIBERO ViT checkpoint is required")
    vision_cfg.pretrained = False
    vision_cfg.frozen = False
    sink_keys = [key for key in state if key.endswith(".attention_sink_tokens")]
    model_cfg.obs_encoder.use_attention_sink = bool(sink_keys)
    if sink_keys:
        model_cfg.obs_encoder.num_attention_sink_tokens = int(state[sink_keys[0]].shape[1])
    shape_meta = OmegaConf.to_container(model_cfg.shape_meta, resolve=True)
    if list(shape_meta["action"]["shape"]) != [10] or shape_meta["action"].get("rep") != "delta":
        raise ValueError("Expected a single-arm 10D LIBERO OSC delta checkpoint")
    # BasePolicy.load_state_dict applies the official legacy key renames itself.
    # Its verbose per-key migration prints are retained as a count in metadata.
    migration_output = io.StringIO()
    from unittest.mock import patch
    from behavior_prompting.train_network.model.vision.transformer_obs_encoder import TransformerObsEncoder
    # The official from-scratch initializer does not handle ViT patch Conv2d.
    # Every visual parameter is restored strictly below, so random reinitializing
    # it is unnecessary. The original class method is restored after construction.
    with contextlib.redirect_stdout(migration_output), patch.object(
        TransformerObsEncoder, "init_weights", lambda self: None,
    ):
        policy = hydra.utils.instantiate(model_cfg)
        policy.load_state_dict(state, strict=True)
    notes.append("Skipped unused visual random initialization before strict checkpoint restoration")
    migrated = migration_output.getvalue().count("Renamed key:")
    if migrated:
        notes.append(f"Official BasePolicy migrated {migrated} legacy state keys")
    policy.eval()
    policy.requires_grad_(False)
    return policy, shape_meta, notes


class FrozenBPPModel:
    """One frozen BPP checkpoint and one cached video prompt.

    ``prepare_video`` takes RGB frames [T,H,W,3] plus their true fps, or a video
    filename. ``get_action`` takes official named observations with explicit
    [B,history,...] dimensions and returns numpy [B,prediction_horizon,10]. The
    runner must maintain history at each actual control step, including while an
    already predicted action chunk is being executed.
    """

    def __init__(
        self, checkpoint_path: str | Path, device: str = "cuda:0", *,
        bpp_repo: str | Path | None = None, amp_dtype: str = "bfloat16",
        prompt_hz: float = 1.0, max_prompt_frames: int | None = None,
    ):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is false")
        if amp_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("amp_dtype must be bfloat16, float16 or float32")
        self.amp_dtype = amp_dtype
        self.prompt_hz = float(prompt_hz)
        if not math.isfinite(self.prompt_hz) or self.prompt_hz <= 0:
            raise ValueError("prompt_hz must be positive")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.policy, self.shape_meta, notes = _load_frozen_policy(
            self.checkpoint_path, None if bpp_repo is None else Path(bpp_repo),
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.requires_grad_(False)
        self.tokenizer = self.policy.obs_encoder.obs_encoder
        self.prompt_image_key = "agentview_rgb"
        self.current_keys = tuple(self.tokenizer.current_obs_keys)
        self.action_horizon = int(self.policy.action_horizon)
        self.action_dim = int(self.policy.action_dim)
        positional_limit = int(self.policy.obs_encoder.prompt_pos_emb.shape[1])
        if max_prompt_frames is not None and not 1 <= max_prompt_frames <= positional_limit:
            raise ValueError(f"max_prompt_frames must be in [1, {positional_limit}]")
        self.max_prompt_frames = positional_limit if max_prompt_frames is None else int(max_prompt_frames)
        image_shape = self.shape_meta["obs"][self.prompt_image_key]["shape"]
        self.image_size = tuple(int(x) for x in image_shape[-2:])
        install_video_only_tokenizer(self.tokenizer, self.prompt_image_key)
        self._prepared = False
        self._encode_calls = 0
        self.prompt_metadata: dict[str, Any] | None = None
        self.model_info: dict[str, Any] = {
            "variant": "BPP video-only inference variant",
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_bytes": self.checkpoint_path.stat().st_size,
            "frozen": True,
            "strict_state_loading": True,
            "demo_modalities": [self.prompt_image_key],
            "demo_action_tokens": 0,
            "demo_proprioception_tokens": 0,
            "current_observation_keys": list(self.current_keys),
            "current_observation_horizons": {
                key: int(self.shape_meta["obs"][key]["horizon"]) for key in self.current_keys
            },
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "action_representation": "LIBERO robosuite OSC delta control input, rotation_6d",
            "diffusion_steps": int(self.policy.num_inference_steps),
            "max_prompt_frames": self.max_prompt_frames,
            "prompt_hz": self.prompt_hz,
            "compatibility_notes": notes,
        }

    def _autocast(self):
        if self.device.type != "cuda" or self.amp_dtype == "float32":
            return contextlib.nullcontext()
        return self._torch.autocast(
            device_type="cuda", dtype=getattr(self._torch, self.amp_dtype),
        )

    def _image_tensor(self, value: Any, *, prompt: bool = False):
        torch = self._torch
        if not isinstance(value, torch.Tensor):
            value = np.ascontiguousarray(value)
        tensor = torch.as_tensor(value)
        if prompt:
            if tensor.ndim != 4 or tensor.shape[-1] != 3:
                raise ValueError("Reference frames must be RGB [T,H,W,3]")
            tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        elif tensor.ndim != 5 or tensor.shape[2] != 3:
            raise ValueError("Current images must be RGB [B,history,3,H,W]")
        byte_input = tensor.dtype == torch.uint8
        tensor = tensor.to(dtype=torch.float32)
        if byte_input:
            tensor = tensor / 255.0
        if not torch.isfinite(tensor).all() or tensor.min().item() < 0 or tensor.max().item() > 1:
            raise ValueError("RGB input must be uint8 or finite floating point in [0,1]")
        batch, length = tensor.shape[:2]
        if tuple(tensor.shape[-2:]) != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.reshape(batch * length, 3, *tensor.shape[-2:]),
                size=self.image_size, mode="bilinear", align_corners=False, antialias=True,
            ).reshape(batch, length, 3, *self.image_size)
        # Official eval transforms and CLIP mean/std are applied by the encoder.
        return tensor.to(self.device)

    def _read_video(self, path: str | Path, fps: float | None):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video filenames require opencv-python-headless; RGB arrays need no video decoder") from exc
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open reference video {path}")
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) if fps is None else fps)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = select_prompt_frame_indices(count, actual_fps, self.prompt_hz, self.max_prompt_frames)
            images = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode reference frame {index} in {path}")
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return np.stack(images), indices, actual_fps, count
        finally:
            cap.release()

    def prepare_video(self, video: Any, fps: float | None = None) -> dict[str, Any]:
        """Encode one video once; no demonstration action/state API exists."""
        torch = self._torch
        if isinstance(video, Mapping):
            raise ValueError("Pass only RGB video, not a demonstration dictionary")
        if isinstance(video, (str, Path)):
            frames, indices, actual_fps, frame_count = self._read_video(video, fps)
        else:
            if fps is None:
                raise ValueError("The actual source fps is required for an RGB frame array")
            if isinstance(video, torch.Tensor):
                video = video.detach().cpu().numpy()
            frames = np.asarray(video)
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError("Reference video must be RGB [T,H,W,3]")
            frame_count = len(frames)
            actual_fps = float(fps)
            indices = select_prompt_frame_indices(frame_count, actual_fps, self.prompt_hz, self.max_prompt_frames)
            frames = frames[indices]
        self.reset()
        with torch.inference_mode(), self._autocast():
            rgb = self._image_tensor(frames, prompt=True)
            # Avoid DiffusionUnetPolicy.prompt(), whose legacy normalization helper
            # unconditionally accesses prompt['action']. No dummy trajectory enters
            # this path: the prompt consists of exactly one RGB key and a mask.
            prompt_normalizer = self.policy.normalizer.get_prompt_normalizer()
            normalized_rgb = prompt_normalizer.normalize({self.prompt_image_key: rgb})
            self.policy.obs_encoder.prompt({
                "obs": normalized_rgb,
                "metadata": {"mask": torch.zeros((1, len(indices)), dtype=torch.bool, device=self.device)},
            })
        self._prepared = True
        self._encode_calls += 1
        self.prompt_metadata = {
            "prompt_frame_count": len(indices),
            "source_frame_count": frame_count,
            "source_fps": actual_fps,
            "selected_frame_indices": indices.tolist(),
            "selected_timestamps_seconds": (indices / actual_fps).tolist(),
            "prompt_hz_requested": self.prompt_hz,
            "max_prompt_frames": self.max_prompt_frames,
            "prompt_modalities": [self.prompt_image_key],
            "demo_action_tokens": 0,
            "demo_proprioception_tokens": 0,
            "prompt_encode_calls": 1,
        }
        return dict(self.prompt_metadata)

    def get_action(self, current_obs: Mapping[str, Any]) -> np.ndarray:
        """Return [B,16,10] OSC input chunks; controller scaling is external."""
        if not self._prepared:
            raise RuntimeError("Call prepare_video before get_action")
        torch = self._torch
        values = dict(current_obs)
        if "wrist_rgb" in values:
            if "eye_in_hand_rgb" in values:
                raise ValueError("Provide wrist_rgb or eye_in_hand_rgb, not both")
            values["eye_in_hand_rgb"] = values.pop("wrist_rgb")
        if set(values) != set(self.current_keys):
            raise ValueError(f"Current observation keys must be {self.current_keys}; received {tuple(values)}")
        prepared = {}
        batch_size = None
        with torch.inference_mode(), self._autocast():
            for key in self.current_keys:
                spec = self.shape_meta["obs"][key]
                if spec["type"] == "rgb":
                    tensor = self._image_tensor(values[key])
                else:
                    value = values[key]
                    if not isinstance(value, torch.Tensor):
                        value = np.ascontiguousarray(value)
                    tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite current proprioception in {key}")
                expected = (int(spec["horizon"]), *[int(x) for x in spec["shape"]])
                if tensor.ndim != len(expected) + 1 or tuple(tensor.shape[1:]) != expected:
                    raise ValueError(f"{key} requires [B,{','.join(map(str, expected))}], got {tuple(tensor.shape)}")
                if batch_size is None:
                    batch_size = tensor.shape[0]
                if tensor.shape[0] != batch_size or batch_size < 1:
                    raise ValueError("All current observations must have the same nonempty batch")
                prepared[key] = tensor
            output = self.policy.predict_action(prepared)["action"]
            expected_output = (batch_size, self.action_horizon, self.action_dim)
            if tuple(output.shape) != expected_output or not torch.isfinite(output).all():
                raise RuntimeError(f"Invalid BPP action output, expected {expected_output}")
            return output.detach().to(device="cpu", dtype=torch.float32).numpy()

    def reset(self, seed: int | None = None) -> None:
        """Clear cached prompt; the runner separately clears action/history queues."""
        self.policy.reset()
        self._prepared = False
        self.prompt_metadata = None
        if seed is not None:
            self._torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                self._torch.cuda.manual_seed_all(int(seed))
