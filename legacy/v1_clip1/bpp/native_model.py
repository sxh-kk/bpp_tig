"""Trainable TIG-native BPP: one RGB demonstration, dual-arm joint actions.

This module is separate from the frozen LIBERO OSC adapter. Demonstration poses,
qpos, grippers and actions have no input path. Native actions retain dataset units.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

SCHEMA = "bpp_tig_native_v1"
GROUPS = (("left_arm_qpos", 0, 7), ("left_fingers", 7, 9),
          ("right_arm_qpos", 9, 16), ("right_fingers", 16, 18))


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def native_model_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep 5 current/query modalities, retaining the 15360D FiLM condition."""
    config = copy.deepcopy(dict(source_config))
    shape = {
        "image_resolution": 224, "prompt_sequence_length": 768,
        "max_prompt_full_demos": 1, "prompt_chunk_n_actions": 20,
        "pad_end_prompt_actions": "zeros", "max_sequence_length": 1000,
        "use_prompting": True, "prompt_sample_mode": "pair",
        "action": {"horizon": 16, "down_sample_steps": 1, "type": "low_dim",
                   "shape": [16], "rep": "pd_joint_pos"},
        "obs": {"agentview_rgb": {
            "horizon": 2, "down_sample_steps": 1, "shape": [3, 224, 224],
            "type": "rgb", "ignore_by_policy": False,
            "prompt_type": "observation", "include_in_prompt_current_obs": True,
        }},
    }
    for key, start, stop in GROUPS:
        shape["obs"][key] = {
            "horizon": 2, "down_sample_steps": 1, "shape": [stop - start],
            "type": "low_dim", "ignore_by_policy": False,
            "prompt_type": "ignore", "include_in_prompt_current_obs": True,
        }

    def replace(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "shape_meta":
                    node[key] = copy.deepcopy(shape)
                else:
                    replace(value)
        elif isinstance(node, list):
            for value in node:
                replace(value)
    replace(config)
    tokenizer = config["obs_encoder"]["obs_encoder"]
    tokenizer["ignore_prompt_action"] = True
    tokenizer["ignore_prompt_proprio"] = True
    tokenizer["use_pool_modality_pos_embed"] = False
    tokenizer["obs_encoder"]["pretrained"] = False
    tokenizer["obs_encoder"]["frozen"] = False
    config["obs_encoder"]["use_attention_sink"] = True
    config["obs_encoder"]["num_attention_sink_tokens"] = 4
    return config


def _instantiate(config, bpp_repo):
    if bpp_repo is not None and str(Path(bpp_repo).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(bpp_repo).resolve()))
    import hydra
    from omegaconf import OmegaConf
    from behavior_prompting.train_network.model.vision.transformer_obs_encoder import TransformerObsEncoder
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with contextlib.redirect_stdout(io.StringIO()), patch.object(TransformerObsEncoder, "init_weights", lambda self: None):
        return hydra.utils.instantiate(OmegaConf.create(config))


def _tensor_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _warm_start(policy, source_state):
    source = copy.copy(source_state)
    source.pop("_extra_training_split_info", None)
    with contextlib.redirect_stdout(io.StringIO()):
        policy._apply_backward_compatibility_renames(source)
    target = policy.state_dict()
    transfer, mappings = {}, []
    for name, value in target.items():
        if not isinstance(value, torch.Tensor) or name.startswith("normalizer."):
            continue
        source_name = name
        for side in ("left", "right"):
            source_name = source_name.replace(f".key_projection_map.{side}_fingers.", ".key_projection_map.gripper_states.")
        if source_name in source and tuple(source[source_name].shape) == tuple(value.shape):
            transfer[name] = source[source_name]
            if source_name != name:
                mappings.append({"target": name, "source": source_name})
    with contextlib.redirect_stdout(io.StringIO()):
        result = policy.load_state_dict(transfer, strict=False)
    intended_missing = set(target) - set(transfer) - {"_extra_training_split_info"}
    if set(result.missing_keys) != intended_missing or result.unexpected_keys:
        raise RuntimeError("Unexpected partial checkpoint-load mismatch")
    transferred, fresh = [], []
    for name, parameter in policy.named_parameters():
        row = {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        (transferred if name in transfer else fresh).append(row)
    return {
        "unique_parameters": sum(p.numel() for p in policy.parameters()),
        "pretrained_parameters": sum(row["numel"] for row in transferred),
        "new_parameters": sum(row["numel"] for row in fresh),
        "new_parameter_details": fresh,
        "explicit_finger_mappings": mappings,
        "source_normalizer_loaded": False,
        "source_training_split_loaded": False,
        "compatibility_means": "parameter-shape compatibility; modality meanings still change",
    }, [row["name"] for row in fresh if row["numel"]]


class NativeBPPModel(nn.Module):
    def __init__(self, checkpoint, bpp_repo=None, device="cuda:0", normalizer_stats=None,
                 freeze_vision=True, amp_dtype="bfloat16", init_seed=1, warm_start=True):
        super().__init__()
        import dill
        from omegaconf import OmegaConf
        from diffusers import DDPMScheduler
        if amp_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unknown amp_dtype")
        payload = torch.load(Path(checkpoint), map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
        is_native = payload.get("schema") == SCHEMA
        if not is_native and not warm_start and freeze_vision:
            raise ValueError("Freezing a randomly initialized visual backbone is not a meaningful scratch setup")
        self.amp_dtype = amp_dtype
        self.init_seed = int(payload.get("init_seed", init_seed)) if is_native else int(init_seed)
        self.freeze_vision = bool(freeze_vision)
        self.model_config = copy.deepcopy(payload["model_config"]) if is_native else native_model_config(
            OmegaConf.to_container(payload["cfg"].model, resolve=True)
        )
        # No global CUDA RNG changes: isolate CPU module initialization only.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.init_seed)
            self.policy = _instantiate(self.model_config, bpp_repo)
        if is_native:
            self.parameter_audit = copy.deepcopy(payload["parameter_audit"])
            self._new_parameter_names = list(payload.get("new_parameter_names", []))
        elif warm_start:
            self.parameter_audit, self._new_parameter_names = _warm_start(self.policy, payload["state_dicts"]["model"])
        else:
            self._new_parameter_names = [name for name, p in self.policy.named_parameters() if p.numel()]
            self.parameter_audit = {
                "unique_parameters": sum(p.numel() for p in self.policy.parameters()),
                "pretrained_parameters": 0, "new_parameters": sum(p.numel() for p in self.policy.parameters()),
                "initialization": "scratch, including vision", "source_normalizer_loaded": False,
            }
        self.init_head_sha256 = payload["init_head_sha256"] if is_native else _tensor_hash(
            [(name, p) for name, p in self.policy.named_parameters() if name in self._new_parameter_names]
        )
        self.register_buffer("state_scale", torch.ones(18))
        self.register_buffer("state_offset", torch.zeros(18))
        self.register_buffer("action_scale", torch.ones(16))
        self.register_buffer("action_offset", torch.zeros(16))
        selected_stats = normalizer_stats if normalizer_stats is not None else payload.get("normalizer_stats")
        self.normalizer_stats = None
        self.set_normalizer_stats(selected_stats)
        if is_native:
            self.load_state_dict(payload["state_dict"], strict=True)
            # Explicit new training statistics override saved buffers only when requested.
            if normalizer_stats is not None:
                self.set_normalizer_stats(normalizer_stats)
        self.train_noise_scheduler = DDPMScheduler.from_config(self.policy.noise_scheduler.config)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(parameter.numel() > 0)
        if self.freeze_vision:
            for module in self._vision_backbones():
                module.requires_grad_(False)
        self.parameter_audit["trainable_parameters"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.parameter_audit["frozen_parameters"] = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        self.parameter_audit["global_condition_dim"] = int(self.policy.obs_feature_dim)
        self.parameter_audit["freeze_vision"] = self.freeze_vision
        self.config = {"model": self.model_config, "freeze_vision": self.freeze_vision,
                       "amp_dtype": self.amp_dtype, "init_seed": self.init_seed,
                       "observation": "one RGB plus four groups of current qpos18", "action_dim": 16}
        self.to(device)
        self.train(True)

    def _vision_backbones(self):
        encoder = self.policy.obs_encoder.obs_encoder.prompt_obs_encoder
        return encoder.key_model_map.values()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_vision", False) and hasattr(self, "policy"):
            for module in self._vision_backbones():
                module.eval()
        return self

    @property
    def device(self):
        return self.state_scale.device

    def _amp(self):
        if self.device.type != "cuda" or self.amp_dtype == "float32":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=getattr(torch, self.amp_dtype))

    def set_normalizer_stats(self, stats):
        self.normalizer_stats = _plain(stats)
        with torch.no_grad():
            for prefix in ("state", "action"):
                scale, offset = getattr(self, prefix + "_scale"), getattr(self, prefix + "_offset")
                if stats is None:
                    scale.fill_(1); offset.zero_()
                    continue
                if not stats.get("source"):
                    raise ValueError("Normalizer stats must identify their train-only source")
                bounds = stats.get("selected_bounds", {}).get(prefix)
                if bounds is None:
                    raw = stats[prefix]
                    bounds = {"min": raw["q01" if stats.get("mode") == "quantile" else "min"],
                              "max": raw["q99" if stats.get("mode") == "quantile" else "max"]}
                low = torch.as_tensor(bounds["min"], dtype=scale.dtype, device=scale.device)
                high = torch.as_tensor(bounds["max"], dtype=scale.dtype, device=scale.device)
                if low.shape != scale.shape or high.shape != scale.shape or not torch.isfinite(low).all() or not torch.isfinite(high).all() or (high < low).any():
                    raise ValueError(f"Invalid {prefix} normalizer bounds")
                constant = (high - low) < 1e-6
                scale.copy_(torch.where(constant, torch.ones_like(low), 2 / (high - low).clamp_min(1e-6)))
                offset.copy_(torch.where(constant, -low, -1 - low * scale))

    def _rgb(self, value):
        value = torch.as_tensor(value, device=self.device)
        value = value.float() / 255 if value.dtype == torch.uint8 else value.float()
        if value.ndim != 5 or tuple(value.shape[2:]) != (3, 224, 224):
            raise ValueError("RGB must be [batch,time,3,224,224]")
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("Floating RGB must be finite and in [0,1]")
        return value

    def _current(self, current):
        if "qpos_gripper_states" in current:
            if set(current) != {"agentview_rgb", "qpos_gripper_states"}:
                raise ValueError("Native current obs must contain RGB and measured qpos18 only")
            qpos = torch.as_tensor(current["qpos_gripper_states"], dtype=torch.float32, device=self.device)
        else:
            if set(current) != {"agentview_rgb", *(key for key, _, _ in GROUPS)}:
                raise ValueError("Current obs must contain the five named native modalities")
            qpos = torch.cat([torch.as_tensor(current[key], dtype=torch.float32, device=self.device)
                              for key, _, _ in GROUPS], dim=-1)
        rgb = self._rgb(current["agentview_rgb"])
        if tuple(qpos.shape) != (rgb.shape[0], 2, 18) or rgb.shape[1] != 2 or not torch.isfinite(qpos).all():
            raise ValueError("Current history requires RGB[B,2,...] and finite qpos[B,2,18]")
        qpos = qpos * self.state_scale + self.state_offset
        return {"agentview_rgb": rgb, **{key: qpos[..., start:stop] for key, start, stop in GROUPS}}

    def _prompt(self, rgb, mask=None):
        rgb = self._rgb(rgb)
        if not 1 <= rgb.shape[1] <= 50:
            raise ValueError("Prompt must contain between 1 and 50 RGB frames")
        mask = torch.zeros(rgb.shape[:2], dtype=torch.bool, device=self.device) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if tuple(mask.shape) != tuple(rgb.shape[:2]) or mask.all(dim=1).any():
            raise ValueError("Prompt padding mask must match RGB and leave a visible frame")
        return {"obs": {"agentview_rgb": rgb}, "metadata": {"mask": mask}}

    def forward(self, batch):
        current = self._current(batch.get("obs", batch.get("current")))
        prompt = batch.get("prompt")
        if prompt is not None:
            if set(prompt) - {"agentview_rgb", "mask"}:
                raise ValueError("Demonstration prompt may contain RGB and padding mask only")
            rgb, prompt_mask = prompt["agentview_rgb"], prompt.get("mask", batch.get("prompt_mask"))
        else:
            rgb, prompt_mask = batch["prompt_rgb"], batch.get("prompt_mask")
        current["prompt"] = self._prompt(rgb, prompt_mask)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        batch_size = current["agentview_rgb"].shape[0]
        if tuple(action.shape) != (batch_size, 16, 16):
            raise ValueError("Action targets must be [batch,16,16]")
        valid = batch.get("action_mask", batch.get("action_valid_mask"))
        valid = torch.ones((batch_size, 16), dtype=torch.bool, device=self.device) if valid is None else torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        if tuple(valid.shape) != (batch_size, 16) or not valid.any(dim=1).all():
            raise ValueError("Action mask must be [batch,16], True for valid targets")
        if not torch.isfinite(action[valid]).all():
            raise ValueError("Valid action targets must be finite")
        normalized = action * self.action_scale + self.action_offset
        normalized = normalized.masked_fill(~valid[..., None], 0)
        noise = torch.randn_like(normalized).masked_fill(~valid[..., None], 0)
        timesteps = torch.randint(0, self.train_noise_scheduler.config.num_train_timesteps,
                                  (batch_size,), device=self.device)
        noisy = self.train_noise_scheduler.add_noise(normalized, noise, timesteps)
        noisy = noisy.masked_fill(~valid[..., None], 0)
        with self._amp():
            condition, _ = self.policy.obs_encoder(current)
            predicted = self.policy.model(noisy, timesteps, global_cond=condition)
            squared = (predicted.float() - noise).square()
            return (squared * valid[..., None]).sum() / (valid.sum() * 16)

    @torch.inference_mode()
    def encode_prompt(self, prompt_rgb, prompt_mask=None):
        if self.training:
            raise RuntimeError("Call eval() before caching prompt features; training must recompute them")
        with self._amp():
            tokens, mask = self.policy.obs_encoder({"prompt": self._prompt(prompt_rgb, prompt_mask)})
        return {"tokens": tokens, "mask": mask}

    @torch.inference_mode()
    def predict_action(self, current, prompt_rgb=None, prompt_features=None, prompt_mask=None):
        if (prompt_rgb is None) == (prompt_features is None):
            raise ValueError("Provide exactly one of prompt_rgb or prompt_features")
        was_training = self.training
        self.eval()
        encoder = self.policy.obs_encoder
        try:
            features = self.encode_prompt(prompt_rgb, prompt_mask) if prompt_features is None else prompt_features
            if isinstance(features, torch.Tensor):
                features = {"tokens": features, "mask": prompt_mask}
            tokens = features["tokens"].to(self.device)
            if tokens.ndim != 3 or tokens.shape[-1] != 768 or not 1 <= tokens.shape[1] <= 50:
                raise ValueError("Cached prompt must contain [prompt_batch,frames,768]")
            mask = features.get("mask")
            encoder.prompt_tokens_cache = tokens
            encoder.prompt_mask_cache = None if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            encoder.is_prompted = True
            with self._amp():
                values = self._current(current)
                condition, _ = encoder(values)
                shape = (values["agentview_rgb"].shape[0], 16, 16)
                zeros = torch.zeros(shape, device=self.device, dtype=torch.float32)
                normalized = self.policy.conditional_sample(zeros, torch.zeros_like(zeros, dtype=torch.bool), global_cond=condition)
            return (normalized.float() - self.action_offset) / self.action_scale
        finally:
            encoder.reset()
            self.train(was_training)

    def export_state(self, metadata=None, additional_state=None):
        # BasePolicy adds non-tensor LIBERO split metadata; never propagate it.
        state = {key: value.detach().cpu() for key, value in self.state_dict().items() if isinstance(value, torch.Tensor)}
        return {"schema": SCHEMA, "model_config": copy.deepcopy(self.model_config),
                "config": copy.deepcopy(self.config), "normalizer_stats": copy.deepcopy(self.normalizer_stats),
                "state_dict": state, "parameter_audit": copy.deepcopy(self.parameter_audit),
                "init_seed": self.init_seed, "init_head_sha256": self.init_head_sha256,
                "new_parameter_names": list(self._new_parameter_names), "freeze_vision": self.freeze_vision,
                "metadata": {} if metadata is None else copy.deepcopy(metadata),
                "additional_state": {} if additional_state is None else additional_state}

    def save_checkpoint(self, path, metadata=None, additional_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(self.export_state(metadata, additional_state), temporary)
        temporary.replace(path)
        return path

    @classmethod
    def from_training_checkpoint(cls, path, device="cuda:0", bpp_repo=None,
                                 freeze_vision=None, amp_dtype="bfloat16", normalizer_stats=None):
        if freeze_vision is None:
            import dill
            payload = torch.load(Path(path), map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
            freeze_vision = payload.get("freeze_vision", True)
        return cls(path, bpp_repo=bpp_repo, device=device, normalizer_stats=normalizer_stats,
                   freeze_vision=freeze_vision, amp_dtype=amp_dtype)
