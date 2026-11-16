"""Train the human-video BPP TIG adapter with explicit, reproducible update budgets.

C15/C30/C45 and Scratch start from the same external BPP initialization. Fine-tune
loads the corresponding scale weights with a fresh optimizer and train-only target statistics.
One sample here is a 16-action query window; Zero-WAM uses complete episodes.
Equal update counts therefore do not assert equal training FLOPs or epochs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, default=json_value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def to_device(value, device):
    if torch.is_tensor(value):
        # Frozen-feature batches use scalar-backed uint8 views for unused pixels.
        # Preserve that representation instead of copying expanded images over PCIe.
        if value.dtype == torch.uint8 and value.numel() and value.untyped_storage().nbytes() == 1:
            scalar = value[(0,) * value.ndim].reshape([1] * value.ndim).to(device, non_blocking=True)
            return scalar.expand(value.shape)
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value



@torch.no_grad()
def preclip_gradient_norms_by_module(model):
    """Read unique trainable gradients, grouped by the first four name parts."""
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        key = ".".join(name.split(".")[:4])
        grouped.setdefault(key, []).append(torch.linalg.vector_norm(gradient))
    return {key: float(torch.linalg.vector_norm(torch.stack(norms).double()))
            for key, norms in grouped.items()}

def learning_rate_factor(step, total, warmup, floor):
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    fraction = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return floor + (1.0 - floor) * .5 * (1.0 + math.cos(math.pi * fraction))


def make_parser():
    from .native_model import ACTION_NORMALIZATION_V2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action-normalization', required=True, choices=(ACTION_NORMALIZATION_V2,),
                        help='v2 uses full train min/max for fourteen joints and identity grippers; state normalization is unchanged')
    parser.add_argument('--regime', required=True, choices=('c15', 'c30', 'c45', 'scratch', 'finetune'))
    parser.add_argument('--pretraining-scale', choices=('c15', 'c30', 'c45'), help='Required lineage for target fine-tuning; inferred from source checkpoint if omitted')
    parser.add_argument('--prompt-source', choices=('human', 'robot'), default='human')
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path(os.environ.get('BPP_CHECKPOINT', '')))
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--bpp-repo', type=Path, default=Path(os.environ.get('BPP_REPOS_DIR', '/gaozt-test1/sxh/bpp/repos')) / 'behavior_prompting')
    parser.add_argument('--batch-size', type=int, default=8, help='Per GPU query windows')
    parser.add_argument('--feature-manifest', type=Path, help='Validated frozen ViT CLS cache for this exact training manifest')
    parser.add_argument('--fixed-vision-preprocessing', action='store_true', help='Explicit fixed224+CLIP normalization, without random image augmentation')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-updates', type=int, help='Default: C15 45000, C30 90000, C45 135000; Scratch/FT 3000')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--min-learning-rate', type=float, default=1e-6)
    parser.add_argument('--warmup-updates', type=int)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--init-seed', type=int, default=1)
    parser.add_argument('--pilot', action='store_true', help='Label small diagnostic runs; not official results')
    parser.add_argument('--cpu', action='store_true')
    return parser


def run(args):
    from .native_model import (NativeBPPModel, ACTION_NORMALIZATION_V2, resolve_action_normalization,
                               training_recipe_fields, create_ema, restore_ema)
    from .training_data import RobotVideoTrainingDataset, collate_training_batch

    if args.action_normalization != ACTION_NORMALIZATION_V2:
        raise ValueError('New training requires explicit v2 action normalization; use archived source for legacy runs')
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.feature_manifest and not args.fixed_vision_preprocessing:
        raise ValueError('Frozen CLS cache requires explicit --fixed-vision-preprocessing')
    if args.cpu and world != 1:
        raise ValueError('CPU diagnostics are single process only')
    pretraining_counts = {'c15': 3000, 'c30': 6000, 'c45': 9000}
    if args.max_updates is None:
        args.max_updates = {'c15': 45000, 'c30': 90000, 'c45': 135000}.get(args.regime, 3000)
    if args.warmup_updates is None:
        args.warmup_updates = 100 if args.regime in pretraining_counts else 10
    if min(args.max_updates, args.batch_size, args.log_every, args.checkpoint_every) <= 0:
        raise ValueError('Update, batch and logging budgets must be positive')
    if not 0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError('Expected 0 < minimum LR <= initial LR')
    if args.regime == 'finetune' and args.initialize_from is None and args.resume_from is None:
        raise ValueError('P+FT requires --initialize-from a completed C15 checkpoint')
    if args.regime != 'finetune' and args.initialize_from is not None:
        raise ValueError('C15/Scratch must use the common external BPP initialization')
    if args.initialize_from is not None and args.resume_from is not None:
        raise ValueError('Choose a new FT stage or resume an existing stage')
    if args.cpu:
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(1)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed + rank)
    initialization_metadata = None
    resume_metadata = None
    if args.resume_from:
        prior = torch.load(args.resume_from, map_location='cpu', weights_only=False, mmap=True)
        resolve_action_normalization(prior, args.action_normalization)
        resume_metadata = prior.get('metadata', {})
        if resume_metadata.get('regime') != args.regime:
            raise ValueError('Resume regime differs from saved training stage')
        initialization_metadata = resume_metadata.get('initialize_from_metadata')
        if args.regime == 'finetune' and args.pretraining_scale is None:
            args.pretraining_scale = resume_metadata.get('pretraining_scale')
        del prior
    if args.regime in pretraining_counts:
        if args.pretraining_scale not in (None, args.regime):
            raise ValueError('Pretraining scale and regime differ')
        args.pretraining_scale = args.regime
    elif args.regime == 'scratch':
        if args.pretraining_scale is not None:
            raise ValueError('Scratch is a shared baseline independent of C15/C30/C45')
    elif args.initialize_from is not None:
        origin = torch.load(args.initialize_from, map_location='cpu', weights_only=False, mmap=True)
        resolve_action_normalization(origin, args.action_normalization)
        previous_metadata = origin.get('metadata', {})
        origin_scale = previous_metadata.get('regime')
        initialization_metadata = {key: previous_metadata.get(key) for key in ['regime','train_step','max_updates','completed','pilot','prompt_source','pretraining_scale','experiment_identity','init_head_sha256','action_normalization','training_recipe','input_perturbation','use_ema','ema_config','evaluation_weights','resume_weights']}
        expected_source_updates = {'c15':45000,'c30':90000,'c45':135000}.get(origin_scale)
        if not args.pilot and (previous_metadata.get('train_step') != expected_source_updates or previous_metadata.get('max_updates') != expected_source_updates or not previous_metadata.get('completed') or previous_metadata.get('pilot') or previous_metadata.get('prompt_source') != args.prompt_source):
            raise ValueError('FT requires the complete corresponding formal pretraining budget and prompt source')
        if origin_scale not in pretraining_counts or args.pretraining_scale not in (None, origin_scale):
            raise ValueError('Fine-tuning must retain its C15/C30/C45 source lineage')
        args.pretraining_scale = origin_scale
        del origin
    elif args.pretraining_scale is None:
        raise ValueError('Resume fine-tuning with --pretraining-scale matching its checkpoint')
    if args.feature_manifest:
        from .feature_cache import CachedCLSDataset, collate_cls_batch, CachedCLSForward, configure_deterministic_vision
        dataset = CachedCLSDataset(args.data_manifest, args.feature_manifest, seed=args.seed)
        collate_fn = collate_cls_batch
    else:
        dataset = RobotVideoTrainingDataset(args.data_manifest, seed=args.seed)
        collate_fn = collate_training_batch
    if len(dataset) < world * args.batch_size:
        raise ValueError('Dataset smaller than one distributed batch')
    stats = dataset.normalizer_stats
    source = json.loads(args.data_manifest.read_text())
    if not args.pilot:
        selected_split = str(source.get('request', {}).get('split', source.get('split', source.get('protocol', '')))).lower()
        expected = args.regime if args.regime in pretraining_counts else 'scratch'
        if selected_split != expected:
            raise ValueError(f'Training manifest split {selected_split!r} does not match {expected!r}')
        expected_count = pretraining_counts.get(args.regime, 200)
        if source.get('query_episode_count') != expected_count or source.get('pilot') or source.get('is_pilot'):
            raise ValueError('Only the complete official episode budget can be reported as a full regime')
    actual_prompt_source = source.get('request', {}).get('prompt_source', 'robot')
    if actual_prompt_source != args.prompt_source:
        raise ValueError('Data prompt source differs from the requested experiment')
    pin_memory = device.type == 'cuda' and not bool(args.feature_manifest)
    provenance = {
        'schema': 'bpp_tig_train_run_v1', 'regime': args.regime,
        'prompt_source': actual_prompt_source, 'pretraining_scale': args.pretraining_scale,
        'action_normalization': args.action_normalization,
        **training_recipe_fields(args.action_normalization),
        'initialize_from_weights': 'ema' if args.initialize_from else (resume_metadata.get('initialize_from_weights') if resume_metadata else None),
        'feature_manifest': str(args.feature_manifest.resolve()) if args.feature_manifest else None,
        'fixed_vision_preprocessing': args.fixed_vision_preprocessing,
        'dataloader_pin_memory': pin_memory,
        'initialize_from_metadata': initialization_metadata,
        'pilot': args.pilot, 'training_sample': '16-action query window',
        'world_size': world, 'per_gpu_batch': args.batch_size,
        'global_batch': world * args.batch_size, 'max_updates': args.max_updates,
        'data_manifest': str(args.data_manifest.resolve()),
        'data_manifest_sha256': sha256(args.data_manifest),
        'external_checkpoint': str(args.checkpoint.resolve()),
        'external_checkpoint_sha256': sha256(args.checkpoint), 'num_workers_per_rank': args.num_workers,
        'initialize_from': str(args.initialize_from.resolve()) if args.initialize_from else (resume_metadata.get('initialize_from') if resume_metadata else None),
        'seed': args.seed, 'init_seed': args.init_seed,
        'freeze_vision': True, 'precision': 'bfloat16' if device.type == 'cuda' else 'float32',
        'learning_rate': args.learning_rate, 'min_learning_rate': args.min_learning_rate,
        'warmup_updates': args.warmup_updates, 'weight_decay': args.weight_decay,
        'clip_grad_norm': args.clip_grad_norm,
        'evaluation_checkpoint_selection': 'fixed final update; no held-out score selection',
        'zero_wam_budget_note': 'Matching update counts does not match sample units, epochs or FLOPs',
        'normalizer_source': stats.get('source'),
    }
    model_file = Path(__file__).with_name('native_model.py')
    provenance['trainer_sha256'] = sha256(Path(__file__))
    provenance['native_model_sha256'] = sha256(model_file)
    provenance['training_data_sha256'] = sha256(Path(__file__).with_name('training_data.py'))
    if args.feature_manifest:
        provenance['feature_manifest_sha256'] = sha256(args.feature_manifest)
        provenance['feature_cache_source_sha256'] = sha256(Path(__file__).with_name('feature_cache.py'))
    if args.initialize_from:
        provenance['initialize_from_sha256'] = sha256(args.initialize_from)
    elif resume_metadata and resume_metadata.get('initialize_from_sha256'):
        provenance['initialize_from_sha256'] = resume_metadata['initialize_from_sha256']
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True, default=json_value).encode()).hexdigest()
    provenance['experiment_identity'] = identity
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        manifest_path = args.output_dir / 'training_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != provenance:
            raise ValueError('Output directory belongs to a different training configuration')
        atomic_json(manifest_path, provenance)
    if world > 1:
        dist.barrier()
    checkpoint = None
    if args.resume_from:
        model = NativeBPPModel.from_training_checkpoint(args.resume_from, device=str(device), bpp_repo=args.bpp_repo, action_normalization=args.action_normalization, load_ema=False)
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False, mmap=True)
    elif args.initialize_from:
        initialized = torch.load(args.initialize_from, map_location='cpu', weights_only=False, mmap=True)
        previous = initialized.get('metadata', {})
        if previous.get('regime') != args.pretraining_scale or (not args.pilot and (previous.get('pilot') or not previous.get('completed'))):
            raise ValueError('P+FT must initialize from the completed corresponding C15/C30/C45 checkpoint')
        del initialized
        model = NativeBPPModel.from_training_checkpoint(args.initialize_from, device=str(device), bpp_repo=args.bpp_repo, freeze_vision=True, action_normalization=args.action_normalization, load_ema=True)
        model.mark_loaded_weights_as_training_initialization()
        model.set_normalizer_stats(stats)
    else:
        model = NativeBPPModel(args.checkpoint, bpp_repo=args.bpp_repo, device=str(device), normalizer_stats=stats, freeze_vision=True, init_seed=args.init_seed, action_normalization=args.action_normalization)
    if args.resume_from:
        saved_metadata = checkpoint.get('metadata', {})
        if saved_metadata.get('experiment_identity') != identity:
            raise ValueError('Resume checkpoint configuration differs')
    if args.fixed_vision_preprocessing:
        from .feature_cache import configure_deterministic_vision
        fixed_preprocessing = configure_deterministic_vision(model)
    else:
        fixed_preprocessing = None
    cached_forward = CachedCLSForward(model, args.feature_manifest) if args.feature_manifest else None
    # Model construction isolates CPU initialization but may touch CUDA RNG.
    # Set per-rank training streams after initialization; resumes restore saved streams below.
    if not args.resume_from:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed + rank)
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    factor = lambda step: learning_rate_factor(step, args.max_updates, args.warmup_updates, args.min_learning_rate / args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start_step = epoch = batch_skip = 0
    if checkpoint:
        saved = checkpoint.get('additional_state', {})
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start_step, epoch, batch_skip = saved['train_step'], saved['epoch'], saved['batch_in_epoch']
        state = saved.get('rank_rng_states', [None] * world)[rank]
        if state:
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch_cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(state['torch_cuda'], device)
    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else model
    ema = create_ema(model)
    if checkpoint:
        restore_ema(model, ema, checkpoint.get('ema'), expected_step=start_step)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=pin_memory, persistent_workers=False, generator=torch.Generator().manual_seed(args.seed + rank), multiprocessing_context='spawn' if args.num_workers else None)
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = last_log = time.monotonic()
    log_start_step = step = start_step
    seen_queries = start_step * world * args.batch_size
    supervised_action_elements = checkpoint.get('additional_state', {}).get('supervised_action_elements', 0) if checkpoint else 0
    step_seconds = []
    peak_memory = 0
    metadata = dict(provenance, parameter_audit=model.parameter_audit, init_head_sha256=model.init_head_sha256,
                    trainable_parameters=sum(p.numel() for p in trainable), visual_preprocessing=fixed_preprocessing)
    if rank == 0:
        atomic_json(args.output_dir / 'model_initialization.json', metadata)

    def save(name, next_batch):
        state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
                 'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        if world > 1:
            states = [None] * world
            dist.all_gather_object(states, state)
        else:
            states = [state]
        if rank == 0:
            temporary = args.output_dir / (name + '.tmp')
            final_metadata = dict(metadata, train_step=step, completed=step >= args.max_updates,
                                  elapsed_wall_seconds=time.monotonic() - started)
            model.save_checkpoint(temporary, metadata=final_metadata, additional_state={
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'rank_rng_states': states, 'train_step': step, 'epoch': epoch, 'batch_in_epoch': next_batch,
                'supervised_action_elements': supervised_action_elements,
            }, ema=ema)
            temporary.replace(args.output_dir / name)
        if world > 1:
            dist.barrier()

    try:
        while step < args.max_updates:
            sampler.set_epoch(epoch)
            dataset.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                if batch_index < batch_skip:
                    continue
                step_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                batch = to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss = cached_forward.forward(batch, forward=distributed_model) if cached_forward else distributed_model(batch)
                local_valid = batch['action_mask'].sum().to(dtype=torch.float32)
                global_valid = local_valid.clone()
                if world > 1:
                    dist.all_reduce(global_valid)
                    loss = loss * (world * local_valid / global_valid)
                finite = torch.isfinite(loss).to(dtype=torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite.item()):
                    raise FloatingPointError(f'Nonfinite training loss at update {step}')
                loss.backward()
                log_this_step = ((step + 1) % args.log_every == 0 or
                                 step == start_step or step + 1 == args.max_updates)
                module_grad_norms = preclip_gradient_norms_by_module(model) if rank == 0 and log_this_step else None
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                ema.step(model.parameters())
                scheduler.step()
                step += 1
                seen_queries += world * args.batch_size
                supervised_action_elements += int(global_valid.item()) * 16
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
                duration = time.monotonic() - step_start
                step_seconds.append(duration)
                if len(step_seconds) > 200:
                    step_seconds.pop(0)
                if log_this_step:
                    mean_loss = loss.detach().float()
                    if world > 1:
                        dist.all_reduce(mean_loss)
                        mean_loss /= world
                    if rank == 0:
                        now = time.monotonic()
                        per_update = (now - last_log) / max(1, step - log_start_step)
                        row = {'step': step, 'max_updates': args.max_updates, 'epoch': epoch,
                               'query_windows_seen': seen_queries, 'supervised_action_elements': supervised_action_elements, 'loss': mean_loss.item(),
                               'grad_norm_rank0': float(grad_norm), 'learning_rate': scheduler.get_last_lr()[0],
                               'preclip_grad_norm_by_module_rank0': module_grad_norms,
                               'grad_clip_coefficient_rank0': min(1.0, args.clip_grad_norm / (float(grad_norm) + 1e-6)),
                               'wall_seconds_per_update': per_update, 'compute_seconds_per_update_rank0': duration,
                               'estimated_remaining_seconds': (args.max_updates - step) * per_update,
                               'peak_memory_allocated_rank0': peak_memory, 'pilot': args.pilot}
                        with (args.output_dir / 'training.jsonl').open('a') as log:
                            log.write(json.dumps(row) + '\n')
                        atomic_json(args.output_dir / 'progress.json', dict(row, status='training'))
                        print(json.dumps(row), flush=True)
                        last_log, log_start_step = now, step
                quit_tensor = torch.tensor(int(stop['requested']), device=device)
                if world > 1:
                    dist.all_reduce(quit_tensor, op=dist.ReduceOp.MAX)
                interrupted = bool(quit_tensor.item())
                if (step % args.checkpoint_every == 0 and step < args.max_updates) or interrupted:
                    save('latest.pt', batch_index + 1)
                if step >= args.max_updates or interrupted:
                    save('final.pt' if step >= args.max_updates else 'interrupted.pt', batch_index + 1)
                    if rank == 0:
                        atomic_json(args.output_dir / 'completion.json', {
                            'status': 'completed' if step >= args.max_updates else 'interrupted', 'pilot': args.pilot,
                            'completed_updates': step, 'target_updates': args.max_updates,
                            'wall_seconds': time.monotonic() - started, 'init_head_sha256': model.init_head_sha256,
                            'peak_memory_allocated_rank0': peak_memory, 'experiment_identity': identity})
                    return
            batch_skip = 0
            epoch += 1
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    run(make_parser().parse_args())


if __name__ == '__main__':
    main()
