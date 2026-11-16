"""Optional per-frame frozen ViT CLS cache, before every trainable BPP layer.

NativeBPPModel and its parameter/state names stay unchanged. This path explicitly
requires deterministic224 preprocessing, an eval/frozen ViT, CLS aggregation,
and an identity visual projection. It never caches learned prompt pooling.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
import copy,hashlib,json,os,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .training_data import RobotVideoTrainingDataset,collate_training_batch

SCHEMA='bpp_tig_frozen_vit_cls_v1'
PREPROCESSING='resize224_in_rgb_source_cache_then_identity_spatial_transforms_and_original_clip_norm'

def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def _visual(model):
    tokenizer=model.policy.obs_encoder.obs_encoder
    if tokenizer.prompt_obs_encoder is not tokenizer.receding_obs_encoder:
        raise ValueError('CLS caching requires the same prompt/current visual encoder')
    encoder=tokenizer.prompt_obs_encoder
    if set(encoder.key_model_map)!= {'agentview_rgb'}:
        raise ValueError('Expected exactly one agentview visual backbone')
    return encoder,encoder.key_model_map['agentview_rgb']

def configure_deterministic_vision(model):
    """Explicit opt-in: identity train/eval crops/jitter, retain original CLIP norm.

    Synchronize exported reconstruction config, and record the removed transforms.
    This is a deliberate training protocol change from LIBERO random augmentation.
    """
    encoder,_=_visual(model)
    old={}
    for field in ('key_train_transform_map','key_eval_transform_map','key_non_prompt_train_image_transform_map'):
        mapping=getattr(encoder,field);old[field]={key:repr(value) for key,value in mapping.items()}
        if any(any(True for _ in value.parameters()) or any(True for _ in value.buffers()) for value in mapping.values()):
            raise ValueError('Cannot silently remove a transform with parameters/buffers')
    for field in old:
        for key in getattr(encoder,field):getattr(encoder,field)[key]=nn.Identity()
    config=model.model_config['obs_encoder']['obs_encoder']['obs_encoder']
    previous={key:copy.deepcopy(config.get(key)) for key in ('train_image_transforms','eval_image_transforms','non_prompt_train_image_transforms')}
    for key in previous:config[key]=None
    note={'schema':'bpp_tig_fixed_visual_preprocessing_v1','preprocessing':PREPROCESSING,'removed_runtime_transforms':old,'removed_config_transforms':previous,'equivalent_to_original_random_augmentation':False}
    model.config['model']=model.model_config;model.config['visual_preprocessing']=note
    model._bpp_fixed_visual_preprocessing=note
    return note

class FrozenCLSExtractor:
    def __init__(self,model):
        self.model=model;self.encoder,self.backbone=_visual(model)
        self.validate()
        h=hashlib.sha256()
        for name,value in sorted(self.backbone.state_dict().items()):
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        norm=self.encoder.model_normalization_transform
        import timm,torchvision
        self.signature={'schema':SCHEMA,'backbone_class':type(self.backbone).__name__,'model_name':self.encoder.model_name,'backbone_parameters':sum(p.numel() for p in self.backbone.parameters()),'backbone_sha256':h.hexdigest(),'feature_aggregation':'cls','feature_width':768,'storage_dtype':'float32','amp_dtype':model.amp_dtype,'device_type':model.device.type,'autocast_enabled':model.device.type=='cuda' and model.amp_dtype!='float32','clip_mean':list(norm.mean),'clip_std':list(norm.std),'preprocessing':PREPROCESSING,'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'timm_version':timm.__version__,'torch_cuda_version':torch.version.cuda,'float32_matmul_precision':torch.get_float32_matmul_precision(),'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}
        self.fingerprint=_identity(self.signature);self.output_dtype=None

    def validate(self):
        if not self.model.freeze_vision or self.backbone.training or any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError('CLS cache requires a frozen eval-mode visual backbone')
        if self.encoder.feature_aggregation!='cls' or not self.encoder.model_name.startswith('vit') or not self.encoder.use_vision_norm:
            raise ValueError('Expected original CLIP-normalized ViT CLS boundary')
        if not isinstance(self.encoder.key_projection_map['agentview_rgb'],nn.Identity):
            raise ValueError('Expected identity RGB key projection')
        for field in ('key_train_transform_map','key_eval_transform_map','key_non_prompt_train_image_transform_map'):
            if any(not isinstance(value,nn.Identity) for value in getattr(self.encoder,field).values()):
                raise ValueError('Fixed CLS cache requires explicit deterministic vision opt-in; random/crop transforms remain active')

    @torch.no_grad()
    def extract(self,rgb,batch_frames=32):
        """Actual uint8/floating RGB[B,T,3,224,224] -> CPU float32 CLS[B,T,768]."""
        self.validate()
        if batch_frames<1:raise ValueError('batch_frames must be positive')
        value=torch.as_tensor(rgb)
        if value.ndim!=5 or tuple(value.shape[2:])!=(3,224,224):raise ValueError('Expected RGB[B,T,3,224,224]')
        batch,length=value.shape[:2];flat=value.reshape(-1,3,224,224);parts=[]
        for start in range(0,len(flat),batch_frames):
            pixels=self.model._rgb(flat[start:start+batch_frames].unsqueeze(0)).flatten(0,1)
            pixels=self.encoder.model_normalization_transform(pixels)
            with self.model._amp():features=self.encoder.aggregate_feature(self.backbone(pixels))[:,0,:]
            if tuple(features.shape)!=(len(pixels),768):raise ValueError('Unexpected CLS shape')
            if not torch.isfinite(features).all():raise ValueError('Non-finite frozen CLS')
            dtype=str(features.dtype).split('.')[-1]
            if self.output_dtype not in (None,dtype):raise ValueError('Visual output dtype changed')
            self.output_dtype=dtype;parts.append(features.detach().float().cpu())
        return torch.cat(parts).reshape(batch,length,768)


def prepare_cls_cache(data_manifest,extractor,cache_root,*,batch_frames=32,max_query_episodes=1,max_prompt_episodes=1,shard_index=0,num_shards=1):
    """Extract selected episodes; None limits explicitly request all episodes.

    Defaults intentionally prepare only a tiny pilot. Root coordinates full runs.
    Artifacts deduplicate across scales/regimes by frozen encoder and RGB content.
    """
    source_path=Path(data_manifest);source=json.loads(source_path.read_text());root=Path(cache_root)
    if not root.resolve().is_relative_to(Path(os.environ.get('BPP_ROOT','/gaozt-test1/sxh/bpp')).resolve()):raise ValueError('Feature cache must stay in project storage')
    if num_shards<1 or not 0<=shard_index<num_shards:raise ValueError('Invalid feature cache shard')
    records={};started=time.monotonic()
    for role,limit in (('query',max_query_episodes),('prompt',max_prompt_episodes)):
        selected=source[role+'_episodes'] if limit is None else source[role+'_episodes'][:limit]
        selected=selected[shard_index::num_shards]
        result=[]
        for record in selected:
            artifact=record['rgb'];image=Path(artifact['path']);stat=image.stat()
            if stat.st_size!=artifact['size_bytes'] or stat.st_mtime_ns!=artifact.get('mtime_ns',stat.st_mtime_ns):raise ValueError('RGB cache signature changed '+str(image))
            values=np.load(image,mmap_mode='r',allow_pickle=False)
            if values.dtype!=np.uint8 or values.ndim!=4 or tuple(values.shape[1:])!=(224,224,3):raise ValueError('Invalid RGB cache array')
            spec={'encoder':extractor.fingerprint,'rgb_sha256':artifact['sha256'],'rgb_shape':list(values.shape)};key=_identity(spec);target=root/'cls'/extractor.fingerprint/record['repo_id']/(key+'.npy');sidecar=target.with_suffix('.json');cached=json.loads(sidecar.read_text()) if sidecar.exists() else None
            reusable=False
            if cached and target.is_file() and cached.get('source')==spec:
                feature=np.load(target,mmap_mode='r',allow_pickle=False);reusable=feature.shape==(len(values),768) and feature.dtype==np.float32 and target.stat().st_size==cached['size_bytes'] and _file_hash(target)==cached['sha256']
            if not reusable:
                # Copy only one chunk from the source mmap at a time.
                features=[]
                for begin in range(0,len(values),batch_frames):
                    rgb=np.ascontiguousarray(values[begin:begin+batch_frames].transpose(0,3,1,2))[None]
                    features.append(extractor.extract(rgb,batch_frames=batch_frames)[0].numpy())
                array=np.concatenate(features);target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
                with temporary.open('wb') as f:np.save(f,array,allow_pickle=False)
                temporary.replace(target);cached={'path':str(target.resolve()),'shape':list(array.shape),'dtype':'float32','backbone_output_dtype':extractor.output_dtype,'size_bytes':target.stat().st_size,'sha256':_file_hash(target),'source':spec};_json(sidecar,cached)
            else:extractor.output_dtype=cached['backbone_output_dtype']
            result.append({'repo_id':record['repo_id'],'episode_index':record['episode_index'],'source_rgb':artifact,'cls':cached})
            print(f'CLS {role} {record["repo_id"]}:{record["episode_index"]} {len(values)} frames',flush=True)
        records[role+'_episodes']=result
    suffix='' if num_shards==1 else f'.shard{shard_index:03d}-of-{num_shards:03d}'
    output=root/'manifests'/f'{source_path.parent.name}-{extractor.fingerprint[:16]}{suffix}.json'
    manifest={'schema':SCHEMA,'feature_fingerprint':extractor.fingerprint,'encoder_signature':extractor.signature,'visual_preprocessing_optin':copy.deepcopy(extractor.model.config.get('visual_preprocessing')),'backbone_output_dtype':extractor.output_dtype,'data_manifest':str(source_path.resolve()),'data_manifest_sha256':_file_hash(source_path),'query_episode_count':len(records['query_episodes']),'prompt_episode_count':len(records['prompt_episodes']),'complete':len(records['query_episodes'])==len(source['query_episodes']) and len(records['prompt_episodes'])==len(source['prompt_episodes']),'elapsed_seconds':time.monotonic()-started,'shard_index':shard_index,'num_shards':num_shards,**records}
    _json(output,manifest);return output

class CachedCLSDataset(RobotVideoTrainingDataset):
    """Original deterministic sample selection, numeric labels, and raw-frame index.

    Only RGB loading is replaced. Its stand-in is zero, while the model receives
    cached CLS via the separate `frozen_vit_cls` field and guarded forward helper.
    """
    def __init__(self,data_manifest,feature_manifest,*,require_complete=True,**kwargs):
        super().__init__(data_manifest,**kwargs)
        self.feature_manifest=json.loads(Path(feature_manifest).read_text())
        if require_complete and not self.feature_manifest.get('complete'):raise ValueError('Training requires a complete CLS manifest; pilot partial caches need explicit opt-out')
        if self.feature_manifest['schema']!=SCHEMA or self.feature_manifest['data_manifest_sha256']!=_file_hash(data_manifest):raise ValueError('Feature cache/source manifest mismatch')
        self._feature_records={(role,row['repo_id'],row['episode_index']):row['cls'] for role in ('query','prompt') for row in self.feature_manifest[role+'_episodes']};self._cls_cache=OrderedDict()
    def _rgb(self,record):
        return np.broadcast_to(np.uint8(0),tuple(record['rgb']['shape']))
    def _features(self,role,repo,episode):
        artifact=self._feature_records.get((role,repo,episode))
        if artifact is None:raise KeyError(f'Missing CLS cache {role}:{repo}:{episode}')
        path=artifact['path']
        if path not in self._cls_cache:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if list(value.shape)!=artifact['shape'] or value.dtype!=np.float32 or Path(path).stat().st_size!=artifact['size_bytes']:raise ValueError('Invalid feature artifact '+path)
            self._cls_cache[path]=value
            if len(self._cls_cache)>128:self._cls_cache.popitem(last=False)
        self._cls_cache.move_to_end(path);return self._cls_cache[path]
    def __getitem__(self,index):
        sample=super().__getitem__(index);meta=sample['metadata'];current=self._features('query',meta['query_repo'],meta['query_episode'])[meta['history_indices']];prompt=self._features('prompt',meta['prompt_repo'],meta['prompt_episode'])
        if prompt.shape!=(16,768):raise ValueError('Expected16 uniform human prompt CLS frames')
        sample['frozen_vit_cls']={'current':np.array(current,copy=True),'prompt':np.array(prompt,copy=True),'fingerprint':self.feature_manifest['feature_fingerprint'],'backbone_output_dtype':self.feature_manifest['backbone_output_dtype']};return sample
    def __getstate__(self):
        state=super().__getstate__();state['_cls_cache']=OrderedDict();return state

def collate_cls_batch(samples):
    if not samples:raise ValueError('Empty cached batch')
    batch=len(samples);fingerprints={x['frozen_vit_cls']['fingerprint'] for x in samples};dtypes={x['frozen_vit_cls']['backbone_output_dtype'] for x in samples}
    if len(fingerprints)!=1 or len(dtypes)!=1:raise ValueError('Mixed feature cache identities')
    zero=torch.zeros(1,1,1,1,1,dtype=torch.uint8)
    return {'obs':{'agentview_rgb':zero.expand(batch,2,3,224,224),'qpos_gripper_states':torch.from_numpy(np.stack([x['obs']['qpos_gripper_states'] for x in samples]))},'prompt':{'agentview_rgb':zero.expand(batch,16,3,224,224)},'action':torch.from_numpy(np.stack([x['action'] for x in samples])),'action_mask':torch.from_numpy(np.stack([x['action_mask'] for x in samples])),'metadata':[x['metadata'] for x in samples],'frozen_vit_cls':{'current':torch.from_numpy(np.stack([x['frozen_vit_cls']['current'] for x in samples])),'prompt':torch.from_numpy(np.stack([x['frozen_vit_cls']['prompt'] for x in samples])),'fingerprint':next(iter(fingerprints)),'backbone_output_dtype':next(iter(dtypes))}}

class CachedCLSForward:
    """Non-module helper: original native model remains the DDP/checkpoint object."""
    def __init__(self,model,feature_manifest):
        self.model=model;self.extractor=FrozenCLSExtractor(model);self.manifest=json.loads(Path(feature_manifest).read_text())
        if self.manifest['feature_fingerprint']!=self.extractor.fingerprint:raise ValueError('Frozen visual weights/preprocessing/precision mismatch')
    @contextmanager
    def context(self,current,prompt,*,fingerprint=None):
        self.extractor.validate()
        if fingerprint is not None and fingerprint!=self.extractor.fingerprint:raise ValueError('Batch visual fingerprint mismatch')
        backbone=self.extractor.backbone
        if getattr(backbone,'_bpp_cls_cache_active',False):raise RuntimeError('Nested/concurrent CLS context on one model')
        dtype=getattr(torch,self.manifest['backbone_output_dtype']);queue=[prompt,current];previous=backbone.forward;backbone._bpp_cls_cache_active=True
        def replay(pixels,*args,**kwargs):
            if args or kwargs or not queue:raise RuntimeError('Unexpected frozen ViT call during cached forward')
            feature=torch.as_tensor(queue.pop(0),device=pixels.device)
            if feature.ndim!=3 or feature.shape[-1]!=768 or feature.shape[0]*feature.shape[1]!=pixels.shape[0] or not torch.isfinite(feature).all():raise ValueError('Cached CLS shape/finite mismatch')
            return feature.detach().reshape(-1,1,768).to(dtype=dtype)
        backbone.forward=replay
        try:
            yield
            if queue:raise RuntimeError('Cached forward did not consume prompt and current CLS exactly once')
        finally:
            backbone.forward=previous;backbone._bpp_cls_cache_active=False
    def forward(self,batch,forward=None):
        cache=batch['frozen_vit_cls'];raw={key:value for key,value in batch.items() if key!='frozen_vit_cls'}
        with self.context(cache['current'],cache['prompt'],fingerprint=cache['fingerprint']):return (self.model if forward is None else forward)(raw)


def merge_cls_manifests(data_manifest,shard_paths,output):
    """Validate all official episode keys exactly once before publishing full cache."""
    source=json.loads(Path(data_manifest).read_text());shards=[json.loads(Path(path).read_text()) for path in shard_paths]
    if not shards:raise ValueError('No feature shards')
    first=shards[0];merged={key:copy.deepcopy(value) for key,value in first.items() if key not in {'query_episodes','prompt_episodes'}}
    digest=_file_hash(data_manifest)
    for role in ('query','prompt'):
        collected={}
        source_rows={(row['repo_id'],row['episode_index']):row for row in source[role+'_episodes']}
        for shard in shards:
            if shard['schema']!=SCHEMA or shard['data_manifest_sha256']!=digest or shard['feature_fingerprint']!=first['feature_fingerprint'] or shard['backbone_output_dtype']!=first['backbone_output_dtype']:raise ValueError('Feature shard identity mismatch')
            for row in shard[role+'_episodes']:
                key=(row['repo_id'],row['episode_index'])
                if key in collected:raise ValueError('Duplicate feature episode '+str(key))
                if key not in source_rows or row['source_rgb']['sha256']!=source_rows[key]['rgb']['sha256']:raise ValueError('Feature episode RGB identity mismatch')
                artifact=row['cls'];expected_rgb=source_rows[key]['rgb']
                if artifact['source']!={'encoder':first['feature_fingerprint'],'rgb_sha256':expected_rgb['sha256'],'rgb_shape':expected_rgb['shape']}:raise ValueError('Feature artifact source fingerprint mismatch')
                if not Path(artifact['path']).is_file() or Path(artifact['path']).stat().st_size!=artifact['size_bytes']:raise ValueError('Missing feature artifact')
                value=np.load(artifact['path'],mmap_mode='r',allow_pickle=False)
                if list(value.shape)!=[expected_rgb['shape'][0],768] or value.dtype!=np.float32 or _file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('Feature artifact content mismatch')
                collected[key]=row
        expected={(row['repo_id'],row['episode_index']) for row in source[role+'_episodes']}
        if set(collected)!=expected:raise ValueError('Incomplete '+role+' feature shards')
        merged[role+'_episodes']=[collected[key] for key in sorted(collected)];merged[role+'_episode_count']=len(collected)
    merged.update(complete=True,shard_index=0,num_shards=len(shards),shard_manifests=[str(Path(path).resolve()) for path in shard_paths]);_json(output,merged);return Path(output)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--data-manifest',type=Path,nargs='+',required=True);prepare.add_argument('--cache-root',type=Path,required=True);prepare.add_argument('--checkpoint',type=Path,default=Path(os.environ.get('BPP_CHECKPOINT','')));prepare.add_argument('--bpp-repo',type=Path,default=Path(os.environ.get('BPP_ROOT','/gaozt-test1/sxh/bpp'))/'repos/behavior_prompting');prepare.add_argument('--device');prepare.add_argument('--amp-dtype',choices=('bfloat16','float16','float32'),default='bfloat16');prepare.add_argument('--batch-frames',type=int,default=32);prepare.add_argument('--shard-index',type=int,default=int(os.environ.get('RANK',0)));prepare.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE',1)));prepare.add_argument('--all-episodes',action='store_true');prepare.add_argument('--max-query-episodes',type=int,default=1);prepare.add_argument('--max-prompt-episodes',type=int,default=1);prepare.add_argument('--fixed-vision-preprocessing',action='store_true')
    merge=commands.add_parser('merge');merge.add_argument('--data-manifest',type=Path,required=True);merge.add_argument('--shards',type=Path,nargs='+',required=True);merge.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='merge':
        output=merge_cls_manifests(args.data_manifest,args.shards,args.output);print(json.dumps({'complete_feature_manifest':str(output)},indent=2));return 0
    if not args.fixed_vision_preprocessing:raise ValueError('Explicit --fixed-vision-preprocessing opt-in is required for the cache protocol')
    from .native_model import NativeBPPModel
    torch.set_num_threads(4);device=args.device or ('cuda:'+os.environ.get('LOCAL_RANK','0') if torch.cuda.is_available() else 'cpu')
    if str(device).startswith('cuda'):torch.cuda.set_device(torch.device(device))
    model=NativeBPPModel(args.checkpoint,bpp_repo=args.bpp_repo,device=device,freeze_vision=True,amp_dtype=args.amp_dtype);configure_deterministic_vision(model);extractor=FrozenCLSExtractor(model)
    for source in args.data_manifest:
        output=prepare_cls_cache(source,extractor,args.cache_root,batch_frames=args.batch_frames,max_query_episodes=None if args.all_episodes else args.max_query_episodes,max_prompt_episodes=None if args.all_episodes else args.max_prompt_episodes,shard_index=args.shard_index,num_shards=args.num_shards)
        print(json.dumps({'feature_manifest':str(output),'feature_fingerprint':extractor.fingerprint,'shard_index':args.shard_index,'num_shards':args.num_shards,'all_episodes_requested':args.all_episodes}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
