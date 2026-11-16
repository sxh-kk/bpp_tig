"""Critical cache coverage, source identity, and temporary forward restoration."""
import copy,hashlib,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from examples.baselines.bpp.bpp.feature_cache import SCHEMA,CachedCLSForward,merge_cls_manifests

def test_merge_requires_disjoint_complete_matching_rgb_sources(tmp_path):
    source=tmp_path/'data.json';roles={};artifacts={}
    for role in ('query','prompt'):
        roles[role+'_episodes']=[];artifacts[role]=[]
        for i in range(2):
            rgb={'sha256':f'{role}-rgb-{i}','shape':[2,224,224,3]};roles[role+'_episodes'].append({'repo_id':role,'episode_index':i,'rgb':rgb})
            file=tmp_path/f'{role}-{i}.npy';np.save(file,np.full((2,768),i,dtype=np.float32));spec={'encoder':'visual-fingerprint','rgb_sha256':rgb['sha256'],'rgb_shape':rgb['shape']}
            artifacts[role].append({'repo_id':role,'episode_index':i,'source_rgb':rgb,'cls':{'path':str(file),'size_bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'source':spec}})
    source.write_text(json.dumps(roles));digest=hashlib.sha256(source.read_bytes()).hexdigest();paths=[]
    for i in range(2):
        payload={'schema':SCHEMA,'data_manifest_sha256':digest,'feature_fingerprint':'visual-fingerprint','backbone_output_dtype':'float32','query_episodes':[artifacts['query'][i]],'prompt_episodes':[artifacts['prompt'][i]]};path=tmp_path/f'shard{i}.json';path.write_text(json.dumps(payload));paths.append(path)
    result=merge_cls_manifests(source,paths,tmp_path/'merged.json');merged=json.loads(result.read_text());assert merged['complete'] and merged['query_episode_count']==2 and merged['prompt_episode_count']==2
    with pytest.raises(ValueError,match='Incomplete'):merge_cls_manifests(source,paths[:1],tmp_path/'incomplete.json')
    with pytest.raises(ValueError,match='Duplicate'):merge_cls_manifests(source,[paths[0],paths[0],paths[1]],tmp_path/'duplicate.json')
    corrupted=json.loads(paths[0].read_text());corrupted['query_episodes'][0]['source_rgb']['sha256']='other-episode';bad=tmp_path/'bad.json';bad.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError,match='RGB identity'):merge_cls_manifests(source,[bad,paths[1]],tmp_path/'bad-merged.json')
    feature=tmp_path/'query-0.npy';values=np.load(feature);values[0,0]+=1;np.save(feature,values)
    with pytest.raises(ValueError,match='content mismatch'):merge_cls_manifests(source,paths,tmp_path/'changed-content.json')

def test_replay_restores_original_forward_after_loss_exception():
    backbone=torch.nn.Identity();before=backbone.forward;helper=object.__new__(CachedCLSForward);helper.extractor=SimpleNamespace(validate=lambda:None,backbone=backbone,fingerprint='same');helper.manifest={'backbone_output_dtype':'float32'};features=torch.zeros(1,2,768)
    with pytest.raises(RuntimeError,match='loss failed'):
        with helper.context(features,features,fingerprint='same'):
            assert backbone(torch.zeros(2,3,224,224)).shape==(2,1,768)
            raise RuntimeError('loss failed')
    assert backbone.forward==before and not backbone._bpp_cls_cache_active
    assert torch.equal(backbone(torch.ones(2)),torch.ones(2))
