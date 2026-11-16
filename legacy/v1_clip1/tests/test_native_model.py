"""Native BPP contract tests using a tiny trainable policy, no large weights."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from examples.baselines.bpp.bpp.native_model import GROUPS, NativeBPPModel


class TinyVision(nn.Module):
    def __init__(self, legacy=False):
        super().__init__()
        self.key_model_map = nn.ModuleDict({'agentview_rgb': nn.Linear(3, 768)})
        fields = [('gripper_states',2),('ee_pos',3)] if legacy else [(k,b-a) for k,a,b in GROUPS]
        self.key_projection_map = nn.ModuleDict({k: nn.Linear(d,768) for k,d in fields})

    def rgb(self, image):
        return self.key_model_map['agentview_rgb'](image.mean(dim=(-1,-2)))


class TinyTokenizer(nn.Module):
    def __init__(self, legacy=False):
        super().__init__()
        self.prompt_obs_encoder = TinyVision(legacy)


class TinyEncoder(nn.Module):
    def __init__(self, legacy=False):
        super().__init__()
        self.obs_encoder = TinyTokenizer(legacy)
        self.pool = nn.Linear(768,768)
        self.is_prompted = False
        self.prompt_tokens_cache = None
        self.prompt_mask_cache = None

    def forward(self, obs):
        visual = self.obs_encoder.prompt_obs_encoder
        if 'prompt' in obs:
            prompt = obs['prompt']
            if set(prompt) != {'obs','metadata'} or set(prompt['obs']) != {'agentview_rgb'}:
                raise AssertionError('Non-RGB demonstration reached the encoder')
            tokens = self.pool(visual.rgb(prompt['obs']['agentview_rgb']))
            mask = prompt['metadata']['mask']
            if len(obs) == 1:
                return tokens, mask
        else:
            assert self.is_prompted
            tokens = self.prompt_tokens_cache
        condition = visual.rgb(obs['agentview_rgb']).mean(dim=1)
        for key, value in obs.items():
            if key not in ('agentview_rgb','prompt'):
                condition = condition + visual.key_projection_map[key](value).mean(dim=1)
        return condition + tokens.mean(dim=1), {}

    def reset(self):
        self.is_prompted = False
        self.prompt_tokens_cache = None
        self.prompt_mask_cache = None


class TinyHead(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Conv1d(action_dim,action_dim,1)
        self.condition = nn.Linear(768,action_dim)

    def forward(self, action, timestep, global_cond):
        return self.conv(action.transpose(1,2)).transpose(1,2) + self.condition(global_cond)[:,None]


class TinyPolicy(nn.Module):
    def __init__(self, action_dim=16, legacy=False):
        super().__init__()
        self.obs_encoder = TinyEncoder(legacy)
        self.model = TinyHead(action_dim)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=50,prediction_type='epsilon')
        self.obs_feature_dim = 15360

    def _apply_backward_compatibility_renames(self, state):
        pass

    def conditional_sample(self, data, mask, global_cond):
        return self.model(data,torch.zeros(data.shape[0],dtype=torch.long),global_cond)


def source_config():
    vision={'pretrained':True,'frozen':False,'shape_meta':{}}
    tokenizer={'obs_encoder':vision,'shape_meta':{}}
    return {'shape_meta':{},'obs_encoder':{'obs_encoder':tokenizer,'shape_meta':{}}}


def stats(low=-1., high=1., source='unit-test train rows only'):
    return {'schema':'bpp_tig_native_normalizer_v1','mode':'minmax','source':{'description':source},
            'selected_bounds':{'state':{'min':[low]*18,'max':[high]*18},
                               'action':{'min':[low]*16,'max':[high]*16}}}


class NativeModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory(dir='/gaozt-test1/sxh/bpp/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.source=Path(self.tmp.name)/'source.ckpt'
        torch.save({'cfg':OmegaConf.create({'model':source_config()}),
                    'state_dicts':{'model':TinyPolicy(action_dim=10,legacy=True).state_dict()}},self.source)
        self.patcher=patch('examples.baselines.bpp.bpp.native_model._instantiate',side_effect=lambda *a:TinyPolicy())
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def model(self, normalization=None):
        return NativeBPPModel(self.source,device='cpu',amp_dtype='float32',
                              normalizer_stats=stats() if normalization is None else normalization)

    def batch(self):
        return {'obs':{'agentview_rgb':torch.zeros(2,2,3,224,224,dtype=torch.uint8),
                       'qpos_gripper_states':torch.zeros(2,2,18)},
                'prompt':{'agentview_rgb':torch.zeros(1,16,3,224,224,dtype=torch.uint8)},
                'action':torch.zeros(2,16,16),'action_mask':torch.ones(2,16,dtype=torch.bool)}

    def test_backward_freezes_only_vision_and_reads_rgb_only_prompt(self):
        model=self.model();loss=model(self.batch());loss.backward()
        self.assertEqual(loss.ndim,0)
        self.assertTrue(torch.isfinite(loss))
        for name,parameter in model.named_parameters():
            if '.key_model_map.' in name:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            elif parameter.requires_grad:
                self.assertIsNotNone(parameter.grad,name)
                self.assertTrue(torch.isfinite(parameter.grad).all(),name)
        self.assertGreater(model.parameter_audit['pretrained_parameters'],0)
        self.assertTrue(model.parameter_audit['explicit_finger_mappings'])
        forbidden=self.batch();forbidden['prompt']['action']=torch.zeros(2,16,16)
        with self.assertRaises(ValueError):model(forbidden)

    def test_masked_padding_values_do_not_change_loss(self):
        model=self.model();batch=self.batch();batch['action_mask'][:,8:]=False
        batch['action'][:,8:]=float('nan')
        torch.manual_seed(9);left=model(batch)
        batch['action'][:,8:]=1e20
        torch.manual_seed(9);right=model(batch)
        self.assertTrue(torch.equal(left,right))

    def test_seed_head_hash_normalizer_override_and_checkpoint_restore(self):
        first=self.model();second=self.model(stats(-2,2))
        self.assertEqual(first.init_head_sha256,second.init_head_sha256)
        first.set_normalizer_stats(stats(-2,2,'new train-only FT rows'))
        self.assertTrue(torch.equal(first.state_scale,torch.full((18,),.5)))
        path=Path(self.tmp.name)/'native.ckpt'
        first.save_checkpoint(path,metadata={'pilot':True},additional_state={'train_step':0})
        restored=NativeBPPModel.from_training_checkpoint(path,device='cpu',amp_dtype='float32')
        self.assertEqual(restored.init_head_sha256,first.init_head_sha256)
        self.assertEqual(restored.normalizer_stats,first.normalizer_stats)
        for key,value in first.state_dict().items():
            if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,restored.state_dict()[key]),key)
        overridden=NativeBPPModel.from_training_checkpoint(path,device='cpu',normalizer_stats=stats(-4,4),amp_dtype='float32')
        self.assertTrue(torch.equal(overridden.action_scale,torch.full((16,),.25)))

    def test_cached_prompt_prediction_and_padding_validation(self):
        model=self.model();batch=self.batch()
        with self.assertRaises(RuntimeError):model.encode_prompt(batch['prompt']['agentview_rgb'])
        model.eval();features=model.encode_prompt(batch['prompt']['agentview_rgb'])
        prediction=model.predict_action(batch['obs'],prompt_features=features)
        self.assertEqual(tuple(prediction.shape),(2,16,16))
        self.assertFalse(prediction.requires_grad)
        self.assertFalse(model.policy.obs_encoder.is_prompted)
        self.assertIsNone(model.policy.obs_encoder.prompt_tokens_cache)
        with self.assertRaises(ValueError):
            model.encode_prompt(batch['prompt']['agentview_rgb'],torch.ones(1,16,dtype=torch.bool))
        invalid=self.batch();invalid['action_mask'].zero_()
        with self.assertRaises(ValueError):model(invalid)


if __name__=='__main__':unittest.main()
