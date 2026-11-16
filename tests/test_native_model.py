"""Native BPP contract tests using a tiny trainable policy, no large weights."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from examples.baselines.bpp.bpp.native_model import (
    GROUPS, NativeBPPModel, ACTION_NORMALIZATION_V2, LEGACY_ACTION_NORMALIZATION,
    ACTION_JOINT_INDICES, resolve_action_normalization, create_ema, restore_ema,
    pack_ema_state, training_recipe_fields, EMA_CONFIG_V2,
)


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
        self.receding_obs_encoder = self.prompt_obs_encoder


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
                               'action':{'min':[low]*16,'max':[high]*16}},
            'state':{'min':[low]*18,'max':[high]*18},
            'action':{'min':[low]*16,'max':[high]*16}}


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

    def model(self, normalization=None, action_normalization=None):
        return NativeBPPModel(self.source,device='cpu',amp_dtype='float32',
                              normalizer_stats=stats() if normalization is None else normalization,
                              action_normalization=action_normalization)

    def v2_stats(self):
        value=stats()
        value['mode']='quantile'
        value['selected_bounds']['state']={'min':[-2.]*18,'max':[2.]*18}
        value['state']={'min':[-9.]*18,'max':[9.]*18}
        low=[-3.]*16;high=[5.]*16
        low[2]=high[2]=.25  # Constant joint, independent of a constant gripper.
        low[7]=high[7]=1.
        low[15]=-1.;high[15]=1.
        value['action']={'min':low,'max':high}
        return value

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


    def test_v2_action_minmax_roundtrip_and_identity_grippers_keep_state_quantiles(self):
        raw=self.v2_stats();model=self.model(raw,ACTION_NORMALIZATION_V2)
        legacy=self.model(raw)
        self.assertTrue(torch.equal(model.state_scale,legacy.state_scale))
        self.assertTrue(torch.equal(model.state_offset,legacy.state_offset))
        self.assertTrue(torch.equal(model.state_scale,torch.full((18,),.5)))
        self.assertEqual(model.action_normalization,ACTION_NORMALIZATION_V2)
        low=torch.tensor(raw['action']['min']);high=torch.tensor(raw['action']['max'])
        values=torch.stack((low,(low+high)/2,high))
        normalized=values*model.action_scale+model.action_offset
        self.assertLessEqual(float(normalized.abs().max()),1.000001)
        joints=[i for i in ACTION_JOINT_INDICES if i!=2]
        self.assertTrue(torch.allclose(normalized[0,joints],-torch.ones(len(joints))))
        self.assertTrue(torch.allclose(normalized[-1,joints],torch.ones(len(joints))))
        self.assertTrue(torch.equal(model.action_scale[[7,15]],torch.ones(2)))
        self.assertTrue(torch.equal(model.action_offset[[7,15]],torch.zeros(2)))
        self.assertTrue(torch.equal(normalized[:,7],torch.ones(3)))
        self.assertTrue(torch.equal(normalized[:,2],torch.zeros(3)))
        restored=(normalized-model.action_offset)/model.action_scale
        self.assertTrue(torch.allclose(restored,values,atol=1e-6))
        self.assertEqual(model.config['action_normalization'],ACTION_NORMALIZATION_V2)

    def test_v2_requires_full_training_bounds_and_legal_gripper_units(self):
        malformed=self.v2_stats();del malformed['action']['min']
        with self.assertRaisesRegex(ValueError,'full train action min/max'):
            self.model(malformed,ACTION_NORMALIZATION_V2)
        malformed=self.v2_stats();malformed['action']['max'][15]=.04
        malformed['action']['min'][15]=-.01
        # Meter-like gripper bounds are legal values within[-1,1], but are never
        # used as a new scale: the native command remains identity.
        model=self.model(malformed,ACTION_NORMALIZATION_V2)
        self.assertEqual(float(model.action_scale[15]),1.)
        self.assertEqual(float(model.action_offset[15]),0.)
        malformed['action']['max'][15]=2.
        with self.assertRaisesRegex(ValueError,'already be in'):
            self.model(malformed,ACTION_NORMALIZATION_V2)

    def test_v2_checkpoint_mode_roundtrip_rejects_legacy_relabel_and_bad_buffers(self):
        model=self.model(self.v2_stats(),ACTION_NORMALIZATION_V2)
        path=Path(self.tmp.name)/'v2.ckpt';model.save_checkpoint(path,metadata={'pilot':True},ema=create_ema(model))
        payload=torch.load(path,weights_only=False)
        self.assertEqual(payload['action_normalization'],ACTION_NORMALIZATION_V2)
        self.assertEqual(payload['config']['action_normalization'],ACTION_NORMALIZATION_V2)
        self.assertEqual(payload['metadata']['action_normalization'],ACTION_NORMALIZATION_V2)
        restored=NativeBPPModel.from_training_checkpoint(path,device='cpu',amp_dtype='float32')
        self.assertTrue(torch.equal(restored.action_scale,model.action_scale))
        self.assertTrue(torch.equal(restored.action_offset,model.action_offset))
        self.assertEqual(restored.action_normalization,ACTION_NORMALIZATION_V2)
        payload['state_dict']['action_scale']=payload['state_dict']['action_scale']*2
        corrupt=Path(self.tmp.name)/'mismatched_buffers.ckpt';torch.save(payload,corrupt)
        with self.assertRaisesRegex(ValueError,'does not match its declared v2'):
            NativeBPPModel.from_training_checkpoint(corrupt,device='cpu',amp_dtype='float32')
        legacy=self.model(self.v2_stats());legacy_path=Path(self.tmp.name)/'legacy.ckpt'
        old=legacy.export_state()
        for place in (old,old['config'],old['metadata']):
            for key in ('action_normalization',*training_recipe_fields(LEGACY_ACTION_NORMALIZATION)):
                place.pop(key,None)
        old.pop('ema',None);old.pop('state_dict_weight_type',None)
        torch.save(old,legacy_path)
        historical=NativeBPPModel.from_training_checkpoint(legacy_path,device='cpu',amp_dtype='float32')
        self.assertEqual(historical.action_normalization,LEGACY_ACTION_NORMALIZATION)
        self.assertTrue(torch.equal(historical.action_scale,legacy.action_scale))
        with self.assertRaisesRegex(ValueError,'Cannot relabel'):
            NativeBPPModel.from_training_checkpoint(legacy_path,device='cpu',amp_dtype='float32',
                normalizer_stats=self.v2_stats(),action_normalization=ACTION_NORMALIZATION_V2)

    def test_normalization_tags_cannot_disagree_or_be_forged_during_export(self):
        self.assertEqual(resolve_action_normalization({'schema':'bpp_tig_native_v1'}),LEGACY_ACTION_NORMALIZATION)
        with self.assertRaisesRegex(ValueError,'disagree'):
            resolve_action_normalization({'schema':'bpp_tig_native_v1','action_normalization':ACTION_NORMALIZATION_V2,
                                          'config':{'action_normalization':LEGACY_ACTION_NORMALIZATION}})
        model=self.model(self.v2_stats(),ACTION_NORMALIZATION_V2)
        with self.assertRaisesRegex(ValueError,'cannot relabel'):
            model.export_state(metadata={'action_normalization':LEGACY_ACTION_NORMALIZATION})

    def test_v2_preserves_common_head_initialization_and_ddim_recipe(self):
        legacy=self.model(self.v2_stats());v2=self.model(self.v2_stats(),ACTION_NORMALIZATION_V2)
        self.assertEqual(legacy.init_head_sha256,v2.init_head_sha256)
        self.assertEqual(legacy.parameter_audit['new_parameters'],v2.parameter_audit['new_parameters'])
        config=dict(v2.policy.noise_scheduler.config)
        self.assertEqual(config,dict(legacy.policy.noise_scheduler.config))
        self.assertTrue(config['clip_sample'])
        self.assertEqual(config['num_train_timesteps'],50)
        self.assertEqual(config['prediction_type'],'epsilon')

    def test_training_cli_requires_explicit_v2_mode(self):
        from examples.baselines.bpp.bpp.train import make_parser
        arguments=['--regime','c15','--data-manifest','unused.json','--output-dir','unused']
        with patch('sys.stderr',new_callable=io.StringIO):
            with self.assertRaises(SystemExit):make_parser().parse_args(arguments)
            with self.assertRaises(SystemExit):make_parser().parse_args(arguments+['--action-normalization',LEGACY_ACTION_NORMALIZATION])
        args=make_parser().parse_args(arguments+['--action-normalization',ACTION_NORMALIZATION_V2])
        self.assertEqual(args.action_normalization,ACTION_NORMALIZATION_V2)


    def test_v2_source_input_noise_perturbation_keeps_original_noise_target(self):
        model=self.model(self.v2_stats(),ACTION_NORMALIZATION_V2);batch=self.batch()
        batch['action_mask'][:,8:]=False
        base=torch.full((2,16,16),2.);extra=torch.full_like(base,3.)
        received=[]
        def add_noise(values,noise,timesteps):
            received.append(noise.clone())
            return values+noise
        with patch('torch.randn_like',side_effect=[base,extra]), \
                patch.object(model.train_noise_scheduler,'add_noise',side_effect=add_noise), \
                patch.object(model.policy.model,'forward',return_value=torch.zeros_like(base)):
            loss=model(batch)
        self.assertEqual(len(received),1)
        self.assertTrue(torch.equal(received[0][:,:8],torch.full((2,8,16),2.3)))
        self.assertTrue(torch.equal(received[0][:,8:],torch.zeros((2,8,16))))
        self.assertAlmostEqual(float(loss),4.,places=6)  # target2, not perturbed2.3

    def test_v2_ema_raw_checkpoint_exact_resume_and_alias_safe_eval_selection(self):
        model=self.model(self.v2_stats(),ACTION_NORMALIZATION_V2);ema=create_ema(model)
        self.assertFalse(ema.use_ema_warmup)
        self.assertEqual(ema.power,.75)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        def update(current,opt,average,gradient):
            opt.zero_grad(set_to_none=True)
            for p in current.parameters():
                if p.requires_grad:p.grad=torch.full_like(p,gradient)
            opt.step();average.step(current.parameters())
        update(model,optimizer,ema,.2);update(model,optimizer,ema,.4)
        self.assertEqual(ema.optimization_step,2)
        raw=[p.detach().clone() for p in model.parameters()]
        self.assertTrue(any(not torch.equal(p,e) for p,e in zip(raw,ema.shadow_params)))
        path=Path(self.tmp.name)/'raw_and_ema.ckpt'
        model.save_checkpoint(path,metadata={'pilot':True,'train_step':2},
                              additional_state={'optimizer':optimizer.state_dict(),'train_step':2},ema=ema)
        payload=torch.load(path,weights_only=False)
        self.assertEqual(payload['state_dict_weight_type'],'raw')
        self.assertEqual(payload['ema']['state_dict']['optimization_step'],2)
        self.assertEqual(payload['ema']['parameter_names'],[n for n,p in model.named_parameters()])
        self.assertNotIn('action_scale',payload['ema']['parameter_names'])
        resumed=NativeBPPModel.from_training_checkpoint(path,device='cpu',amp_dtype='float32',load_ema=False)
        evaluation=NativeBPPModel.from_training_checkpoint(path,device='cpu',amp_dtype='float32')
        self.assertEqual(evaluation.loaded_weight_selection,'ema')
        self.assertEqual(evaluation.loaded_ema_optimization_step,2)
        self.assertIsNone(resumed.loaded_ema_optimization_step)
        self.assertEqual(resumed.loaded_weight_selection,'raw')
        for actual,expected in zip(resumed.parameters(),raw):self.assertTrue(torch.equal(actual,expected))
        for actual,expected in zip(evaluation.parameters(),ema.shadow_params):self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(evaluation.action_scale,resumed.action_scale))
        self.assertIs(evaluation.policy.obs_encoder.obs_encoder.prompt_obs_encoder,
                      evaluation.policy.obs_encoder.obs_encoder.receding_obs_encoder)
        with self.assertRaisesRegex(ValueError,'cannot be exported as a raw'):
            evaluation.export_state(ema=ema)
        opt2=torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=.01)
        opt2.load_state_dict(payload['additional_state']['optimizer'])
        average2=create_ema(resumed);restore_ema(resumed,average2,payload['ema'],expected_step=2)
        update(model,optimizer,ema,.7);update(resumed,opt2,average2,.7)
        self.assertEqual(average2.optimization_step,3)
        for actual,expected in zip(resumed.parameters(),model.parameters()):self.assertTrue(torch.equal(actual,expected))
        for actual,expected in zip(average2.shadow_params,ema.shadow_params):self.assertTrue(torch.equal(actual,expected))

    def test_v2_ema_integrity_names_steps_and_recipe_tags_are_required(self):
        model=self.model(self.v2_stats(),ACTION_NORMALIZATION_V2);average=create_ema(model)
        packed=pack_ema_state(model,average,expected_step=0)
        broken=copy.deepcopy(packed);broken['parameter_names']=list(reversed(broken['parameter_names']))
        with self.assertRaisesRegex(ValueError,'names/shapes/order'):
            restore_ema(model,average,broken)
        with self.assertRaisesRegex(ValueError,'optimization_step'):
            restore_ema(model,average,packed,expected_step=1)
        with self.assertRaisesRegex(ValueError,'requires its independent EMA'):
            model.export_state()
        exported=model.export_state(ema=average)
        for key,value in training_recipe_fields(ACTION_NORMALIZATION_V2).items():
            self.assertEqual(exported[key],value)
            self.assertEqual(exported['config'][key],value)
            self.assertEqual(exported['metadata'][key],value)
        exported['metadata']['input_perturbation']=0.
        path=Path(self.tmp.name)/'wrong_recipe.ckpt';torch.save(exported,path)
        with self.assertRaisesRegex(ValueError,'recipe tags'):
            NativeBPPModel.from_training_checkpoint(path,device='cpu',amp_dtype='float32')


if __name__=='__main__':unittest.main()
