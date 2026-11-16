"""One CPU fixture for read-only pre-clip module gradient telemetry."""
import math
import unittest
import torch
from torch import nn
from examples.baselines.bpp.bpp.train import preclip_gradient_norms_by_module


class GradientTelemetryTests(unittest.TestCase):
    def test_unique_group_partition_matches_global_norm_without_mutation(self):
        model = nn.Module()
        model.policy = nn.Module()
        model.policy.model = nn.Module()
        model.policy.model.down_modules = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 1)])
        model.policy.model.shared_alias = model.policy.model.down_modules[0]
        model.policy.obs_encoder = nn.Module()
        model.policy.obs_encoder.prompt_encoder = nn.Module()
        model.policy.obs_encoder.prompt_encoder.layers = nn.ModuleList([nn.Linear(2, 2)])
        model.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        model.frozen.grad = torch.full((2,), 1000.0)
        model.unused = nn.Parameter(torch.ones(2))
        for index, parameter in enumerate(model.parameters()):
            if parameter.requires_grad and parameter is not model.unused:
                parameter.grad = torch.full_like(parameter, (index + 1) / 10)
        before = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
        rng_before = torch.get_rng_state().clone()
        actual = preclip_gradient_norms_by_module(model)
        norms = [torch.linalg.vector_norm(p.grad) for p in model.parameters()
                 if p.requires_grad and p.grad is not None]
        expected = float(torch.linalg.vector_norm(torch.stack(norms)))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in actual.values())), expected, places=6)
        self.assertIn('policy.model.down_modules.0', actual)
        self.assertIn('policy.model.down_modules.1', actual)
        self.assertIn('policy.obs_encoder.prompt_encoder.layers', actual)
        self.assertEqual(len(actual), 3)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for name, parameter in model.named_parameters():
            if before[name] is None:
                self.assertIsNone(parameter.grad)
            else:
                self.assertTrue(torch.equal(before[name], parameter.grad))


if __name__ == '__main__':
    unittest.main()
