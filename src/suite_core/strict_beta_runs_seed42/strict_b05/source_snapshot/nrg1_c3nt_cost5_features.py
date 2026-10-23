"""CNN3 feature extractor from CNN_구조_교체_연곤_전달용.docx.

6x50x50 -> 32x25x25 -> 64x13x13 -> 64x7x7 -> 3136 -> 256.
No BatchNorm/Dropout/attention/GAP; SB3 normalizes the uint8 map once.
Combined feature=256 map + 4 heading + 1 budget. No episode clock.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

OBS_KEYS = {'map', 'heading', 'detour_budget'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class CNN3MapEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, normalized_image=False):
        require(isinstance(observation_space, spaces.Box), 'Expected map Box.')
        require(observation_space.shape == (6, 50, 50), 'CNN3_NOCLOCK_TRUNC ablation requires 6x50x50 map.')
        require(is_image_space(observation_space, check_channels=False,
                               normalized_image=normalized_image), 'Expected uint8 map [0,255].')
        require(features_dim == 256, 'Keep original map feature dimension 256.')
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened = self.cnn(torch.zeros(1, 6, 50, 50)).shape[1]
        require(flattened == 3136, f'Unexpected map flatten size: {flattened}')
        self.linear = nn.Sequential(nn.Linear(flattened, features_dim), nn.ReLU())

    def forward(self, observations):
        # Inputs were already converted to float and divided by 255 by SB3.
        return self.linear(self.cnn(observations))


class CNN3CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_output_dim=256, normalized_image=False):
        require(isinstance(observation_space, spaces.Dict), 'Expected NRG1 Dict space.')
        require(set(observation_space.spaces) == OBS_KEYS, 'NRG1 observation keys changed.')
        require(cnn_output_dim == 256, 'cnn_output_dim must remain 256.')
        require(isinstance(observation_space['heading'], spaces.Discrete)
                and observation_space['heading'].n == 4, 'Expected Discrete(4) heading.')
        for key in ('detour_budget',):
            sub = observation_space[key]
            require(isinstance(sub, spaces.Box) and sub.shape == (1,)
                    and sub.dtype == np.float32 and np.all(sub.low == 0)
                    and np.all(sub.high == 1), f'Unexpected scalar space: {key}')
        super().__init__(observation_space, features_dim=1)
        modules = {}
        size = 0
        # Preserve the original Gym Dict iteration/concatenation order.
        for key, subspace in observation_space.spaces.items():
            if key == 'map':
                modules[key] = CNN3MapEncoder(subspace, cnn_output_dim, normalized_image)
                size += cnn_output_dim
            else:
                modules[key] = nn.Flatten()
                size += get_flattened_obs_dim(subspace)
        self.extractors = nn.ModuleDict(modules)
        self._features_dim = size
        require(size == 261, f'Unexpected merged dimension: {size}')

    def forward(self, observations):
        return torch.cat([extractor(observations[key])
                          for key, extractor in self.extractors.items()], dim=1)


def spatial_connectivity_audit(cnn):
    """Positive-weight structural probe, not a claim about learned sensitivity.

    Deep-copying consumes no RNG. The real model/weights/gradients are unchanged.
    Every input cell must have a path to at least one convolution output.
    """
    probe = deepcopy(cnn).cpu().double()
    for layer in probe.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.constant_(layer.weight, 0.01)
            nn.init.constant_(layer.bias, 0.1)
    x = torch.ones(1, 6, 50, 50, dtype=torch.float64, requires_grad=True)
    y = probe(x)
    gradient, = torch.autograd.grad(y.sum(), x)
    connected = (gradient.abs().sum(dim=(0, 1)) > 0)
    require(bool(connected.all()), 'CNN3_NOCLOCK_TRUNC still drops input cells.')
    return dict(connected_spatial_cells=int(connected.sum()), total_spatial_cells=2500,
                structural_blind_cells=int((~connected).sum()),
                note='Positive-weight copied network; connectivity, not learned usefulness.')


def audit_cnn3_model(model):
    policy = model.policy
    extractor = policy.features_extractor
    require(isinstance(extractor, CNN3CombinedExtractor), 'Wrong extractor: not NRG1 CNN3_NOCLOCK_TRUNC.')
    require(set(model.observation_space.spaces) == OBS_KEYS, 'Wrong observation keys.')
    require(model.gamma == 1.0 and model.gae_lambda == 0.95, 'NRG1 gamma/lambda changed.')
    require(bool(policy.normalize_images), 'Image /255 preprocessing was disabled.')
    require(policy.share_features_extractor, 'Feature sharing changed.')
    require(policy.pi_features_extractor is policy.vf_features_extractor,
            'Actor/critic are not sharing the original feature branch.')
    branch = extractor.extractors['map']
    convs = [m for m in branch.cnn if isinstance(m, nn.Conv2d)]
    actual = [(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding)
              for m in convs]
    expected = [(6, 32, (3,3), (2,2), (1,1)),
                (32, 64, (3,3), (2,2), (1,1)),
                (64, 64, (3,3), (2,2), (1,1))]
    require(actual == expected, f'Unexpected CNN: {actual}')
    require(branch.linear[0].in_features == 3136 and branch.linear[0].out_features == 256,
            'Wrong map projection shape.')
    require(policy.mlp_extractor.policy_net[0].in_features == 261,
            'Actor input dimension changed.')
    require(policy.mlp_extractor.value_net[0].in_features == 261,
            'Critic input dimension changed.')
    for network in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
        linear = [m for m in network if isinstance(m, nn.Linear)]
        require([(m.in_features, m.out_features) for m in linear] == [(261,256),(256,256)],
                'Original actor/critic MLP was changed.')
    require(policy.action_net.out_features == 4 and policy.value_net.out_features == 1,
            'Unexpected output heads.')
    require(all(bool(torch.isfinite(p).all()) for p in policy.parameters()),
            'Non-finite policy weights.')
    result = dict(encoder='NRG1_CNN3_NOCLOCK_TRUNC', observation_keys=list(model.observation_space.spaces),
                  observation_map_shape=[6,50,50], map_preprocessing='float32 / 255 by SB3',
                  spatial_sizes=[50,25,13,7], map_flatten=3136, map_features=256,
                  merged_features=261, gamma=float(model.gamma), gae_lambda=float(model.gae_lambda),
                  trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                  map_encoder_parameters=sum(p.numel() for p in branch.parameters()),
                  input_cells=spatial_connectivity_audit(branch.cnn))
    return result


def write_cnn3_model_audit(model, path):
    data = audit_cnn3_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)
    print('[CNN3_NOCLOCK_TRUNC MODEL] 50->25->13->7 | map FC 3136->256 | merged=261 | '
          'connected spatial cells=2500/2500', flush=True)
    return data


def multi_input_sensitivity_audit(branch):
    """Initial weights + multiple random inputs. Zero local sensitivity is logged,
    not equated with no structural connectivity. Does not alter model/RNG state.
    """
    probe = deepcopy(branch).cpu().eval()
    gen = torch.Generator().manual_seed(12045)
    positions = [(0, 0), (0, 49), (49, 0), (49, 49), (25, 25)]
    rows = []
    with torch.no_grad():
        for sample in range(4):
            x = torch.rand((1, 6, 50, 50), generator=gen)
            y = probe(x)
            for r, c in positions:
                z = x.clone(); z[:, :, r, c] = 1 - z[:, :, r, c]
                d = (probe(z) - y).abs()
                rows.append(dict(sample=sample, row=r, col=c,
                                 feature_max_abs_change=float(d.max()),
                                 feature_mean_abs_change=float(d.mean())))
    return rows
