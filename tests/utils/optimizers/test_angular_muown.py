"""Tests for the AngularMuown optimizer and its factory wiring."""

from types import SimpleNamespace

import torch
from torch import nn

from axolotl.utils.optimizers.angular_muown import (
    AngularMuown,
    AngularMuownOptimizerFactory,
    zeropower_via_newtonschulz5,
)

# imported from the existing (non-new) schema module that the builder dispatch
# checks against to route an optimizer onto the custom-factory code path
from axolotl.utils.schemas.enums import CustomSupportedOptimizers


def test_angular_muown_registered_as_custom_optimizer():
    """The dispatch in core/builders/base.py routes any optimizer whose name is in
    CustomSupportedOptimizers through the custom-factory path, so registration here
    is what wires `angular_muown` to AngularMuownOptimizerFactory."""
    names = [opt.value for opt in CustomSupportedOptimizers]
    assert "angular_muown" in names


def test_newton_schulz_bounds_singular_values():
    torch.manual_seed(0)
    mat = torch.randn(8, 16) * 100.0  # arbitrary scale; NS should normalize it away
    ortho = zeropower_via_newtonschulz5(mat, steps=6)
    # Muon's Newton-Schulz pushes every singular value into a tight band near 1
    # (an approximate orthogonalization, not an exact one) regardless of input scale
    svals = torch.linalg.svdvals(ortho)
    assert svals.min() > 0.5
    assert svals.max() < 1.5


def test_angular_step_decreases_quadratic_loss():
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(6, 6))
    target = torch.randn(6, 6)
    opt = AngularMuown([{"params": [weight], "use_angular": True}], lr=0.1)

    def loss_fn():
        return ((weight - target) ** 2).sum()

    first = loss_fn().item()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert loss_fn().item() < first


def test_angular_multiplier_warmup_and_decay():
    weight = nn.Parameter(torch.randn(4, 4))
    opt = AngularMuown(
        [{"params": [weight], "use_angular": True}],
        lr=0.05,
        angular_warmup_steps=4,
        angular_decay_steps=4,
        angular_min_mult=0.0,
    )
    warmup = opt.angular_multiplier(4, 4, 0.0)
    assert 0.0 < warmup <= 1.0  # still warming up at step 0
    opt._angular_step = 4
    assert opt.angular_multiplier(4, 4, 0.0) == 1.0  # peak right after warmup
    opt._angular_step = 8
    assert opt.angular_multiplier(4, 4, 0.0) == 0.0  # fully decayed


def test_factory_splits_matrices_from_embeddings_and_norms():
    # name the modules so the factory's name-based split sees an "embed" param
    model = nn.ModuleDict(
        {
            "embed_tokens": nn.Embedding(10, 8),
            "proj": nn.Linear(8, 8, bias=True),
            "norm": nn.LayerNorm(8),
        }
    )
    training_args = SimpleNamespace(learning_rate=0.01, weight_decay=0.0)
    optimizer = AngularMuownOptimizerFactory()(model, training_args)

    assert isinstance(optimizer, AngularMuown)
    angular_groups = [g for g in optimizer.param_groups if g.get("use_angular")]
    adamw_groups = [g for g in optimizer.param_groups if not g.get("use_angular")]

    # only proj.weight (a 2D non-embedding matrix) goes to the angular group
    angular_params = [p for g in angular_groups for p in g["params"]]
    assert len(angular_params) == 1
    assert angular_params[0].shape == (8, 8)

    # embedding weight, bias, and layernorm params fall back to AdamW
    adamw_param_count = sum(len(g["params"]) for g in adamw_groups)
    assert adamw_param_count >= 3
