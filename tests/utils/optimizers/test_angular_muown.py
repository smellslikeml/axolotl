"""Tests for the AngularMuown optimizer and its factory wiring."""

from types import SimpleNamespace

import pytest
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
    # Polar Express pushes every singular value into a tight band near 1
    # (an approximate orthogonalization, not an exact one) regardless of input scale.
    # The band is tighter than classic Muon's NS — empirically singular values land
    # well inside [0.8, 1.2] after 5+ steps — but we keep the [0.5, 1.5] envelope
    # so the assertion stays robust to small numerical differences across torch
    # versions / dtype paths.
    svals = torch.linalg.svdvals(ortho)
    assert svals.min() > 0.5
    assert svals.max() < 1.5


def test_newton_schulz_handles_packed_qkv_via_chunked_orthogonalize():
    """Packed QKV matrices (rows == 3 * cols) must be orthogonalized as three
    square chunks, not as one tall block — that's what _orthogonalize_update
    does. Without this, LLaMA-family fused qkv_proj matrices get the wrong
    block structure on the directional step."""
    from axolotl.utils.optimizers.angular_muown import _orthogonalize_update

    torch.manual_seed(0)
    cols = 8
    packed = torch.randn(3 * cols, cols)  # mirrors a qkv_proj shape
    ortho = _orthogonalize_update(packed, ns_steps=5)
    assert ortho.shape == packed.shape
    # Each square chunk's singular values should be near 1 (the chunk-wise
    # orthogonalization invariant) — verify on the middle chunk.
    mid_chunk = ortho[cols : 2 * cols]
    svals = torch.linalg.svdvals(mid_chunk)
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


def test_angular_multiplier_warmup_and_inverse_polynomial_decay():
    """The angular multiplier matches fhueb/angular-muown's
    ``_angular_lr_multiplier``: linear warmup to 1.0, then inverse-polynomial
    decay ``(1 + decay_scale * t)**(-decay_degree)`` — the paper's actual
    "implicit decay" mechanism made explicit."""
    weight = nn.Parameter(torch.randn(4, 4))
    opt = AngularMuown(
        [{"params": [weight], "use_angular": True}],
        lr=0.05,
        angular_warmup_steps=4,
        angular_decay_scale=0.5,
        angular_decay_degree=0.5,  # inverse-sqrt-style decay
    )

    # Warmup: linear ramp from 1/warmup_steps at step 0 up to 1.0 at step=warmup
    warmup_at_0 = opt.angular_multiplier(4, 0.5, 0.5)
    assert 0.0 < warmup_at_0 <= 1.0
    opt._angular_step = 4
    assert opt.angular_multiplier(4, 0.5, 0.5) == 1.0  # peak right after warmup

    # Post-warmup: (1 + 0.5 * t_after_warmup)**(-0.5), strictly monotone decay
    opt._angular_step = 8  # t_after_warmup = 4
    mult_at_8 = opt.angular_multiplier(4, 0.5, 0.5)
    assert mult_at_8 == pytest.approx((1 + 0.5 * 4) ** -0.5, rel=1e-6)
    opt._angular_step = 20  # t_after_warmup = 16, mult should be smaller
    mult_at_20 = opt.angular_multiplier(4, 0.5, 0.5)
    assert mult_at_20 < mult_at_8
    # Inverse-polynomial decay never reaches zero; ours doesn't either.
    opt._angular_step = 10_000
    assert opt.angular_multiplier(4, 0.5, 0.5) > 0.0

    # decay_degree=0 → constant 1.0 post-warmup (recovers Muown's pre-paper baseline)
    opt._angular_step = 100
    assert opt.angular_multiplier(4, 0.5, 0.0) == 1.0


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
