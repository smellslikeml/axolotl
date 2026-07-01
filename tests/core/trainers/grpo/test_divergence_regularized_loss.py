"""Unit tests for the DRPO divergence-regularized GRPO loss and its wiring.

The math properties pinned here mirror the reference implementation at
``Tencent-Hunyuan/UniRL/unirl/algorithms/drpo.py`` (Apache-2.0). The DRPO
per-token loss is::

    L_t = −A_t · r_t  +  |A_t| · μ · (r_t − 1)² / (2 · ε)

with ``μ = exp(old_logp).detach()``, ``ε = 12.5`` (paper §4), and the
``μ`` factor realizing the token-adaptive trust region ``ε_t = ε / μ``
(paper §3 — the headline contribution beyond plain SPO).
"""

import math

import pytest
import torch

from axolotl.core.trainers.grpo.divergence_regularized_loss import drpo_per_token_loss


def _logps(prob):
    """Helper: build a log-prob tensor from explicit probabilities."""
    return torch.log(torch.tensor(prob))


class TestDrpoPerTokenLoss:
    """Properties of the SPO advantage-weighted quadratic regularizer with
    Binary-TV token-adaptive ε_t = ε / μ (paper §3, gradient = Eq 9)."""

    def test_on_policy_matches_plain_policy_gradient(self):
        """When ``r_t = 1`` (logp == logp_old), ``(r_t − 1)² = 0`` so the
        quadratic penalty vanishes and the loss collapses to the standard
        importance-weighted PG term ``−A_t · r_t``."""
        old = _logps([[0.5, 0.5]])
        coef_1 = torch.ones(1, 2)
        advantages = torch.tensor([[1.0, 1.0]])
        loss = drpo_per_token_loss(coef_1, advantages, old, epsilon=12.5)
        torch.testing.assert_close(loss, -advantages)

    def test_penalty_is_advantage_sign_invariant(self):
        """The reference uses ``|A_t|`` on the quadratic penalty: same |A|,
        opposite signs ⇒ same penalty magnitude added on top of the
        signed PG term. This is what distinguishes a regularized PG from a
        PPO-clipping-style weighted PG."""
        old = _logps([[0.5]])
        # Same ratio drift, opposite-sign advantages of identical magnitude
        coef_1 = torch.tensor([[1.3]])
        eps = 12.5
        loss_pos = drpo_per_token_loss(coef_1, torch.tensor([[1.0]]), old, epsilon=eps)
        loss_neg = drpo_per_token_loss(coef_1, torch.tensor([[-1.0]]), old, epsilon=eps)
        # PG term flips sign; penalty (|A|·μ·(r-1)²/(2ε)) is identical
        pg_pos = -1.0 * 1.3
        pg_neg = -(-1.0) * 1.3
        penalty = 1.0 * 0.5 * (0.3 ** 2) / (2.0 * eps)  # |A|·μ·(r-1)²/(2ε)
        torch.testing.assert_close(loss_pos, torch.tensor([[pg_pos + penalty]]))
        torch.testing.assert_close(loss_neg, torch.tensor([[pg_neg + penalty]]))

    def test_penalty_grows_quadratically_with_divergence(self):
        """The SPO penalty is a true quadratic in ``r_t - 1`` (no clipping,
        no floor). Doubling the ratio delta should ~4× the penalty term.
        Computes everything in float64 to avoid float32-subtraction noise
        masking the math; production runs are float32+autograd-friendly."""
        old = _logps([[0.5]]).double()
        advantages = torch.tensor([[1.0]], dtype=torch.float64)
        eps = 12.5
        # Strip the PG term to isolate the penalty: loss + A·r = penalty
        for r_delta in [0.05, 0.1, 0.2]:
            coef_1 = torch.tensor([[1.0 + r_delta]], dtype=torch.float64)
            loss = drpo_per_token_loss(coef_1, advantages, old, epsilon=eps)
            penalty = loss.item() + advantages.item() * coef_1.item()
            expected = 1.0 * 0.5 * (r_delta ** 2) / (2.0 * eps)
            assert math.isclose(penalty, expected, rel_tol=1e-6)

    def test_mu_weighted_realizes_token_adaptive_trust_region(self):
        """Paper §3 — the headline contribution. With ``mu_weighted=True``
        (default), a rare token (small μ) gets a *smaller* penalty per unit
        of ratio delta than a common token (large μ): rare tokens have a
        wider effective trust region ε_t = ε / μ. Uses float64 to keep the
        PG-term cancellation noise out of the penalty-ratio check."""
        # Same ratio delta, same advantage; only μ differs between rows.
        rare_token_old = _logps([[0.05]]).double()   # μ = 0.05 (rare)
        common_token_old = _logps([[0.95]]).double() # μ = 0.95 (common)
        coef_1 = torch.tensor([[1.2]], dtype=torch.float64)
        advantages = torch.tensor([[1.0]], dtype=torch.float64)
        eps = 12.5
        loss_rare = drpo_per_token_loss(coef_1, advantages, rare_token_old, epsilon=eps)
        loss_common = drpo_per_token_loss(coef_1, advantages, common_token_old, epsilon=eps)
        # PG term is identical; penalty differs by μ ratio
        penalty_rare = loss_rare.item() + 1.2
        penalty_common = loss_common.item() + 1.2
        assert penalty_common > penalty_rare  # common token bigger penalty
        # Ratio of penalties equals μ_common / μ_rare = 0.95 / 0.05 = 19.
        torch.testing.assert_close(
            torch.tensor(penalty_common / penalty_rare),
            torch.tensor(0.95 / 0.05),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_mu_weighted_false_recovers_plain_spo(self):
        """``mu_weighted=False`` removes the ``μ`` factor on the penalty —
        this is the verl ``spo`` (non-adaptive) baseline. Same penalty
        regardless of rollout-policy token probability."""
        rare = _logps([[0.05]])
        common = _logps([[0.95]])
        coef_1 = torch.tensor([[1.2]])
        advantages = torch.tensor([[1.0]])
        loss_rare = drpo_per_token_loss(
            coef_1, advantages, rare, epsilon=12.5, mu_weighted=False
        )
        loss_common = drpo_per_token_loss(
            coef_1, advantages, common, epsilon=12.5, mu_weighted=False
        )
        torch.testing.assert_close(loss_rare, loss_common)

    def test_rejects_nonpositive_epsilon(self):
        """``epsilon`` is the regularization threshold of an SPO quadratic
        divisor — it must be strictly positive. Validation catches the
        common config-typo case (``ε = 0`` or negative) loudly rather than
        producing nan/inf gradients deep in a training run."""
        old = _logps([[0.5]])
        with pytest.raises(ValueError, match="epsilon must be positive"):
            drpo_per_token_loss(
                torch.ones(1, 1), torch.ones(1, 1), old, epsilon=0.0
            )
        with pytest.raises(ValueError, match="epsilon must be positive"):
            drpo_per_token_loss(
                torch.ones(1, 1), torch.ones(1, 1), old, epsilon=-1.0
            )

    def test_is_differentiable_through_new_logp(self):
        """The ratio ``r_t`` must remain differentiable so the smooth
        Table-1 gradient weight ``1 − sign(A_t(r_t − 1)) · |π − μ| / δ``
        arises via autograd. Backward must produce finite gradients on the
        new-policy log-probs (i.e., on ``coef_1``)."""
        old = _logps([[0.5, 0.5]])
        coef_1 = torch.tensor([[1.1, 1.2]], requires_grad=True)
        advantages = torch.tensor([[1.0]])
        drpo_per_token_loss(coef_1, advantages, old, epsilon=12.5).sum().backward()
        assert coef_1.grad is not None
        assert torch.isfinite(coef_1.grad).all()

    def test_advantages_grad_blocked_via_detach(self):
        """Reference detaches advantages inside the loss (``adv.detach()``)
        so gradients never flow back through the advantage estimator. We
        mirror that — verifies no grad leaks into the advantage tensor
        while normal grad still flows through ``coef_1`` (the new-policy
        log-probs are the only thing the optimizer should be moving)."""
        old = _logps([[0.5]])
        # coef_1 must require grad so backward has a real graph to walk;
        # advantages also requires grad but should NOT receive any.
        coef_1 = torch.tensor([[1.1]], requires_grad=True)
        advantages = torch.tensor([[1.0]], requires_grad=True)
        drpo_per_token_loss(coef_1, advantages, old).sum().backward()
        # Coef_1 carries the policy gradient (sanity check that backward ran)
        assert coef_1.grad is not None
        assert torch.isfinite(coef_1.grad).all()
        # Advantages was detached inside the loss → no grad leaked back
        assert advantages.grad is None or torch.allclose(
            advantages.grad, torch.zeros_like(advantages.grad)
        )


class TestDrpoWiredIntoTrainer:
    """The async GRPO trainer (the call site) must dispatch loss_type 'drpo'."""

    def test_call_site_imports_drpo_loss(self):
        async_trainer = pytest.importorskip("axolotl.core.trainers.grpo.async_trainer")
        assert async_trainer.drpo_per_token_loss is drpo_per_token_loss

    def test_drpo_branch_present_in_dispatch(self):
        """The dispatch must include the new ``drpo`` branch and the
        aggregation tuple must include ``drpo`` so it inherits the same
        loss-aggregation as ``grpo``/``sapo``."""
        import inspect

        async_trainer = pytest.importorskip("axolotl.core.trainers.grpo.async_trainer")
        src = inspect.getsource(async_trainer)
        assert 'self.loss_type == "drpo"' in src
        assert '("grpo", "sapo", "drpo")' in src
        # Argument shape must match the new signature: epsilon + mu_weighted.
        # Pins the wire-in invariant via source inspection (so we don't need
        # the full conftest stack to verify the contract). The knobs are real
        # validated config fields (schemas/trl.py → AsyncGRPOConfig), so the
        # trainer reads them as attributes rather than getattr defaults.
        assert "epsilon=self.args.drpo_epsilon" in src
        assert "mu_weighted=self.args.drpo_mu_weighted" in src
