"""
Divergence-regularized policy loss for GRPO.

Implements the per-token surrogate from DRPO ("Rethinking the Divergence
Regularization in LLM RL", https://arxiv.org/abs/2606.09821). The reference
implementation lives at https://github.com/Tencent-Hunyuan/UniRL/tree/main/DRPO
(``unirl/algorithms/drpo.py``, ``_drpo_loss``) under the Apache-2.0 license.
This file ports that loss faithfully into Axolotl's existing GRPO dispatch.

The paper's derivation (§3): DRPO takes DPPO's Binary-TV trust region
(``|π(a|s) − μ(a|s)| ≤ δ``), rewrites it as the *token-adaptive ratio bound*
``|r_t − 1| ≤ δ / μ(a_t|s_t)``, and applies SPO's advantage-weighted χ² / ℓ²
construction. Substituting the per-token ``ε_t = ε / μ`` into SPO yields:

    L_t = −A_t · r_t  +  |A_t| · μ · (r_t − 1)² / (2 · ε)

where ``r_t = π / μ`` is the importance ratio, ``μ = exp(old_logp)`` is the
rollout-policy token probability, and ``ε`` is the regularization threshold
(``12.5`` in the paper, §4). The first term is the standard importance-weighted
policy gradient; the second is SPO's advantage-weighted quadratic regularizer
carrying the Binary-TV's token-adaptive ``ε_t = ε / μ`` (§3).

Three properties the reference enforces that we mirror exactly:

* The penalty is **sign-invariant** in the advantage (``|A_t|`` factor) — the
  regularizer's magnitude tracks the size of the advantage, not its sign, so
  large-advantage tokens get the strongest pull back toward the sampling
  policy regardless of update direction.
* The token-adaptive trust region ``ε_t = ε / μ`` is realized by the ``μ``
  factor on the penalty: rare tokens (small ``μ``) get a larger effective
  trust region, common tokens a tighter one. This is the paper's headline §3
  result distinguishing DRPO from plain SPO.
* The ratio ``r_t`` is kept differentiable (no ``.detach()``, no TIS
  truncation), so the smooth Table-1 gradient weight ``1 − sign(A_t (r_t − 1))
  · |π − μ| / δ`` arises naturally via autograd.

A ``mu_weighted=False`` toggle is exposed for parity with the reference's
plain-SPO baseline (verl ``spo`` loss mode); production calls leave it at the
default ``True`` (verl ``spo_adaptive_eps``).
"""

import torch


def drpo_per_token_loss(
    coef_1: torch.Tensor,
    advantages: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    epsilon: float = 12.5,
    mu_weighted: bool = True,
) -> torch.Tensor:
    """Compute the DRPO smooth divergence-regularized per-token loss.

    Faithful port of ``Tencent-Hunyuan/UniRL/unirl/algorithms/drpo.py::_drpo_loss``.

    Args:
        coef_1: Importance ratio ``r_t = exp(logp - logp_old)``, the same
            tensor GRPO would clip in the ``"grpo"`` branch. Must remain
            differentiable through the new-policy log-probs.
        advantages: Per-token (or broadcastable per-sequence) advantages.
        old_per_token_logps: Detached log-probs of the rollout / sampling
            policy. ``μ = exp(old_per_token_logps).detach()`` is recovered
            from this and used as the per-token weight on the quadratic
            penalty — the realization of the token-adaptive ``ε_t = ε / μ``.
        epsilon: Regularization threshold ``ε`` of the SPO quadratic.
            Defaults to ``12.5`` per the paper (§4: *"For SPO and DRPO, we
            set the regularization threshold to 12.5."*). Larger ``ε`` ⇒
            weaker regularization.
        mu_weighted: If ``True`` (default, = verl ``spo_adaptive_eps``),
            apply the Binary-TV token-adaptive ``ε_t = ε / μ`` via the ``μ``
            factor on the penalty — DRPO's §3 contribution. If ``False`` (=
            verl ``spo``), penalty uses a fixed ``ε`` without per-token
            adaptation — the SPO baseline DRPO improves on.

    Returns:
        Per-token loss with the same shape as ``coef_1``. Reduction (mean
        over valid tokens, sequence-mean-token-sum, etc.) is the caller's
        responsibility — matches the existing ``grpo`` / ``sapo`` /
        ``dr_grpo`` branches in ``_compute_loss``.

    Raises:
        ValueError: If ``epsilon <= 0`` — the SPO quadratic is undefined.
    """
    if epsilon <= 0:
        raise ValueError(
            f"epsilon must be positive (it's the regularization threshold of "
            f"the SPO quadratic — paper §4 uses 12.5); got {epsilon}"
        )

    ratio = coef_1  # r_t = π/μ, differentiable through new_logp
    adv_detached = advantages.detach()
    # μ = exp(old_logp) — the rollout-policy token probability. Detached
    # because the rollout policy is fixed in this update step (matches the
    # reference's `torch.exp(old_logp).detach()`).
    old_prob = torch.exp(old_per_token_logps).detach()

    ratio_delta = ratio - 1.0
    # μ-weighted (default) = Binary-TV token-adaptive ε_t = ε / μ → DRPO.
    # ones-weighted = plain SPO with fixed ε (the baseline DRPO improves on).
    penalty_weight = old_prob if mu_weighted else torch.ones_like(old_prob)
    quadratic_penalty = (
        adv_detached.abs() * penalty_weight * ratio_delta.square() / (2.0 * epsilon)
    )

    # L_t = −A_t · r_t  +  |A_t| · (μ or 1) · (r_t − 1)² / (2 · ε)
    return -adv_detached * ratio + quadratic_penalty
