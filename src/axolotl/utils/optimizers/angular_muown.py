"""
AngularMuown optimizer.

Adapted from "Muown Implicitly Performs Angular Step-size Decay"
(https://arxiv.org/abs/2606.23637, reference implementation
https://github.com/fhueb/angular-muown, MIT licensed).

The paper shows that Muown's directional update is equivalent to a Riemannian
step on the per-row *normalized* directions, while the magnitude of the
un-normalized parameterization only modulates the angular step size. Making
that angular step size explicit yields AngularMuown: each 2D weight matrix is
split into per-row magnitudes ``r`` (updated radially with Adam) and unit-norm
directions ``D`` (updated with a Muon-orthogonalized Riemannian step whose
angular step size is a schedulable multiplier decoupled from ``r``).

Non-matrix parameters (embeddings, heads, norms, biases) fall back to AdamW,
matching the param-splitting contract used by the existing Muon/Dion factories.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

from axolotl.integrations.base import BaseOptimizerFactory

__all__ = ["AngularMuown", "AngularMuownOptimizerFactory"]


# Polar Express adaptive quintic coefficients (one per iteration). Sourced
# from fhueb/angular-muown (the paper's reference implementation, MIT). Each
# row is the (a, b, c) triple applied at step i, falling back to the last
# row for any iteration beyond len(POLAR_EXPRESS_COEFFS). This is a strictly
# more aggressive orthogonalization than classic Muon's fixed coefficients,
# converging to a tighter singular-value band in the same step budget.
POLAR_EXPRESS_COEFFS = (
    (8.237312490495555, -23.157747414558198, 16.680568411445915),
    (4.082441999064836, -2.8930477353325887, 0.5252849256975651),
    (3.9263479922546556, -2.8547468034765293, 0.5318022422894989),
    (3.2982187133085143, -2.4245419810267062, 0.48632008358844075),
    (2.320007312889811, -1.6862169729967622, 0.42068027340235137),
)


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Quintic Polar Express iteration approximating the orthogonalization UV^T
    of a 2D matrix.

    Matches fhueb/angular-muown's ``zeropower_via_polar_express`` (the paper's
    reference implementation) using the 5-level adaptive Polar Express
    coefficients, but stays in float32 on the default device for CPU/test
    portability rather than the reference's bfloat16 + ``torch.compile`` path
    (which targets large-scale GPU pretraining). The function name is preserved
    for callers that already import it; the underlying iteration is now the
    Polar Express variant rather than the classic Muon coefficients.
    """
    assert grad.ndim == 2
    x = grad.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    # Reference divides by (||X|| * 1.01 + eps); the 1.01 keeps singular
    # values strictly inside the contraction region throughout iteration.
    x = x / (x.norm() * 1.01 + eps)
    for step in range(steps):
        a, b, c = POLAR_EXPRESS_COEFFS[min(step, len(POLAR_EXPRESS_COEFFS) - 1)]
        gram = x @ x.T
        # Equivalent to addmm(gram, gram, gram, beta=b, alpha=c) — kept in
        # explicit form so the math is readable on a code review.
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    if transpose:
        x = x.T
    return x.to(grad.dtype)


def _shape_for_u_step(param: torch.Tensor) -> tuple[int, int]:
    """Return the (m, n) shape used for the Muon-style sqrt(max(1, m/n)) scaling
    of the directional update.

    Packed QKV projection matrices (rows == 3 * cols) are orthogonalized as
    three square chunks, so they should use the square chunk shape for scaling
    rather than the packed shape. Mirrors fhueb/angular-muown's
    ``_shape_for_u_step``.
    """
    rows = param.size(0)
    cols = param.size(1)
    if rows == 3 * cols:
        return cols, cols
    return rows, cols


def _orthogonalize_update(update: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Polar-orthogonalize the (Nesterov-mixed) gradient update.

    For packed-QKV projections (rows == 3 * cols) each of the three square
    chunks is orthogonalized independently, which is materially different from
    orthogonalizing the packed matrix as one tall block. Without this, models
    that fuse Q/K/V into a single projection (LLaMA-family ``qkv_proj``) get
    the wrong block structure on the directional step.
    """
    if update.size(0) != 3 * update.size(1):
        return zeropower_via_newtonschulz5(update, steps=ns_steps)
    chunk = update.size(1)
    return torch.cat(
        [
            zeropower_via_newtonschulz5(c, steps=ns_steps)
            for c in update.split(chunk, dim=0)
        ],
        dim=0,
    )


class AngularMuown(Optimizer):
    """Optimizer implementing AngularMuown for 2D ``use_angular`` param groups and
    AdamW for the rest.

    Group flags:
        use_angular: run the angular (directional) update; otherwise AdamW.
        angular_lr:  angular step size base (defaults to ``lr``).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        angular_lr: float | None = None,
        angular_warmup_steps: int = 0,
        # Inverse-polynomial decay: mult = (1 + decay_scale * t_after_warmup) ** (-decay_degree).
        # decay_degree=0 disables decay (multiplier pegged at 1.0 after warmup).
        # Defaults match the paper's framing of "implicit angular decay" as a
        # mild inverse-square-root-ish schedule when made explicit.
        angular_decay_scale: float = 0.0,
        angular_decay_degree: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            angular_lr=angular_lr,
            angular_warmup_steps=angular_warmup_steps,
            angular_decay_scale=angular_decay_scale,
            angular_decay_degree=angular_decay_degree,
            use_angular=True,
        )
        super().__init__(params, defaults)
        # global step counter driving the schedulable angular multiplier
        self._angular_step = 0

    def angular_multiplier(
        self, warmup_steps: int, decay_scale: float, decay_degree: float
    ):
        """Linear warmup to 1.0, then inverse-polynomial decay.

        Matches fhueb/angular-muown's ``_angular_lr_multiplier``: during
        ``warmup_steps`` the multiplier grows linearly from ``1/warmup_steps``
        to ``1.0``; afterwards it follows
        ``(1 + decay_scale * t_after_warmup) ** (-decay_degree)``.

        This is the paper's actual mechanism — *"Muown Implicitly Performs
        Angular Step-size Decay"* names the inverse-polynomial decay shape as
        the implicit schedule Muown induces. Setting ``decay_degree=0``
        recovers a flat post-warmup multiplier of 1.0, matching the paper's
        baseline before the schedule is made explicit.
        """
        step = self._angular_step
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if decay_degree == 0.0:
            return 1.0
        t_after_warmup = max(0, step - warmup_steps)
        return (1.0 + decay_scale * t_after_warmup) ** (-decay_degree)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_angular", True):
                self._step_angular(group)
            else:
                self._step_adamw(group)

        self._angular_step += 1
        return loss

    def _step_angular(self, group):
        lr = group["lr"]
        eps = group["eps"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        angular_lr = group["angular_lr"] if group["angular_lr"] is not None else lr
        theta = angular_lr * self.angular_multiplier(
            group["angular_warmup_steps"],
            group["angular_decay_scale"],
            group["angular_decay_degree"],
        )

        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.ndim != 2:
                # safety fallback for any non-matrix params placed in an angular group
                self._adamw_update(param, grad, group)
                continue

            state = self.state[param]
            if not state:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(grad)
                state["r_exp_avg"] = torch.zeros(
                    grad.size(0), 1, device=grad.device, dtype=grad.dtype
                )
                state["r_exp_avg_sq"] = torch.zeros_like(state["r_exp_avg"])
            state["step"] += 1

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if nesterov else buf
            # Polar-orthogonalize, with per-chunk handling for packed QKV
            ortho = _orthogonalize_update(update, ns_steps=group["ns_steps"])
            # Muon-style shape-invariant scale; uses the chunk shape for
            # packed-QKV layouts (rows == 3 * cols) rather than the packed shape.
            scale_rows, scale_cols = _shape_for_u_step(param)
            ortho.mul_(max(1.0, scale_rows / scale_cols) ** 0.5)

            row_norm = param.norm(dim=1, keepdim=True).clamp_min(eps)
            directions = param / row_norm

            # radial (magnitude) update: Adam on the gradient component along D
            grad_radial = (grad * directions).sum(dim=1, keepdim=True)
            r_exp_avg = state["r_exp_avg"]
            r_exp_avg_sq = state["r_exp_avg_sq"]
            r_exp_avg.mul_(beta1).add_(grad_radial, alpha=1 - beta1)
            r_exp_avg_sq.mul_(beta2).addcmul_(grad_radial, grad_radial, value=1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            r_step = (r_exp_avg / bias1) / ((r_exp_avg_sq / bias2).sqrt() + eps)
            new_norm = (row_norm * (1 - lr * weight_decay) - lr * r_step).clamp_min(eps)

            # angular (directional) update: Riemannian step on the unit sphere
            ortho_radial = (ortho * directions).sum(dim=1, keepdim=True)
            ortho_tangent = ortho - ortho_radial * directions
            new_dir = directions - theta * ortho_tangent
            new_dir = new_dir / new_dir.norm(dim=1, keepdim=True).clamp_min(eps)

            param.copy_(new_norm * new_dir)

    def _step_adamw(self, group):
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            self._adamw_update(param, grad, group)

    def _adamw_update(self, param, grad, group):
        lr = group["lr"]
        eps = group["eps"]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]

        state = self.state[param]
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] += 1

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias1 = 1 - beta1 ** state["step"]
        bias2 = 1 - beta2 ** state["step"]
        denom = (exp_avg_sq / bias2).sqrt().add_(eps)
        param.addcdiv_(exp_avg / bias1, denom, value=-lr)


class AngularMuownOptimizerFactory(BaseOptimizerFactory):
    """Build AngularMuown with the Muon-style param split: 2D hidden matrices use
    the angular update; embeddings, heads, norms and biases use AdamW.
    """

    def __call__(self, opt_model, training_args, **optimizer_kwargs):
        lr = optimizer_kwargs.pop("lr", training_args.learning_rate)
        weight_decay = optimizer_kwargs.pop("weight_decay", training_args.weight_decay)
        # angular hyperparameters may arrive as strings via cfg.optim_args
        angular_lr = optimizer_kwargs.pop("angular_lr", None)
        angular_warmup_steps = int(optimizer_kwargs.pop("angular_warmup_steps", 0))
        # Inverse-polynomial decay parameters matching fhueb/angular-muown.
        # ``angular_decay_scale`` controls how quickly the multiplier shrinks
        # per step; ``angular_decay_degree`` controls the asymptotic shape
        # (0 = no decay, 0.5 ≈ inverse-sqrt, 1 = harmonic, >1 = aggressive).
        angular_decay_scale = float(optimizer_kwargs.pop("angular_decay_scale", 0.0))
        angular_decay_degree = float(optimizer_kwargs.pop("angular_decay_degree", 0.0))
        if angular_lr is not None:
            angular_lr = float(angular_lr)

        decay_params = set(self.get_decay_parameter_names(opt_model))
        angular_params, adamw_decay, adamw_no_decay = [], [], []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            is_matrix = param.ndim == 2 and not (
                "embed" in name or "lm_head" in name or "wte" in name
            )
            if is_matrix:
                angular_params.append(param)
            elif name in decay_params:
                adamw_decay.append(param)
            else:
                adamw_no_decay.append(param)

        groups = []
        if angular_params:
            groups.append(
                {
                    "params": angular_params,
                    "use_angular": True,
                    "weight_decay": weight_decay,
                    "angular_lr": angular_lr,
                    "angular_warmup_steps": angular_warmup_steps,
                    "angular_decay_scale": angular_decay_scale,
                    "angular_decay_degree": angular_decay_degree,
                }
            )
        if adamw_decay:
            groups.append(
                {
                    "params": adamw_decay,
                    "use_angular": False,
                    "weight_decay": weight_decay,
                }
            )
        if adamw_no_decay:
            groups.append(
                {"params": adamw_no_decay, "use_angular": False, "weight_decay": 0.0}
            )

        return AngularMuown(
            groups, lr=lr, weight_decay=weight_decay, **optimizer_kwargs
        )
