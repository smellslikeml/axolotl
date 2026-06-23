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


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz iteration approximating the orthogonalization UV^T
    of a 2D matrix, as used by Muon. Operates in float32 for CPU/test stability.
    """
    assert grad.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * gram @ gram
        x = a * x + poly @ x
    if transpose:
        x = x.T
    return x.to(grad.dtype)


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
        angular_decay_steps: int = 0,
        angular_min_mult: float = 1.0,
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
            angular_decay_steps=angular_decay_steps,
            angular_min_mult=angular_min_mult,
            use_angular=True,
        )
        super().__init__(params, defaults)
        # global step counter driving the schedulable angular multiplier
        self._angular_step = 0

    def angular_multiplier(self, warmup_steps: int, decay_steps: int, min_mult: float):
        """Linear warmup to 1.0, then optional linear decay to ``min_mult``.

        This is the explicit, schedulable angular step-size factor the paper
        argues for — decoupled from the radial (magnitude) update.
        """
        step = self._angular_step
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if decay_steps > 0:
            progress = min(1.0, max(0, step - warmup_steps) / decay_steps)
            return 1.0 + (min_mult - 1.0) * progress
        return 1.0

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
            group["angular_decay_steps"],
            group["angular_min_mult"],
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
            ortho = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
            # Muon-style scale so the update RMS is shape-invariant
            rows, cols = param.shape
            ortho.mul_(max(1.0, rows / cols) ** 0.5)

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
        angular_decay_steps = int(optimizer_kwargs.pop("angular_decay_steps", 0))
        angular_min_mult = float(optimizer_kwargs.pop("angular_min_mult", 1.0))
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
                    "angular_decay_steps": angular_decay_steps,
                    "angular_min_mult": angular_min_mult,
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
