"""DPO with an Offset (ODPO) trainer — arxiv:2402.10571.

ODPO generalizes sigmoid DPO by requiring the policy's implicit reward for the
chosen response to beat the rejected one by more than a configurable offset ``δ``::

    loss = -log σ( β · (Δπ − Δπ_ref) − δ )

``δ = 0`` is exactly DPO; ``δ > 0`` raises the preference bar so pairs where the
chosen response is only marginally better are penalized less aggressively. The
offset is a constant config margin (``dpo_offset``) — per-pair offsets from
preference-strength metadata would need a dataset-schema change and are out of
scope.

TRL's ``DPOTrainer._compute_loss`` (trl==1.5.1) computes the policy/reference
logps and the loss inline and returns only ``(loss, outputs)``, so there is no
hook to shift the sigmoid margin. With ``dpo_offset == 0`` we delegate to the
base trainer verbatim (zero overhead, every loss variant). With an offset we
reproduce the standard sigmoid / reverse-KL DPO step once — a single policy and
single reference forward, identical cost to plain DPO — and log the same metrics.
ODPO is defined for the standard sigmoid DPO loss, so the offset path is scoped
to that and refuses configurations it does not model rather than silently
mis-optimizing.
"""

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import (
    entropy_from_logits,
    selective_log_softmax,
    use_adapter,
)

from axolotl.core.trainers.dpo import DPOStrategy
from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer


def odpo_sigmoid_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    offset: float,
) -> torch.Tensor:
    """Per-sequence ODPO loss for the standard (reverse-KL) sigmoid DPO setting.

    ``-log σ( β · ((π_c−π_r) − (πref_c−πref_r)) − δ )``; ``offset = 0`` is DPO.
    """
    delta = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * delta - offset)


class AxolotlODPOTrainer(AxolotlDPOTrainer):
    """DPO trainer that applies an offset margin (ODPO) to the sigmoid loss."""

    tag_names = ["axolotl", "dpo", "odpo"]

    def _supported_offset_setup(self) -> bool:
        return (
            list(self.loss_types) == ["sigmoid"]
            and self.f_divergence_type == "reverse_kl"
            and not self.use_weighting
            and self.ld_alpha is None
        )

    def _compute_loss(self, model, inputs, return_outputs):
        offset = getattr(self.args, "dpo_offset", 0.0)
        if not offset:
            return super()._compute_loss(model, inputs, return_outputs)
        if not self._supported_offset_setup():
            raise ValueError(
                "dpo_offset (ODPO, arxiv:2402.10571) supports the standard DPO "
                "setup only: sigmoid loss, reverse-KL f-divergence, no "
                "dpo_use_weighting, no ld_alpha. Set dpo_offset: 0.0 to use "
                "other DPO loss variants."
            )

        mode = "train" if self.model.training else "eval"

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        if self.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
        else:
            # Mirror DPOTrainer: disable gradient checkpointing for the no-grad
            # reference forward, and for PEFT use the disabled/base adapter.
            with (
                torch.no_grad(),
                disable_gradient_checkpointing(
                    self.model, self.args.gradient_checkpointing_kwargs
                ),
            ):
                if is_peft_model(model) and self.ref_model is None:
                    model = self.accelerator.unwrap_model(model)
                    with use_adapter(
                        model,
                        adapter_name="ref" if "ref" in model.peft_config else None,
                    ):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        per_sequence_loss = odpo_sigmoid_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self.beta,
            float(offset),
        )
        loss = per_sequence_loss.mean()

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        # Metrics — mirror DPOTrainer so ODPO runs keep the same DPO dashboards.
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        mask = shift_completion_mask
        entropy_sum = (per_token_entropy * mask).sum()
        total_tokens = mask.sum()
        entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
        total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
        entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = (
                self.accelerator.gather_for_metrics(inputs["attention_mask"].sum())
                .sum()
                .item()
            )
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        chosen_logits, rejected_logits = shift_logits.detach().chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask.sum()
        total_chosen_logits = (
            self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        )
        total_chosen_tokens = (
            self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        )
        total_rejected_logits = (
            self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        )
        total_rejected_tokens = (
            self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        )
        avg_chosen_logits = (
            total_chosen_logits / total_chosen_tokens
            if total_chosen_tokens > 0
            else 0.0
        )
        avg_rejected_logits = (
            total_rejected_logits / total_rejected_tokens
            if total_rejected_tokens > 0
            else 0.0
        )
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        predictions = chosen_logits.argmax(dim=-1)
        chosen_mask = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
        chosen_labels = shift_labels[: len(shift_labels) // 2]
        correct_predictions = (predictions == chosen_labels) & chosen_mask
        chosen_total = chosen_mask.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_predictions.sum())
        chosen_total = self.accelerator.gather_for_metrics(chosen_total)
        total_sum = chosen_total.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self._metrics[mode]["rewards/chosen"].append(
            self.accelerator.gather(chosen_rewards).mean().item()
        )
        self._metrics[mode]["rewards/rejected"].append(
            self.accelerator.gather(rejected_rewards).mean().item()
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        self._metrics[mode]["rewards/accuracies"].append(
            self.accelerator.gather(reward_accuracies).mean().item()
        )
        margins = chosen_rewards - rejected_rewards
        self._metrics[mode]["rewards/margins"].append(
            self.accelerator.gather(margins).mean().item()
        )
        self._metrics[mode]["logps/chosen"].append(
            self.accelerator.gather(chosen_logps).mean().item()
        )
        self._metrics[mode]["logps/rejected"].append(
            self.accelerator.gather(rejected_logps).mean().item()
        )

        return (loss, outputs) if return_outputs else loss


class ODPOStrategy(DPOStrategy):
    """Strategy for ODPO training — DPO with an offset margin.

    Reuses ``DPOStrategy`` for arg threading; only the trainer class differs.
    """

    @classmethod
    def get_trainer_class(cls):
        return AxolotlODPOTrainer

    @classmethod
    def get_training_args_class(cls):
        return AxolotlDPOConfig
