"""Tests for ODPO (DPO with an offset, arxiv:2402.10571).

Covers: the offset loss math, config/arg threading, and RL dispatch through the
existing ``HFRLTrainerBuilder`` (``core/builders/rl.py``) — the integration
surface this feature slots into.
"""

import dataclasses
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from axolotl.core.builders import HFRLTrainerBuilder
from axolotl.core.trainers.dpo import DPOStrategy
from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.dpo.odpo import AxolotlODPOTrainer, odpo_sigmoid_loss
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.config import AxolotlInputConfig
from axolotl.utils.schemas.enums import RLType

BETA = 0.1
OFFSET = 0.5


def _sample_logps():
    """Synthetic policy/reference logps for chosen/rejected sequences."""
    chosen = torch.tensor([1.0, 2.0, -0.5])
    rejected = torch.tensor([0.5, 0.0, 0.3])
    ref_chosen = torch.tensor([0.8, 1.5, -0.2])
    ref_rejected = torch.tensor([0.6, 0.2, 0.1])
    return chosen, rejected, ref_chosen, ref_rejected


def _delta():
    chosen, rejected, ref_chosen, ref_rejected = _sample_logps()
    return (chosen - ref_chosen) - (rejected - ref_rejected)


class TestODPOLoss:
    """Pure loss-function tests for the ODPO offset margin."""

    def test_loss_matches_offset_formula(self):
        # loss = -log σ(β·Δ − δ)
        expected = -F.logsigmoid(BETA * _delta() - OFFSET)
        got = odpo_sigmoid_loss(*_sample_logps(), BETA, OFFSET)
        assert torch.allclose(got, expected)

    def test_offset_zero_reduces_to_dpo(self):
        # δ = 0 is exactly base DPO: -log σ(β·Δ)
        expected_dpo = -F.logsigmoid(BETA * _delta())
        got = odpo_sigmoid_loss(*_sample_logps(), BETA, 0.0)
        assert torch.allclose(got, expected_dpo)

    def test_larger_offset_raises_loss(self):
        # Raising δ raises the bar, so the loss is non-decreasing in δ.
        base = odpo_sigmoid_loss(*_sample_logps(), BETA, 0.0)
        raised = odpo_sigmoid_loss(*_sample_logps(), BETA, OFFSET)
        assert torch.all(raised >= base - 1e-6)
        assert float(raised.mean()) > float(base.mean())


class TestODPOConfigThreading:
    """dpo_offset flows cfg → AxolotlDPOConfig → training args."""

    def test_input_config_offset_is_optional(self):
        # Introspect (rather than build) so the test doesn't depend on full
        # Pydantic validation of a heavy config object.
        field = AxolotlInputConfig.model_fields["dpo_offset"]
        assert not field.is_required()
        assert field.default is None

    def test_dpo_config_offset_defaults_zero(self):
        # AxolotlDPOConfig is the shared args class for DPO/IPO/ODPO; offset
        # defaults to 0.0 so plain DPO is unaffected.
        fields = {f.name: f for f in dataclasses.fields(AxolotlDPOConfig)}
        assert "dpo_offset" in fields
        assert fields["dpo_offset"].default == 0.0

    def test_strategy_threads_offset_into_kwargs(self):
        cfg = DictDefault({"rl": RLType.ODPO, "dpo_offset": 0.5, "sequence_len": 512})
        kwargs = DPOStrategy.set_training_args_kwargs(cfg)
        assert kwargs["dpo_offset"] == 0.5


class TestODPODispatch:
    """ODPO is dispatched through the existing RL builder like DPO/IPO."""

    @staticmethod
    def _builder(rl_cfg):
        # _get_trainer_cls / _build_training_arguments don't touch the model or
        # tokenizer objects, so sentinels are sufficient to exercise dispatch.
        builder = HFRLTrainerBuilder(rl_cfg, object(), object())
        builder.model_ref = object()
        return builder

    def test_get_trainer_cls_returns_odpo_trainer(self):
        cfg = DictDefault({"rl": RLType.ODPO, "plugins": None, "trainer_cls": None})
        builder = self._builder(cfg)
        trainer_cls, trainer_cls_args = builder._get_trainer_cls({})
        assert trainer_cls is AxolotlODPOTrainer
        # [model, model_ref] — same shape as the DPO/IPO branch.
        assert len(trainer_cls_args) == 2

    def test_training_arguments_thread_offset(self, base_cfg):
        cfg = base_cfg.copy()
        cfg.update({"rl": RLType.ODPO, "dpo_offset": 0.5})
        builder = self._builder(cfg)
        training_args, _ = builder._build_training_arguments(100)
        assert isinstance(training_args, AxolotlDPOConfig)
        assert training_args.dpo_offset == 0.5

    def test_dpo_branch_unchanged_by_odpo_addition(self, base_cfg):
        # Regression guard: plain DPO still dispatches to the DPO trainer.
        cfg = base_cfg.copy()
        cfg.update({"rl": RLType.DPO, "beta": 0.1})
        builder = self._builder(cfg)
        trainer_cls, _ = builder._get_trainer_cls({})
        from axolotl.core.trainers.dpo import AxolotlDPOTrainer

        assert trainer_cls is AxolotlDPOTrainer


class TestODPOComputeLoss:
    """End-to-end check of the overridden ``_compute_loss`` offset path.

    Builds a tiny model and drives the trainer's loss computation directly
    (bypassing TRL's dataset-heavy ``__init__``) to confirm the offset margin,
    the metric logging, and the unsupported-setup guard all behave.
    """

    @staticmethod
    def _trainer(model, ref_model, offset):
        from transformers import LlamaConfig, LlamaForCausalLM

        if model is None:
            cfg = LlamaConfig(
                vocab_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_size=16,
                intermediate_size=32,
                max_position_embeddings=64,
                num_key_value_heads=2,
                rope_scaling=None,
            )
            model = LlamaForCausalLM(cfg).eval()
            ref_model = LlamaForCausalLM(cfg).eval()

        tr = AxolotlODPOTrainer.__new__(AxolotlODPOTrainer)
        tr.model = model
        tr.ref_model = ref_model
        tr.beta = BETA
        tr.loss_types = ["sigmoid"]
        tr.loss_weights = [1.0]
        tr.f_divergence_type = "reverse_kl"
        tr.use_weighting = False
        tr.ld_alpha = None
        tr.precompute_ref_logps = False
        tr.label_smoothing = 0.0
        tr._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        tr._total_train_tokens = 0
        tr.args = SimpleNamespace(dpo_offset=offset, gradient_checkpointing_kwargs=None)

        class _FakeAccel:
            device = torch.device("cpu")

            def gather_for_metrics(self, t):
                return t

            def gather(self, t):
                return t

            def unwrap_model(self, m):
                return m

        tr.accelerator = _FakeAccel()
        model.training = True
        return model, ref_model, tr

    @staticmethod
    def _batch():
        torch.manual_seed(0)
        batch, seq = 2, 16
        input_ids = torch.randint(0, 32, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        completion_mask = torch.zeros(batch, seq, dtype=torch.long)
        completion_mask[:, seq // 2 :] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

    def test_compute_loss_matches_offset_formula(self):
        model, ref_model, tr = self._trainer(None, None, OFFSET)
        inputs = self._batch()
        loss = tr._compute_loss(model, inputs, return_outputs=False)

        # Closed-form ODPO loss on the same forward.
        labels = inputs["input_ids"][:, 1:].contiguous()
        cmask = inputs["completion_mask"][:, 1:].contiguous()

        def seq_logps(m):
            out = m(
                **{k: v for k, v in inputs.items() if k != "completion_mask"},
                use_cache=False,
            )
            return (
                torch.log_softmax(out.logits[:, :-1, :], -1)
                .gather(-1, labels.unsqueeze(-1))
                .squeeze(-1)
                * cmask
            ).sum(1)

        chosen, rejected = seq_logps(model).chunk(2)
        ref_chosen, ref_rejected = seq_logps(ref_model).chunk(2)
        expected = odpo_sigmoid_loss(
            chosen, rejected, ref_chosen, ref_rejected, BETA, OFFSET
        ).mean()

        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected, atol=1e-5)
        # The standard DPO dashboard metrics still get logged.
        assert "rewards/margins" in tr._metrics["train"]
        assert "logps/chosen" in tr._metrics["train"]

    def test_offset_raises_loss_vs_dpo(self):
        model, ref_model, tr_off = self._trainer(None, None, OFFSET)
        inputs = self._batch()
        loss_off = tr_off._compute_loss(model, inputs, return_outputs=False)
        _, _, tr_zero = self._trainer(model, ref_model, 0.0)
        loss_zero = tr_zero._compute_loss(model, inputs, return_outputs=False)
        assert float(loss_off) > float(loss_zero)

    def test_guard_rejects_unsupported_setup(self):
        model, ref_model, tr = self._trainer(None, None, OFFSET)
        tr.loss_types = ["ipo"]  # ODPO offset is sigmoid-DPO-specific
        with pytest.raises(ValueError, match="dpo_offset"):
            tr._compute_loss(model, self._batch(), return_outputs=False)
