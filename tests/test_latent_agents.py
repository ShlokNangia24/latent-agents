"""Tests for the latent-agents library.

Unit tests use mocks/fakes so they run instantly on CPU without downloading
models.  Integration tests (marked ``@pytest.mark.integration``) load a real
small HuggingFace model and exercise the full pipeline end-to-end.

Run unit tests only:
    pytest tests/test_latent_agents.py -v

Run everything (needs network + a GPU or patience on CPU):
    pytest tests/test_latent_agents.py -v --run-integration
"""

from __future__ import annotations

import os
import random
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from latent_agents import (
    Agent,
    LatentModel,
    LatentPipeline,
    LatentRealigner,
    PipelineResult,
    auto_device,
    set_seed,
)
from latent_agents.model import _ensure_pad_token, past_kv_length
from latent_agents.pipeline import (
    _extract_answer,
    _majority_vote,
    _slice_tensor,
    truncate_kv_cache,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
VOCAB_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
DEVICE = torch.device("cpu")


def _make_fake_embeddings() -> Tuple[nn.Embedding, nn.Linear]:
    """Return a small input-embedding and lm_head for testing."""
    input_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
    lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
    return input_embed, lm_head


def _make_fake_kv_cache(
    batch_size: int = 1,
    seq_len: int = 10,
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Create a dummy KV-cache tuple."""
    cache: list = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.append((k, v))
    return tuple(cache)


class _FakeModel(nn.Module):
    """Minimal fake causal-LM that mimics the HuggingFace interface."""

    def __init__(self):
        super().__init__()
        self.input_embed, self.lm_head = _make_fake_embeddings()
        self.config = MagicMock()
        self.config.use_cache = True

    def get_input_embeddings(self):
        return self.input_embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size):
        pass


# ===================================================================
# Tests: Agent
# ===================================================================


class TestAgent:
    """Tests for the Agent dataclass."""

    def test_creation(self):
        agent = Agent(
            name="Thinker",
            role="thinker",
            prompt_fn=lambda q, c: [{"role": "user", "content": q}],
        )
        assert agent.name == "Thinker"
        assert agent.role == "thinker"
        assert agent.is_final is False

    def test_final_agent(self):
        agent = Agent(
            name="Speaker",
            role="speaker",
            prompt_fn=lambda q, c: [],
            is_final=True,
        )
        assert agent.is_final is True

    def test_prompt_fn_called(self):
        prompt_fn = lambda q, c: [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"{q} | {c}"},
        ]
        agent = Agent(name="A", role="a", prompt_fn=prompt_fn)
        msgs = agent.prompt_fn("hello", "ctx")
        assert len(msgs) == 2
        assert "hello" in msgs[1]["content"]
        assert "ctx" in msgs[1]["content"]

    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(Agent)}
        assert field_names == {"name", "role", "prompt_fn", "is_final", "latent_steps", "convergence_threshold"}


# ===================================================================
# Tests: utils
# ===================================================================


class TestUtils:
    """Tests for set_seed and auto_device."""

    def test_set_seed_reproducibility(self):
        set_seed(123)
        a = torch.randn(5)
        r1 = random.random()
        n1 = np.random.rand()

        set_seed(123)
        b = torch.randn(5)
        r2 = random.random()
        n2 = np.random.rand()

        assert torch.allclose(a, b)
        assert r1 == r2
        assert n1 == n2

    def test_set_seed_env_var(self):
        set_seed(999)
        assert os.environ["PYTHONHASHSEED"] == "999"

    def test_auto_device_explicit(self):
        dev = auto_device("cpu")
        assert dev == torch.device("cpu")

    def test_auto_device_default(self):
        dev = auto_device()
        if torch.cuda.is_available():
            assert dev.type == "cuda"
        else:
            assert dev.type == "cpu"


# ===================================================================
# Tests: model helpers
# ===================================================================


class TestModelHelpers:
    """Tests for _ensure_pad_token and past_kv_length."""

    def test_ensure_pad_token_already_set(self):
        tok = MagicMock()
        tok.pad_token_id = 0
        _ensure_pad_token(tok)
        # Should not modify anything
        assert tok.pad_token_id == 0

    def test_ensure_pad_token_uses_eos(self):
        tok = MagicMock()
        tok.pad_token_id = None
        tok.eos_token = "<eos>"
        _ensure_pad_token(tok)
        assert tok.pad_token == "<eos>"

    def test_ensure_pad_token_adds_special(self):
        tok = MagicMock()
        tok.pad_token_id = None
        tok.eos_token = None
        _ensure_pad_token(tok)
        tok.add_special_tokens.assert_called_once_with({"pad_token": "<pad>"})

    def test_past_kv_length_none(self):
        assert past_kv_length(None) == 0

    def test_past_kv_length_empty(self):
        assert past_kv_length(()) == 0

    def test_past_kv_length_with_cache(self):
        cache = _make_fake_kv_cache(seq_len=42)
        assert past_kv_length(cache) == 42


# ===================================================================
# Tests: LatentRealigner
# ===================================================================


class TestLatentRealigner:
    """Tests for the LatentRealigner class."""

    def _build_realigner(self, enabled: bool = True) -> LatentRealigner:
        model = _FakeModel()
        return LatentRealigner(model, DEVICE, enabled=enabled)

    def test_build_enabled(self):
        r = self._build_realigner(enabled=True)
        assert r._matrix is not None
        assert r._target_norm is not None
        assert r._matrix.shape == (HIDDEN_DIM, HIDDEN_DIM)

    def test_build_disabled_is_identity(self):
        r = self._build_realigner(enabled=False)
        expected = torch.eye(HIDDEN_DIM, dtype=r._matrix.dtype)
        assert torch.allclose(r._matrix, expected)

    def test_apply_2d(self):
        r = self._build_realigner()
        hidden = torch.randn(2, HIDDEN_DIM)
        out = r.apply(hidden)
        assert out.shape == hidden.shape

    def test_apply_3d(self):
        r = self._build_realigner()
        hidden = torch.randn(2, 5, HIDDEN_DIM)
        out = r.apply(hidden)
        assert out.shape == hidden.shape

    def test_apply_preserves_dtype(self):
        r = self._build_realigner()
        hidden = torch.randn(1, HIDDEN_DIM, dtype=torch.float16)
        out = r.apply(hidden)
        assert out.dtype == torch.float16

    def test_apply_norm_matching(self):
        """After realignment, vectors should have norms close to the target."""
        r = self._build_realigner()
        hidden = torch.randn(10, HIDDEN_DIM) * 100  # large norms
        out = r.apply(hidden)
        norms = out.norm(dim=-1)
        # All output norms should be close to the target norm
        target = r._target_norm.item()
        assert torch.allclose(norms, torch.full_like(norms, target), atol=1e-3)

    def test_missing_embeddings_raises(self):
        """If the model has no embeddings, construction should fail."""
        model = MagicMock()
        model.get_input_embeddings.return_value = None
        model.get_output_embeddings.return_value = None
        with pytest.raises(RuntimeError, match="Cannot build latent realignment"):
            LatentRealigner(model, DEVICE)


# ===================================================================
# Tests: KV-cache helpers (pipeline module)
# ===================================================================


class TestKVCacheHelpers:
    """Tests for _slice_tensor and truncate_kv_cache."""

    def test_slice_tensor_keep_all(self):
        t = torch.randn(1, 2, 10, 8)
        out = _slice_tensor(t, tokens_to_keep=10)
        assert out.shape == t.shape

    def test_slice_tensor_keep_some(self):
        t = torch.randn(1, 2, 10, 8)
        out = _slice_tensor(t, tokens_to_keep=3)
        assert out.shape == (1, 2, 3, 8)
        # Should be the last 3 entries
        assert torch.equal(out, t[..., 7:, :])

    def test_slice_tensor_keep_zero(self):
        t = torch.randn(1, 2, 10, 8)
        out = _slice_tensor(t, tokens_to_keep=0)
        assert out.shape[-2] == 0

    def test_slice_tensor_keep_more_than_exists(self):
        t = torch.randn(1, 2, 5, 8)
        out = _slice_tensor(t, tokens_to_keep=100)
        assert out.shape == t.shape

    def test_truncate_kv_cache_none(self):
        assert truncate_kv_cache(None, 5) is None

    def test_truncate_kv_cache_zero_keep(self):
        cache = _make_fake_kv_cache(seq_len=10)
        assert truncate_kv_cache(cache, 0) is None

    def test_truncate_kv_cache_tuple(self):
        cache = _make_fake_kv_cache(seq_len=20)
        trimmed = truncate_kv_cache(cache, tokens_to_keep=5)
        assert trimmed is not None
        # Check all layers got trimmed
        for layer in trimmed:
            assert layer[0].shape[-2] == 5
            assert layer[1].shape[-2] == 5


# ===================================================================
# Tests: PipelineResult
# ===================================================================


class TestPipelineResult:
    """Tests for the PipelineResult dataclass."""

    def test_creation(self):
        r = PipelineResult(text="hello world")
        assert r.text == "hello world"
        assert r.agent_traces == []

    def test_with_traces(self):
        traces = [
            {"name": "A", "role": "a", "output": ""},
            {"name": "B", "role": "b", "output": "answer"},
        ]
        r = PipelineResult(text="answer", agent_traces=traces)
        assert len(r.agent_traces) == 2
        assert r.agent_traces[1]["output"] == "answer"


# ===================================================================
# Tests: LatentPipeline validation
# ===================================================================


class TestLatentPipelineValidation:
    """Tests for LatentPipeline construction validation."""

    def _dummy_agent(self, name: str, is_final: bool = False) -> Agent:
        return Agent(
            name=name,
            role=name.lower(),
            prompt_fn=lambda q, c: [{"role": "user", "content": q}],
            is_final=is_final,
        )

    def test_no_final_agent_raises(self):
        agents = [self._dummy_agent("A"), self._dummy_agent("B")]
        with pytest.raises(ValueError, match="Exactly one agent"):
            LatentPipeline(MagicMock(), agents)

    def test_multiple_final_agents_raises(self):
        agents = [
            self._dummy_agent("A", is_final=True),
            self._dummy_agent("B", is_final=True),
        ]
        with pytest.raises(ValueError, match="Exactly one agent"):
            LatentPipeline(MagicMock(), agents)

    def test_single_final_agent_ok(self):
        agents = [
            self._dummy_agent("A"),
            self._dummy_agent("B", is_final=True),
        ]
        pipeline = LatentPipeline(MagicMock(), agents)
        assert len(pipeline.agents) == 2

    def test_default_parameters(self):
        agents = [self._dummy_agent("A", is_final=True)]
        pipeline = LatentPipeline(MagicMock(), agents)
        assert pipeline.latent_steps == 20
        assert pipeline.max_new_tokens == 2048
        assert pipeline.temperature == 0.6
        assert pipeline.top_p == 0.95
        assert pipeline.keep_only_latent is False

    def test_custom_parameters(self):
        agents = [self._dummy_agent("A", is_final=True)]
        pipeline = LatentPipeline(
            MagicMock(),
            agents,
            latent_steps=10,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
            keep_only_latent=True,
        )
        assert pipeline.latent_steps == 10
        assert pipeline.max_new_tokens == 512
        assert pipeline.temperature == 0.8
        assert pipeline.top_p == 0.9
        assert pipeline.keep_only_latent is True


# ===================================================================
# Tests: LatentPipeline with mocked model
# ===================================================================


class TestLatentPipelineWithMocks:
    """Tests that exercise pipeline logic by mocking the LatentModel."""

    def _make_mock_model(self) -> MagicMock:
        """Create a mock LatentModel that returns deterministic results."""
        model = MagicMock(spec=LatentModel)

        # prepare_chat_batch returns (prompts, input_ids, attention_mask)
        def fake_prepare(batch_messages, add_generation_prompt=True):
            bs = len(batch_messages)
            prompts = [str(m) for m in batch_messages]
            ids = torch.ones(bs, 5, dtype=torch.long)
            mask = torch.ones(bs, 5, dtype=torch.long)
            return prompts, ids, mask

        model.prepare_chat_batch.side_effect = fake_prepare

        # generate_latent_batch returns (fake KV-cache, actual_steps)
        def fake_latent(input_ids, attention_mask=None, latent_steps=20,
                        past_key_values=None, convergence_threshold=None):
            bs = input_ids.shape[0]
            seq_len = 10 if past_key_values is None else past_kv_length(past_key_values) + 10
            return _make_fake_kv_cache(batch_size=bs, seq_len=seq_len), latent_steps

        model.generate_latent_batch.side_effect = fake_latent

        # generate_text_batch returns fake text
        def fake_text(input_ids, attention_mask, max_new_tokens=256,
                      temperature=0.7, top_p=0.95, past_key_values=None):
            bs = input_ids.shape[0]
            texts = [f"Answer_{i}" for i in range(bs)]
            return texts, None

        model.generate_text_batch.side_effect = fake_text

        return model

    def test_single_agent_pipeline(self):
        """Pipeline with only one final agent -- no latent steps."""
        model = self._make_mock_model()
        agents = [
            Agent(
                name="Solver",
                role="solver",
                prompt_fn=lambda q, c: [{"role": "user", "content": q}],
                is_final=True,
            ),
        ]
        pipeline = LatentPipeline(model, agents)
        result = pipeline.run("What is 2+2?")

        assert isinstance(result, PipelineResult)
        assert result.text == "Answer_0"
        assert len(result.agent_traces) == 1
        assert result.agent_traces[0]["name"] == "Solver"
        model.generate_latent_batch.assert_not_called()

    def test_two_agent_pipeline(self):
        """Pipeline with one latent agent and one final agent."""
        model = self._make_mock_model()
        agents = [
            Agent(
                name="Thinker",
                role="thinker",
                prompt_fn=lambda q, c: [{"role": "user", "content": f"Think: {q}"}],
            ),
            Agent(
                name="Speaker",
                role="speaker",
                prompt_fn=lambda q, c: [{"role": "user", "content": f"Answer: {q}"}],
                is_final=True,
            ),
        ]
        pipeline = LatentPipeline(model, agents)
        result = pipeline.run("Why is the sky blue?")

        assert result.text == "Answer_0"
        assert len(result.agent_traces) == 2
        assert result.agent_traces[0]["name"] == "Thinker"
        assert result.agent_traces[0]["output"] == ""
        assert "latent_steps" in result.agent_traces[0]
        assert result.agent_traces[1]["name"] == "Speaker"
        model.generate_latent_batch.assert_called_once()
        model.generate_text_batch.assert_called_once()

    def test_four_agent_pipeline(self):
        """Pipeline with three latent agents and one final agent."""
        model = self._make_mock_model()
        agents = [
            Agent(name="Planner", role="planner",
                  prompt_fn=lambda q, c: [{"role": "user", "content": f"Plan: {q}"}]),
            Agent(name="Critic", role="critic",
                  prompt_fn=lambda q, c: [{"role": "user", "content": f"Critique: {q}"}]),
            Agent(name="Refiner", role="refiner",
                  prompt_fn=lambda q, c: [{"role": "user", "content": f"Refine: {q}"}]),
            Agent(name="Solver", role="solver", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": f"Solve: {q}"}]),
        ]
        pipeline = LatentPipeline(model, agents)
        result = pipeline.run("Hard problem")

        assert len(result.agent_traces) == 4
        assert model.generate_latent_batch.call_count == 3
        assert model.generate_text_batch.call_count == 1

    def test_batch_processing(self):
        """run_batch should return one result per question."""
        model = self._make_mock_model()
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents)
        questions = ["Q1", "Q2", "Q3"]
        results = pipeline.run_batch(questions)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert isinstance(r, PipelineResult)
            assert r.text == f"Answer_{i}"
            assert len(r.agent_traces) == 2

    def test_keep_only_latent(self):
        """When keep_only_latent=True, KV-cache should be truncated."""
        model = self._make_mock_model()
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, keep_only_latent=True)
        result = pipeline.run("test")

        # Should still produce a valid result
        assert isinstance(result, PipelineResult)
        assert result.text == "Answer_0"

    def test_context_passed_to_prompt_fn(self):
        """The context kwarg should be forwarded to each agent's prompt_fn."""
        received_contexts: List[str] = []

        def capturing_prompt(q, c):
            received_contexts.append(c)
            return [{"role": "user", "content": q}]

        model = self._make_mock_model()
        agents = [
            Agent(name="A", role="a", prompt_fn=capturing_prompt, is_final=True),
        ]
        pipeline = LatentPipeline(model, agents)
        pipeline.run("q", context="my-context")

        assert received_contexts == ["my-context"]


# ===================================================================
# Tests: LatentModel (mocked internals)
# ===================================================================


class TestLatentModelRenderChat:
    """Test render_chat with a mocked tokenizer."""

    def _make_latent_model_mock(self) -> LatentModel:
        """Create a LatentModel without actually loading a real model."""
        with patch.object(LatentModel, "__init__", lambda self, *a, **kw: None):
            lm = LatentModel.__new__(LatentModel)

        lm.device = DEVICE
        lm.tokenizer = MagicMock()
        lm.tokenizer.chat_template = None  # force fallback
        lm.tokenizer.pad_token_id = 0
        lm.model = _FakeModel()
        lm.realigner = LatentRealigner(lm.model, DEVICE)
        return lm

    def test_render_chat_fallback(self):
        lm = self._make_latent_model_mock()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        rendered = lm.render_chat(messages, add_generation_prompt=True)
        assert "<|system|>" in rendered
        assert "You are helpful." in rendered
        assert "<|user|>" in rendered
        assert "Hello" in rendered
        assert "<|assistant|>" in rendered

    def test_render_chat_no_generation_prompt(self):
        lm = self._make_latent_model_mock()
        messages = [{"role": "user", "content": "Hi"}]
        rendered = lm.render_chat(messages, add_generation_prompt=False)
        assert "<|assistant|>" not in rendered

    def test_render_chat_uses_template_when_available(self):
        lm = self._make_latent_model_mock()
        lm.tokenizer.chat_template = "some_template"
        lm.tokenizer.apply_chat_template.return_value = "TEMPLATED"

        result = lm.render_chat([{"role": "user", "content": "hi"}])
        assert result == "TEMPLATED"
        lm.tokenizer.apply_chat_template.assert_called_once()


# ===================================================================
# Integration tests (require a real model)
# ===================================================================

SMALL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.integration
class TestIntegration:
    """End-to-end tests with a real (small) HuggingFace model.

    These tests are slow and require network access on first run.
    Enable with ``pytest --run-integration``.
    """

    @pytest.fixture(scope="class")
    def model(self):
        """Load the model once for all integration tests."""
        return LatentModel(SMALL_MODEL, device="cpu", realign=True)

    def test_model_loads(self, model):
        assert model.tokenizer is not None
        assert model.model is not None
        assert model.realigner is not None

    def test_prepare_chat_batch(self, model):
        messages = [[{"role": "user", "content": "Hello"}]]
        prompts, input_ids, attention_mask = model.prepare_chat_batch(messages)
        assert len(prompts) == 1
        assert input_ids.shape[0] == 1
        assert attention_mask.shape == input_ids.shape

    def test_generate_text(self, model):
        messages = [[{"role": "user", "content": "Say hello."}]]
        _, input_ids, attention_mask = model.prepare_chat_batch(messages)
        texts, kv = model.generate_text_batch(
            input_ids, attention_mask, max_new_tokens=20, temperature=0.7,
        )
        assert len(texts) == 1
        assert isinstance(texts[0], str)
        assert len(texts[0]) > 0

    def test_generate_latent(self, model):
        messages = [[{"role": "user", "content": "Think about 2+2."}]]
        _, input_ids, attention_mask = model.prepare_chat_batch(messages)
        kv, actual_steps = model.generate_latent_batch(
            input_ids, attention_mask=attention_mask, latent_steps=3,
        )
        assert kv is not None
        assert actual_steps == 3
        assert past_kv_length(kv) > input_ids.shape[1]

    def test_full_pipeline_two_agents(self, model):
        set_seed(42)
        agents = [
            Agent(
                name="Thinker",
                role="thinker",
                prompt_fn=lambda q, c: [{"role": "user", "content": f"Think: {q}"}],
            ),
            Agent(
                name="Speaker",
                role="speaker",
                prompt_fn=lambda q, c: [{"role": "user", "content": f"Answer: {q}"}],
                is_final=True,
            ),
        ]
        pipeline = LatentPipeline(
            model, agents, latent_steps=5, max_new_tokens=50,
        )
        result = pipeline.run("What is 2+2?")
        assert isinstance(result, PipelineResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert len(result.agent_traces) == 2

    def test_full_pipeline_batch(self, model):
        set_seed(42)
        agents = [
            Agent(
                name="T",
                role="t",
                prompt_fn=lambda q, c: [{"role": "user", "content": f"Think: {q}"}],
            ),
            Agent(
                name="S",
                role="s",
                prompt_fn=lambda q, c: [{"role": "user", "content": f"Answer: {q}"}],
                is_final=True,
            ),
        ]
        pipeline = LatentPipeline(
            model, agents, latent_steps=3, max_new_tokens=30,
        )
        results = pipeline.run_batch(["2+2", "3+3"])
        assert len(results) == 2
        for r in results:
            assert isinstance(r.text, str)

    def test_realigner_with_real_model(self, model):
        """The realigner should project and norm-match on real hidden dims."""
        hidden_dim = model.model.config.hidden_size
        hidden = torch.randn(2, hidden_dim, dtype=torch.float32)
        out = model.realigner.apply(hidden)
        assert out.shape == hidden.shape

    def test_pipeline_keep_only_latent(self, model):
        set_seed(42)
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(
            model, agents, latent_steps=3, max_new_tokens=30, keep_only_latent=True,
        )
        result = pipeline.run("test")
        assert isinstance(result.text, str)


# ===================================================================
# Tests: Per-Agent Latent Steps
# ===================================================================


class TestPerAgentLatentSteps:
    """Tests for per-agent latent_steps and convergence_threshold overrides."""

    def test_agent_latent_steps_default_none(self):
        agent = Agent(name="A", role="a",
                      prompt_fn=lambda q, c: [{"role": "user", "content": q}])
        assert agent.latent_steps is None
        assert agent.convergence_threshold is None

    def test_agent_latent_steps_override(self):
        agent = Agent(name="A", role="a",
                      prompt_fn=lambda q, c: [{"role": "user", "content": q}],
                      latent_steps=5, convergence_threshold=0.01)
        assert agent.latent_steps == 5
        assert agent.convergence_threshold == 0.01

    def test_pipeline_uses_agent_override(self):
        """When agent has latent_steps set, pipeline should use it."""
        model = MagicMock(spec=LatentModel)

        def fake_prepare(batch_messages, add_generation_prompt=True):
            bs = len(batch_messages)
            return [str(m) for m in batch_messages], torch.ones(bs, 5, dtype=torch.long), torch.ones(bs, 5, dtype=torch.long)

        model.prepare_chat_batch.side_effect = fake_prepare
        model.generate_latent_batch.return_value = (_make_fake_kv_cache(), 7)
        model.generate_text_batch.return_value = (["Answer_0"], None)

        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}],
                  latent_steps=7),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, latent_steps=20)
        result = pipeline.run("test")

        call_kwargs = model.generate_latent_batch.call_args
        assert call_kwargs[1]["latent_steps"] == 7  # agent override, not 20

    def test_pipeline_falls_back_to_global(self):
        """When agent has no override, pipeline global latent_steps is used."""
        model = MagicMock(spec=LatentModel)

        def fake_prepare(batch_messages, add_generation_prompt=True):
            bs = len(batch_messages)
            return [str(m) for m in batch_messages], torch.ones(bs, 5, dtype=torch.long), torch.ones(bs, 5, dtype=torch.long)

        model.prepare_chat_batch.side_effect = fake_prepare
        model.generate_latent_batch.return_value = (_make_fake_kv_cache(), 15)
        model.generate_text_batch.return_value = (["Answer_0"], None)

        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, latent_steps=15)
        result = pipeline.run("test")

        call_kwargs = model.generate_latent_batch.call_args
        assert call_kwargs[1]["latent_steps"] == 15


# ===================================================================
# Tests: Convergence
# ===================================================================


class TestConvergence:
    """Tests for convergence threshold passing and trace recording."""

    def _make_simple_mock(self):
        model = MagicMock(spec=LatentModel)

        def fake_prepare(batch_messages, add_generation_prompt=True):
            bs = len(batch_messages)
            return [str(m) for m in batch_messages], torch.ones(bs, 5, dtype=torch.long), torch.ones(bs, 5, dtype=torch.long)

        model.prepare_chat_batch.side_effect = fake_prepare
        model.generate_latent_batch.return_value = (_make_fake_kv_cache(), 8)
        model.generate_text_batch.return_value = (["Answer_0"], None)
        return model

    def test_pipeline_passes_convergence_threshold(self):
        model = self._make_simple_mock()
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, convergence_threshold=0.05)
        pipeline.run("test")

        call_kwargs = model.generate_latent_batch.call_args
        assert call_kwargs[1]["convergence_threshold"] == 0.05

    def test_agent_threshold_overrides_pipeline(self):
        model = self._make_simple_mock()
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}],
                  convergence_threshold=0.1),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, convergence_threshold=0.01)
        pipeline.run("test")

        call_kwargs = model.generate_latent_batch.call_args
        assert call_kwargs[1]["convergence_threshold"] == 0.1  # agent wins

    def test_trace_records_actual_and_configured_steps(self):
        model = self._make_simple_mock()
        # Mock returns actual_steps=8 regardless
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}],
                  latent_steps=20),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, latent_steps=50)
        result = pipeline.run("test")

        trace = result.agent_traces[0]
        assert trace["latent_steps"] == 8  # actual steps from mock
        assert trace["latent_steps_configured"] == 20  # agent override


# ===================================================================
# Tests: Self-Consistency Voting
# ===================================================================


class TestSelfConsistencyVoting:
    """Tests for n_samples voting in the pipeline."""

    def _make_voting_mock(self, answers=None):
        model = MagicMock(spec=LatentModel)

        def fake_prepare(batch_messages, add_generation_prompt=True):
            bs = len(batch_messages)
            return [str(m) for m in batch_messages], torch.ones(bs, 5, dtype=torch.long), torch.ones(bs, 5, dtype=torch.long)

        model.prepare_chat_batch.side_effect = fake_prepare
        model.generate_latent_batch.return_value = (_make_fake_kv_cache(), 10)

        call_count = [0]
        def fake_text(input_ids, attention_mask, max_new_tokens=256,
                      temperature=0.7, top_p=0.95, past_key_values=None):
            bs = input_ids.shape[0]
            if answers:
                texts = [answers[call_count[0] % len(answers)]] * bs
            else:
                texts = [f"Answer_{call_count[0]}_{i}" for i in range(bs)]
            call_count[0] += 1
            return texts, None

        model.generate_text_batch.side_effect = fake_text
        return model

    def test_n_samples_one_unchanged(self):
        """n_samples=1 should behave identically to before (no candidates key)."""
        model = self._make_voting_mock()
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, n_samples=1)
        result = pipeline.run("test")

        assert "candidates" not in result.agent_traces[-1]
        assert model.generate_text_batch.call_count == 1

    def test_n_samples_multiple_calls(self):
        """n_samples=3 should call generate_text_batch 3 times."""
        model = self._make_voting_mock()
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, n_samples=3)
        result = pipeline.run("test")

        assert model.generate_text_batch.call_count == 3

    def test_candidates_stored_in_trace(self):
        """With n_samples=3, trace should have candidates list."""
        model = self._make_voting_mock(answers=["#### 42", "#### 42", "#### 7"])
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        pipeline = LatentPipeline(model, agents, n_samples=3)
        result = pipeline.run("test")

        trace = result.agent_traces[-1]
        assert "candidates" in trace
        assert len(trace["candidates"]) == 3
        assert trace["n_samples"] == 3

    def test_custom_vote_fn(self):
        """A custom vote_fn should be used instead of _majority_vote."""
        model = self._make_voting_mock(answers=["first", "second", "third"])
        agents = [
            Agent(name="T", role="t",
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
            Agent(name="S", role="s", is_final=True,
                  prompt_fn=lambda q, c: [{"role": "user", "content": q}]),
        ]
        # Custom vote: always pick last candidate
        pipeline = LatentPipeline(model, agents, n_samples=3,
                                  vote_fn=lambda candidates: candidates[-1])
        result = pipeline.run("test")

        assert result.text == "third"


# ===================================================================
# Tests: Majority Vote
# ===================================================================


class TestMajorityVote:
    """Tests for _extract_answer and _majority_vote."""

    def test_boxed_extraction(self):
        assert _extract_answer("The answer is \\boxed{42}") == "42"

    def test_hash_extraction(self):
        assert _extract_answer("Working... #### 5") == "5"

    def test_answer_is_extraction(self):
        assert _extract_answer("Therefore the answer is 90.\nDone") == "90"

    def test_final_extraction(self):
        assert _extract_answer("FINAL ANSWER: blue") == "blue"

    def test_fallback_last_line(self):
        assert _extract_answer("line1\nline2\nline3") == "line3"

    def test_majority_vote_boxed(self):
        candidates = ["blah \\boxed{42}", "stuff \\boxed{42}", "thing \\boxed{7}"]
        result = _majority_vote(candidates)
        assert "42" in result

    def test_majority_vote_hash(self):
        candidates = ["#### 5", "#### 5", "#### 3"]
        result = _majority_vote(candidates)
        assert "5" in result

    def test_majority_vote_all_different(self):
        candidates = ["alpha", "beta", "gamma"]
        result = _majority_vote(candidates)
        # Returns first candidate when all different (Counter picks first seen)
        assert result in candidates

    def test_majority_vote_single(self):
        assert _majority_vote(["only one"]) == "only one"
