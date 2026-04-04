"""Unit tests for TurboQuant KV-cache quantization.

All tests run on CPU with small synthetic tensors — no model download required.
"""

from __future__ import annotations

import torch
import pytest

from latent_agents.turboquant import (
    TurboQuant,
    TurboQuantConfig,
    CompressedKV,
    CompressedKVLayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_cache(n_layers=2, batch=2, heads=4, seq=10, head_dim=64, dtype=torch.float32):
    """Build a fake legacy-tuple KV cache filled with random floats."""
    layers = []
    for _ in range(n_layers):
        k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
        layers.append((k, v))
    return tuple(layers)


def _make_tq(bits=3, qjl_dim=64):
    return TurboQuant(
        TurboQuantConfig(bits=bits, qjl_dim=qjl_dim, random_seed=0),
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# PolarQuant round-trip fidelity
# ---------------------------------------------------------------------------

class TestPolarQuantRoundTrip:
    def test_cosine_similarity_with_qjl(self):
        """PolarQuant + QJL should improve cosine similarity over PolarQuant alone.

        Note: with random N(0,1) test tensors and 3-bit quantization, the unit
        vector elements (~1/√D) are small relative to the quantization step (~0.29),
        so absolute cosine similarity is moderate (~0.86).  With real LLM activations
        (lower variance, more structured), fidelity is higher.  The key property we
        test here is that QJL improves on PolarQuant-only.
        """
        tq_no_qjl = _make_tq(bits=3, qjl_dim=0)
        tq_with_qjl = _make_tq(bits=3, qjl_dim=64)
        kv = _make_kv_cache(n_layers=1, batch=1, heads=2, seq=8, head_dim=64)

        orig_k = kv[0][0].reshape(-1, 64)

        recon_no_qjl = tq_no_qjl.decompress_kv(tq_no_qjl.compress_kv(kv))[0][0].reshape(-1, 64)
        recon_with_qjl = tq_with_qjl.decompress_kv(tq_with_qjl.compress_kv(kv))[0][0].reshape(-1, 64)

        cos_no_qjl = torch.nn.functional.cosine_similarity(orig_k, recon_no_qjl, dim=-1).mean()
        cos_with_qjl = torch.nn.functional.cosine_similarity(orig_k, recon_with_qjl, dim=-1).mean()

        assert cos_with_qjl > cos_no_qjl, (
            f"QJL should improve cosine sim: no_qjl={cos_no_qjl:.4f}, with_qjl={cos_with_qjl:.4f}"
        )
        # Sanity bound: even 3-bit should achieve decent similarity
        assert cos_with_qjl > 0.80, f"Mean cosine sim {cos_with_qjl:.4f} < 0.80"

    def test_cosine_similarity_polar_only(self):
        """PolarQuant alone (3-bit) should achieve reasonable cosine similarity (> 0.75)."""
        tq = _make_tq(bits=3, qjl_dim=0)
        kv = _make_kv_cache(n_layers=1, batch=1, heads=2, seq=8, head_dim=64)

        compressed = tq.compress_kv(kv)
        reconstructed = tq.decompress_kv(compressed)

        orig_k = kv[0][0].reshape(-1, 64)
        recon_k = reconstructed[0][0].reshape(-1, 64)

        cos_sim = torch.nn.functional.cosine_similarity(orig_k, recon_k, dim=-1)
        assert cos_sim.mean().item() > 0.75, f"Mean cosine sim {cos_sim.mean():.4f} < 0.75"

    def test_larger_tensor(self):
        """4-bit + QJL on a larger [4, 8, 20, 128] tensor should have reasonable cosine sim."""
        tq = _make_tq(bits=4, qjl_dim=128)
        kv = _make_kv_cache(n_layers=2, batch=4, heads=8, seq=20, head_dim=128)

        compressed = tq.compress_kv(kv)
        reconstructed = tq.decompress_kv(compressed)

        for layer_idx in range(2):
            for tensor_idx in range(2):  # key, value
                orig = kv[layer_idx][tensor_idx].reshape(-1, 128)
                recon = reconstructed[layer_idx][tensor_idx].reshape(-1, 128)
                cos_sim = torch.nn.functional.cosine_similarity(orig, recon, dim=-1)
                assert cos_sim.mean().item() > 0.90, (
                    f"layer={layer_idx}, kv={tensor_idx}: cosine sim {cos_sim.mean():.4f} < 0.90"
                )


# ---------------------------------------------------------------------------
# QJL improves reconstruction
# ---------------------------------------------------------------------------

class TestQJLErrorCorrection:
    def test_qjl_reduces_mse(self):
        """PolarQuant + QJL should have lower MSE than PolarQuant alone."""
        torch.manual_seed(7)
        kv = _make_kv_cache(n_layers=1, batch=2, heads=4, seq=15, head_dim=64)

        tq_no_qjl = _make_tq(bits=3, qjl_dim=0)
        tq_with_qjl = _make_tq(bits=3, qjl_dim=64)

        orig = kv[0][0]

        recon_no_qjl = tq_no_qjl.decompress_kv(tq_no_qjl.compress_kv(kv))[0][0]
        recon_with_qjl = tq_with_qjl.decompress_kv(tq_with_qjl.compress_kv(kv))[0][0]

        mse_no_qjl = (orig - recon_no_qjl).pow(2).mean().item()
        mse_with_qjl = (orig - recon_with_qjl).pow(2).mean().item()

        assert mse_with_qjl < mse_no_qjl, (
            f"QJL should reduce MSE: no_qjl={mse_no_qjl:.6f}, with_qjl={mse_with_qjl:.6f}"
        )

    def test_qjl_matrix_reproducible(self):
        """Same config → same QJL matrix, so compress/decompress always round-trips correctly."""
        tq1 = TurboQuant(TurboQuantConfig(bits=3, qjl_dim=32, random_seed=99),
                         torch.device("cpu"), torch.float32)
        tq2 = TurboQuant(TurboQuantConfig(bits=3, qjl_dim=32, random_seed=99),
                         torch.device("cpu"), torch.float32)

        R1 = tq1._get_qjl_matrix(64)
        R2 = tq2._get_qjl_matrix(64)
        assert torch.allclose(R1, R2), "Same seed must produce identical QJL matrices"

    def test_different_seeds_give_different_matrices(self):
        tq_a = TurboQuant(TurboQuantConfig(random_seed=1), torch.device("cpu"), torch.float32)
        tq_b = TurboQuant(TurboQuantConfig(random_seed=2), torch.device("cpu"), torch.float32)
        R_a = tq_a._get_qjl_matrix(64)
        R_b = tq_b._get_qjl_matrix(64)
        assert not torch.allclose(R_a, R_b)


# ---------------------------------------------------------------------------
# Structure and dtype preservation
# ---------------------------------------------------------------------------

class TestStructurePreservation:
    def test_shape_preserved(self):
        """compress→decompress must return tensors with the original shapes."""
        tq = _make_tq()
        kv = _make_kv_cache(n_layers=4, batch=2, heads=8, seq=12, head_dim=64)

        reconstructed = tq.decompress_kv(tq.compress_kv(kv))

        assert len(reconstructed) == 4
        for layer_idx in range(4):
            assert reconstructed[layer_idx][0].shape == kv[layer_idx][0].shape
            assert reconstructed[layer_idx][1].shape == kv[layer_idx][1].shape

    def test_dtype_preserved(self):
        """Decompressed tensors must have the same dtype as the originals."""
        tq = _make_tq()
        kv = _make_kv_cache(n_layers=2, batch=1, heads=4, seq=8, head_dim=64, dtype=torch.float32)

        reconstructed = tq.decompress_kv(tq.compress_kv(kv))
        for layer in reconstructed:
            assert layer[0].dtype == torch.float32
            assert layer[1].dtype == torch.float32

    def test_none_passthrough(self):
        tq = _make_tq()
        assert tq.compress_kv(None) is None
        assert tq.decompress_kv(None) is None

    def test_compressed_kv_layer_fields(self):
        """CompressedKVLayer must contain correctly typed/shaped tensors."""
        tq = _make_tq(bits=3, qjl_dim=32)
        kv = _make_kv_cache(n_layers=1, batch=1, heads=2, seq=5, head_dim=64)

        compressed = tq.compress_kv(kv)
        assert isinstance(compressed, CompressedKV)
        layer = compressed.layers[0]

        assert layer.k_quant.dtype == torch.int8
        assert layer.k_quant.shape == (1, 2, 5, 64)
        assert layer.k_radius.dtype == torch.float16
        assert layer.k_radius.shape == (1, 2, 5, 1)
        assert layer.k_qjl.dtype == torch.int8
        assert layer.k_qjl.shape == (1, 2, 5, 32)
        assert layer.k_qjl_scale.dtype == torch.float16
        assert layer.k_qjl_scale.shape == (1, 2, 5, 1)


# ---------------------------------------------------------------------------
# enabled=False pass-through
# ---------------------------------------------------------------------------

class TestDisabledPassthrough:
    def test_compress_returns_none_when_disabled(self):
        tq = TurboQuant(
            TurboQuantConfig(enabled=False),
            torch.device("cpu"),
            torch.float32,
        )
        kv = _make_kv_cache()
        assert tq.compress_kv(kv) is None

    def test_decompress_none_returns_none(self):
        tq = TurboQuant(
            TurboQuantConfig(enabled=False),
            torch.device("cpu"),
            torch.float32,
        )
        assert tq.decompress_kv(None) is None


# ---------------------------------------------------------------------------
# Higher bits → lower MSE
# ---------------------------------------------------------------------------

class TestBitwidthEffect:
    def test_4bit_lower_mse_than_3bit(self):
        """Increasing bit-width should monotonically reduce reconstruction error."""
        torch.manual_seed(42)
        kv = _make_kv_cache(n_layers=1, batch=2, heads=4, seq=20, head_dim=64)
        orig = kv[0][0]

        def mse(bits):
            tq = _make_tq(bits=bits, qjl_dim=0)
            recon = tq.decompress_kv(tq.compress_kv(kv))[0][0]
            return (orig - recon).pow(2).mean().item()

        assert mse(4) < mse(3), "4-bit should have lower MSE than 3-bit"


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_ratio_greater_than_one(self):
        """Compressed size should be smaller than original for typical head_dim."""
        tq = _make_tq(bits=3, qjl_dim=64)
        # head_dim=128: original float32 → 4 bytes/elem; compressed int8 (128) + f16 (1) + int8 (64)
        kv = _make_kv_cache(n_layers=32, batch=1, heads=32, seq=60, head_dim=128,
                             dtype=torch.float32)
        ratio = tq.compression_ratio(kv)
        # float32 = 4 bytes/elem; compressed ≈ 128+2+64=194 bytes vs 128*4=512 bytes → ~2.6x
        assert ratio > 1.5, f"Expected ratio > 1.5, got {ratio:.2f}"

    def test_ratio_none_input(self):
        tq = _make_tq()
        assert tq.compression_ratio(None) == 1.0


# ---------------------------------------------------------------------------
# QJL matrix caching
# ---------------------------------------------------------------------------

class TestQJLMatrixCaching:
    def test_matrix_cached(self):
        """Second call for same head_dim must return the cached tensor (same object)."""
        tq = _make_tq()
        R1 = tq._get_qjl_matrix(64)
        R2 = tq._get_qjl_matrix(64)
        assert R1 is R2, "QJL matrix should be cached (same object)"

    def test_different_head_dims_cached_separately(self):
        tq = _make_tq()
        R64 = tq._get_qjl_matrix(64)
        R128 = tq._get_qjl_matrix(128)
        assert R64.shape == (64, 64)
        assert R128.shape == (64, 128)
        assert R64 is not R128
