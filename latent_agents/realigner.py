"""Training-free latent-space realignment.

Projects transformer output hidden states into the input-embedding space using
a closed-form least-squares solution derived from the model's own embedding
weights.  No additional training is required.
"""

import torch
import torch.nn as nn
from typing import Optional


class LatentRealigner:
    """Converts output hidden states into input-embedding-compatible vectors.

    The core math:
        R = (W_out^T W_out + lambda I)^{-1} W_out^T W_in

    where ``W_in`` is the input embedding weight and ``W_out`` is the output
    (lm_head) weight.  After projection, vectors are rescaled to match the
    average norm of input embeddings so the model receives activations in the
    expected magnitude range.

    Parameters
    ----------
    model : nn.Module
        Any ``AutoModelForCausalLM``-compatible model that exposes
        ``get_input_embeddings()`` and ``get_output_embeddings()``.
    device : torch.device
        Device on which to store the realignment matrix.
    enabled : bool
        When ``False`` the realignment matrix is replaced with an identity
        matrix, but norm-matching is still applied.  This is useful as an
        ablation baseline.
    reg : float
        Tikhonov regularisation constant to stabilise the matrix inverse.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        enabled: bool = True,
        reg: float = 1e-5,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self._matrix: Optional[torch.Tensor] = None
        self._target_norm: Optional[torch.Tensor] = None
        self._build(model, reg)

    def _build(self, model: nn.Module, reg: float) -> None:
        """Build the realignment matrix from the model's embedding weights."""
        input_embeds = model.get_input_embeddings()
        output_embeds = model.get_output_embeddings()
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: "
                "input/output embedding weights are not accessible on this model."
            )

        W_in = input_embeds.weight.detach().to(device=self.device, dtype=torch.float32)
        W_out = output_embeds.weight.detach().to(device=self.device, dtype=torch.float32)

        # Closed-form least-squares: R = (W_out^T W_out + reg*I)^-1 W_out^T W_in
        gram = W_out.T @ W_out
        gram += reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = W_out.T @ W_in
        realign_matrix = torch.linalg.solve(gram, rhs)

        target_norm = W_in.norm(dim=1).mean().detach()

        if not self.enabled:
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )

        self._matrix = realign_matrix
        self._target_norm = target_norm.to(device=self.device, dtype=realign_matrix.dtype)

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project a hidden state into input-embedding space and norm-match.

        Parameters
        ----------
        hidden : torch.Tensor
            Hidden state(s) of shape ``[B, D]`` or ``[B, L, D]``.

        Returns
        -------
        torch.Tensor
            Realigned embedding(s), same shape as *hidden*.
        """
        assert self._matrix is not None, "Realigner not built."

        orig_dtype = hidden.dtype
        h = hidden.to(torch.float32)
        aligned = h @ self._matrix

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (self._target_norm / aligned_norm)

        return aligned.to(orig_dtype)
