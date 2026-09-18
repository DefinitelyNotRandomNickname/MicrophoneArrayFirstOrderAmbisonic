"""Selective state-space scan (S6) used by the Mamba mixers.

For every channel ``d`` and state ``n`` the recurrence is

    h_t[d, n] = exp(delta_t[d] * A[d, n]) * h_{t-1}[d, n] + delta_t[d] * B_t[n] * u_t[d]
    y_t[d] = sum_n C_t[n] * h_t[d, n] + D[d] * u_t[d]

``selective_scan_reference`` is a plain loop over time and is kept for tests.
``selective_scan`` runs a chunked parallel scan inside a custom autograd function
that stores only the chunk boundary states and recomputes the hidden states in
the backward pass, so memory stays O(B * L * D) instead of O(B * L * D * N).
"""

import math

import torch

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _cuda_selective_scan,
    )
except (ImportError, OSError):
    # CPU-only fallback
    _cuda_selective_scan = None

SCAN_BACKENDS = ("auto", "torch", "cuda")


def cuda_scan_available():
    return _cuda_selective_scan is not None


def selective_scan_reference(u, delta, A, B, C, D):
    """Sequential reference scan.

    u, delta: [B, L, D]; A: [D, N]; B, C: [B, L, N]; D: [D]. Returns [B, L, D].
    """
    batch_size, length, _ = u.shape
    state = u.new_zeros(batch_size, A.size(0), A.size(1))
    outputs = []
    for t in range(length):
        decay = torch.exp(delta[:, t, :, None] * A)
        source = delta[:, t, :, None] * u[:, t, :, None] * B[:, t, None, :]
        state = decay * state + source
        outputs.append((state * C[:, t, None, :]).sum(-1))
    return torch.stack(outputs, dim=1) + u * D


def _linear_scan(decay, source, state):
    """Solve h_t = decay_t * h_{t-1} + source_t along dim 1 with h_{-1} = state.

    Hillis-Steele scan: ceil(log2(L)) steps of whole-tensor operations.
    """
    first = (source[:, 0] + decay[:, 0] * state).unsqueeze(1)
    source = torch.cat([first, source[:, 1:]], dim=1)

    length = source.size(1)
    shift = 1
    while shift < length:
        source = torch.cat(
            [
                source[:, :shift],
                source[:, shift:] + decay[:, shift:] * source[:, :-shift],
            ],
            dim=1,
        )
        decay = torch.cat(
            [decay[:, :shift], decay[:, shift:] * decay[:, :-shift]], dim=1
        )
        shift *= 2
    return source


def _chunk_bounds(length, chunk_size):
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def _default_chunk_size(length):
    return max(1, math.ceil(math.sqrt(length)))


class _SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, chunk_size):
        length = u.size(1)
        state = u.new_zeros(u.size(0), A.size(0), A.size(1))

        boundaries = []
        outputs = []
        for start, end in _chunk_bounds(length, chunk_size):
            delta_chunk = delta[:, start:end]
            decay = torch.exp(delta_chunk.unsqueeze(-1) * A)
            source = (delta_chunk * u[:, start:end]).unsqueeze(-1) * B[
                :, start:end
            ].unsqueeze(2)

            boundaries.append(state)
            states = _linear_scan(decay, source, state)
            outputs.append(torch.einsum("bldn,bln->bld", states, C[:, start:end]))
            state = states[:, -1]

        ctx.save_for_backward(u, delta, A, B, C, D, torch.stack(boundaries))
        ctx.chunk_size = chunk_size
        return torch.cat(outputs, dim=1) + u * D

    @staticmethod
    def backward(ctx, grad_output):
        u, delta, A, B, C, D, boundaries = ctx.saved_tensors
        length = u.size(1)
        bounds = _chunk_bounds(length, ctx.chunk_size)

        grad_u = torch.empty_like(u)
        grad_delta = torch.empty_like(delta)
        grad_A = torch.zeros_like(A)
        grad_B = torch.empty_like(B)
        grad_C = torch.empty_like(C)
        grad_D = torch.einsum("bld,bld->d", grad_output, u)

        # Gradient wrt the hidden state entering the chunk that follows the
        # current one; the last chunk has no successor.
        carry = torch.zeros_like(boundaries[0])
        for index in reversed(range(len(bounds))):
            start, end = bounds[index]
            u_chunk = u[:, start:end]
            delta_chunk = delta[:, start:end]
            B_chunk = B[:, start:end]
            C_chunk = C[:, start:end]
            grad_chunk = grad_output[:, start:end]

            # Recompute the hidden states of this chunk from its boundary state.
            decay = torch.exp(delta_chunk.unsqueeze(-1) * A)
            source = (delta_chunk * u_chunk).unsqueeze(-1) * B_chunk.unsqueeze(2)
            states = _linear_scan(decay, source, boundaries[index])
            previous_states = torch.cat(
                [boundaries[index].unsqueeze(1), states[:, :-1]], dim=1
            )

            # dh_t = C_t * dy_t + decay_{t+1} * dh_{t+1} is a linear recurrence
            # backwards in time, so the same scan runs on flipped sequences.
            if end < length:
                next_decay = torch.exp(delta[:, end].unsqueeze(-1) * A).unsqueeze(1)
            else:
                next_decay = torch.zeros_like(decay[:, :1])
            next_decays = torch.cat([decay[:, 1:], next_decay], dim=1)
            state_source = grad_chunk.unsqueeze(-1) * C_chunk.unsqueeze(2)
            grad_states = _linear_scan(
                next_decays.flip(1), state_source.flip(1), carry
            ).flip(1)
            carry = grad_states[:, 0]

            grad_exponent = grad_states * previous_states * decay
            grad_source = torch.einsum("bldn,bln->bld", grad_states, B_chunk)

            grad_delta[:, start:end] = (grad_exponent * A).sum(-1) + (
                grad_source * u_chunk
            )
            grad_u[:, start:end] = grad_source * delta_chunk + grad_chunk * D
            grad_A += torch.einsum("bldn,bld->dn", grad_exponent, delta_chunk)
            grad_B[:, start:end] = torch.einsum(
                "bldn,bld->bln", grad_states, delta_chunk * u_chunk
            )
            grad_C[:, start:end] = torch.einsum("bldn,bld->bln", states, grad_chunk)

        return grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D, None


def selective_scan(u, delta, A, B, C, D, backend="auto", chunk_size=None):
    """Selective scan with the same shapes as ``selective_scan_reference``.

    backend: "auto" uses the mamba_ssm CUDA kernel for GPU tensors when it is
    installed and the chunked PyTorch scan otherwise; "torch" and "cuda" force
    one of them. chunk_size only affects the PyTorch scan and defaults to
    ceil(sqrt(L)), balancing saved boundary states against recompute buffers.
    """
    if backend not in SCAN_BACKENDS:
        raise ValueError(f"Unsupported scan backend: {backend}")
    if u.dim() != 3:
        raise ValueError("selective_scan expects u with shape [B, L, D]")

    if backend == "cuda" and not cuda_scan_available():
        raise RuntimeError("scan backend 'cuda' requires the mamba_ssm package")
    if backend == "cuda" or (backend == "auto" and cuda_scan_available() and u.is_cuda):
        output = _cuda_selective_scan(
            u.transpose(1, 2),
            delta.transpose(1, 2),
            A,
            B.transpose(1, 2),
            C.transpose(1, 2),
            D=D,
        )
        return output.transpose(1, 2)

    if chunk_size is None:
        chunk_size = _default_chunk_size(u.size(1))
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    dtype = torch.promote_types(u.dtype, torch.float32)
    output = _SelectiveScan.apply(
        u.to(dtype),
        delta.to(dtype),
        A.to(dtype),
        B.to(dtype),
        C.to(dtype),
        D.to(dtype),
        chunk_size,
    )
    return output.to(u.dtype)
