"""Batched MDCT / IMDCT via FFT. Shapes: signal [B, C, T] -> coeffs [B, C, F, M], M = 1024."""
import math
import torch

M = 1024          # coefficients per frame (AAC long block)
N = 2 * M         # window length, 50% overlap


def sine_window(n=N, device=None, dtype=torch.float32):
    i = torch.arange(n, device=device, dtype=dtype)
    return torch.sin(math.pi * (i + 0.5) / n)


def frame(x):
    """[B, C, T] -> [B, C, F, N] overlapping frames (hop M). Pads M zeros on both ends so
    every input sample is covered by two windows (needed for perfect reconstruction)."""
    T = x.shape[-1]
    pad_end = M + (-T) % M
    x = torch.nn.functional.pad(x, (M, pad_end))
    return x.unfold(-1, N, M)  # a view: no copy, contiguous along the window axis


def _twiddles(device, dtype):
    n0 = (N / 2 + 1) / 2
    n = torch.arange(N, device=device, dtype=dtype)
    k = torch.arange(M, device=device, dtype=dtype)
    pre = torch.exp(-1j * math.pi * n / N)                  # e^{-i pi n / N}
    post = torch.exp(-1j * 2 * math.pi * n0 * (k + 0.5) / N)
    return pre, post


def mdct(frames, window=None):
    """frames [..., N] -> coeffs [..., M]. One batched FFT over every frame/channel/stream."""
    w = sine_window(device=frames.device) if window is None else window
    pre, post = _twiddles(frames.device, torch.float64)
    xw = (frames * w).to(torch.complex64)
    spec = torch.fft.fft(xw * pre.to(torch.complex64), dim=-1)[..., :M]
    return (spec * post.to(torch.complex64)).real


def imdct(coeffs, window=None):
    """coeffs [..., M] -> windowed time frames [..., N], scaled for TDAC overlap-add."""
    w = sine_window(device=coeffs.device) if window is None else window
    n0 = (N / 2 + 1) / 2
    k = torch.arange(M, device=coeffs.device, dtype=torch.float64)
    n = torch.arange(N, device=coeffs.device, dtype=torch.float64)
    pre = torch.exp(1j * 2 * math.pi * n0 * k / N).to(torch.complex64)
    post = torch.exp(1j * math.pi * (n + n0) / N).to(torch.complex64)
    Xp = torch.zeros(*coeffs.shape[:-1], N, dtype=torch.complex64, device=coeffs.device)
    Xp[..., :M] = coeffs.to(torch.complex64) * pre
    y = (torch.fft.ifft(Xp, dim=-1) * N * post).real
    return y * w * (2.0 / M)


def overlap_add(frames_out, T):
    """[B, C, F, N] -> [B, C, T]. Elementwise add of each frame's second half to the next
    frame's first half: fully parallel once every IMDCT is done."""
    first, second = frames_out[..., :M], frames_out[..., M:]
    halves = first[..., 1:, :] + second[..., :-1, :]          # [B, C, F-1, M]
    out = halves.reshape(*halves.shape[:-2], -1)
    return out[..., :T]


def mdct_direct(frames, window=None):
    """O(N*M) reference straight from the definition, used only in tests."""
    w = sine_window(device=frames.device, dtype=torch.float64) if window is None else window
    n0 = (N / 2 + 1) / 2
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(M, dtype=torch.float64)
    basis = torch.cos(2 * math.pi / N * (n[:, None] + n0) * (k[None, :] + 0.5))
    return (frames.double() * w) @ basis
