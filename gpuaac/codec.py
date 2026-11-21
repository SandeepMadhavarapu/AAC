"""End-to-end: audio -> MDCT -> thresholds -> trial quantization -> reservoir -> bitstream,
and back. Global gain per frame only (real AAC also has per-band scalefactors)."""
import torch
from . import mdct as T, psycho, quant, bitpack


def encode(x, bitrate=128_000, sr=48_000, gains=torch.arange(-40, 81, 2)):
    """x [C, T] float -> dict with bitstream and side info (one stream, batched over channels)."""
    frames = T.frame(x.unsqueeze(0))[0]                         # [C, F, N]
    coeffs = T.mdct(frames)                                     # [C, F, M]
    C, F, M = coeffs.shape
    edges = psycho.band_edges(M)
    thr, _ = psycho.thresholds(coeffs, edges)                   # computed in parallel
    flat = coeffs.reshape(C * F, M)
    bits, noise = quant.trial_quantize(flat, gains.float())
    frame_budget = int(bitrate * M / sr / C)                    # bits per channel-frame
    choice = quant.reservoir_pick(bits, noise, frame_budget, reservoir_max=6144)
    g = gains[choice].float().view(-1, 1)
    q = quant.quantize(flat, g).long()
    codes, lengths = bitpack.encode_symbols(q)
    data, nbits = bitpack.pack_parallel(codes, lengths)
    return dict(data=data, nbits=nbits, gains=gains[choice], shape=(C, F, M),
                T=x.shape[-1], thresholds=thr)


def decode(enc):
    C, F, M = enc["shape"]
    q = torch.tensor(bitpack.decode(enc["data"], C * F * M), dtype=torch.float32).view(C * F, M)
    coeffs = quant.dequantize(q, enc["gains"].float().view(-1, 1)).view(1, C, F, M)
    return T.overlap_add(T.imdct(coeffs), enc["T"])[0]


def snr_db(ref, out):
    return float(10 * torch.log10((ref ** 2).sum() / ((ref - out) ** 2).sum()))
