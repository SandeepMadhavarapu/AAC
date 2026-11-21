"""AAC's non-uniform quantizer, run for many candidate global gains at once, plus the one
step that is genuinely sequential: the bit reservoir that carries unused bits forward."""
import torch

MAGIC = 0.4054  # AAC rounding offset


def quantize(x, gain):
    """q = sign(x) * floor(|x / 2^(gain/4)|^0.75 + 0.4054). gain broadcasts."""
    s = 2.0 ** (gain / 4.0)
    return torch.sign(x) * torch.floor((x.abs() / s) ** 0.75 + MAGIC)


def dequantize(q, gain):
    return torch.sign(q) * q.abs() ** (4.0 / 3.0) * 2.0 ** (gain / 4.0)


def code_lengths(q):
    """Bits for each value under the prefix code used by bitpack (Exp-Golomb on |q|,
    plus a sign bit for non-zero values)."""
    a = q.abs().long()
    eg = 2 * torch.floor(torch.log2((a + 1).double())).long() + 1
    return eg + (a > 0).long()


def trial_quantize(coeffs, gains, chunk=256):
    """coeffs [F, M], gains [G] -> bits [G, F], noise [G, F]. Every (gain, frame) pair is
    independent, so each chunk of frames is one batched tensor op. Chunking bounds memory
    at G * chunk * M values instead of G * F * M."""
    if coeffs.shape[0] > chunk:
        parts = [trial_quantize(coeffs[i:i + chunk], gains, chunk)
                 for i in range(0, coeffs.shape[0], chunk)]
        return torch.cat([b for b, _ in parts], 1), torch.cat([n for _, n in parts], 1)
    g = gains.view(-1, 1, 1).to(coeffs.dtype)
    q = quantize(coeffs.unsqueeze(0), g)                       # [G, F, M]
    noise = ((coeffs.unsqueeze(0) - dequantize(q, g)) ** 2).sum(-1)
    bits = code_lengths(q).sum(-1)
    return bits, noise


def reservoir_pick(bits, noise, frame_budget, reservoir_max):
    """Sequential pass over frames. For each frame, pick the lowest-noise candidate that fits
    in this frame's budget plus whatever the reservoir has saved. Cheap: it only reads the
    precomputed [G, F] tables, it never re-quantizes."""
    G, F = bits.shape
    bits_c, noise_c = bits.cpu(), noise.cpu()
    choice = torch.empty(F, dtype=torch.long)
    reservoir = 0
    for f in range(F):
        allowed = frame_budget + reservoir
        ok = bits_c[:, f] <= allowed
        if ok.any():
            n = noise_c[:, f].clone(); n[~ok] = float("inf")
            c = int(torch.argmin(n))
        else:
            c = int(torch.argmin(bits_c[:, f]))   # coarsest option; overshoot is reported
        choice[f] = c
        reservoir = min(reservoir_max, max(0, allowed - int(bits_c[c, f])))
    return choice
