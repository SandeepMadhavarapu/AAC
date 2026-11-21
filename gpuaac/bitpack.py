"""Variable-length codes written into one bitstream.
Sequential reference: one write after another.
Parallel version: codeword lengths -> exclusive prefix sum -> every bit gets a unique
position, so all bits are written in one scatter with no conflicts."""
import torch


def encode_symbols(q):
    """q (int tensor, any shape) -> (codes, lengths) for Exp-Golomb(|q|) + sign bit."""
    q = q.reshape(-1).long()
    a = q.abs()
    v = a + 1
    nbits = torch.floor(torch.log2(v.double())).long() + 1     # bits in binary(v)
    eg_len = 2 * nbits - 1                                      # zeros prefix + binary(v)
    sign = (q < 0).long()
    has_sign = (a > 0).long()
    codes = (v << has_sign) | (sign & has_sign)
    lengths = eg_len + has_sign
    return codes, lengths


def pack_parallel(codes, lengths):
    total = int(lengths.sum())
    offsets = torch.cumsum(lengths, 0) - lengths                # exclusive scan
    L = int(lengths.max())
    j = torch.arange(L, device=codes.device)
    valid = j.unsqueeze(0) < lengths.unsqueeze(1)               # [S, L]
    shift = (lengths.unsqueeze(1) - 1 - j).clamp(min=0)
    bit = ((codes.unsqueeze(1) >> shift) & 1).to(torch.uint8)
    pos = offsets.unsqueeze(1) + j
    stream = torch.zeros(total, dtype=torch.uint8, device=codes.device)
    stream[pos[valid]] = bit[valid]                             # unique positions: no races
    return _to_bytes(stream), total


def pack_sequential(codes, lengths):
    out, acc, nacc, total = bytearray(), 0, 0, 0
    for c, l in zip(codes.tolist(), lengths.tolist()):
        acc = (acc << l) | c; nacc += l; total += l
        while nacc >= 8:
            nacc -= 8; out.append((acc >> nacc) & 0xFF)
        acc &= (1 << nacc) - 1
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out), total


def _to_bytes(bits):
    pad = (-bits.numel()) % 8
    bits = torch.nn.functional.pad(bits, (0, pad)).view(-1, 8).long()
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=bits.device)
    return bytes((bits * weights).sum(1).to(torch.uint8).cpu().tolist())


def decode(data, n_symbols):
    """Sequential decoder (a prefix code must be read in order inside one stream)."""
    bits = "".join(f"{b:08b}" for b in data)
    i, out = 0, []
    for _ in range(n_symbols):
        z = 0
        while bits[i] == "0":
            z += 1; i += 1
        v = int(bits[i:i + z + 1], 2); i += z + 1
        a = v - 1
        if a:
            s = bits[i]; i += 1
            out.append(-a if s == "1" else a)
        else:
            out.append(0)
    return out
