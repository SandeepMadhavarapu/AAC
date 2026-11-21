"""Simplified per-frame psychoacoustic model: band energies, spreading across bands, and
an allowed-noise threshold per band. Every frame is independent, so it batches fully.
Bands are a Bark-like partition, not the official AAC scale-factor-band tables."""
import torch

def band_edges(m=1024, n_bands=49):
    # roughly logarithmic band widths, narrow at low frequencies like the ear
    e = torch.logspace(0, torch.log10(torch.tensor(float(m))), n_bands + 1).round().long()
    e = torch.unique(torch.cat([torch.tensor([0]), e.clamp(max=m)]))
    return e


def band_ids(edges, m=1024):
    ids = torch.zeros(m, dtype=torch.long)
    for b in range(len(edges) - 1):
        ids[edges[b]:edges[b + 1]] = b
    return ids


def thresholds(coeffs, edges, smr_db=18.0):
    """coeffs [..., M] -> allowed noise energy per band [..., n_bands]."""
    nb = len(edges) - 1
    ids = band_ids(edges, coeffs.shape[-1]).to(coeffs.device)
    energy = torch.zeros(*coeffs.shape[:-1], nb, device=coeffs.device)
    energy.index_add_(-1, ids, coeffs ** 2)        # scatter-add: one pass over memory
    # spreading: each band partly masks its neighbours (a 3-tap conv across bands)
    k = torch.tensor([0.15, 1.0, 0.15], device=coeffs.device).view(1, 1, 3)
    flat = energy.reshape(-1, 1, nb)
    spread = torch.nn.functional.conv1d(flat, k, padding=1).reshape(energy.shape)
    return spread * 10 ** (-smr_db / 10), energy
