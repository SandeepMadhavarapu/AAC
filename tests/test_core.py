import math, torch, pytest
from gpuaac import mdct as T, quant, bitpack, codec, psycho

torch.manual_seed(0)


def test_fft_mdct_matches_definition():
    fr = torch.randn(3, T.N)
    fast = T.mdct(fr).double()
    ref = T.mdct_direct(fr)
    assert torch.allclose(fast, ref, atol=1e-3 * ref.abs().max())


def test_perfect_reconstruction_tdac():
    x = torch.randn(2, 2, 10 * T.M + 37)           # [B, C, T], length not a multiple of M
    y = T.overlap_add(T.imdct(T.mdct(T.frame(x))), x.shape[-1])
    assert (x - y).abs().max() < 1e-4


def test_batched_equals_per_frame():
    fr = torch.randn(4, 5, T.N)
    batched = T.mdct(fr)
    single = torch.stack([T.mdct(fr[i, j]) for i in range(4) for j in range(5)]).view(4, 5, T.M)
    assert torch.allclose(batched, single, atol=1e-4)


@pytest.mark.parametrize("n", [1, 7, 5000])
def test_parallel_pack_equals_sequential(n):
    q = torch.randint(-300, 300, (n,))
    q[torch.rand(n) < 0.4] = 0
    codes, lengths = bitpack.encode_symbols(q)
    assert bitpack.pack_parallel(codes, lengths) == bitpack.pack_sequential(codes, lengths)


def test_bitstream_roundtrip():
    q = torch.randint(-50, 50, (3000,))
    codes, lengths = bitpack.encode_symbols(q)
    data, nbits = bitpack.pack_parallel(codes, lengths)
    assert bitpack.decode(data, q.numel()) == q.tolist()
    assert nbits == int(quant.code_lengths(q).sum())          # bit estimate is exact


def test_finer_gain_means_more_bits_less_noise():
    c = torch.randn(8, T.M) * 100
    bits, noise = quant.trial_quantize(c, torch.tensor([0., 20., 40.]))
    assert (bits[0] > bits[1]).all() and (bits[1] > bits[2]).all()
    assert (noise[0] < noise[1]).all() and (noise[1] < noise[2]).all()


def test_reservoir_respects_budget():
    c = torch.randn(40, T.M) * 1000
    c[10:12] *= 20                                             # a loud transient
    gains = torch.arange(-20, 81, 2).float()
    bits, noise = quant.trial_quantize(c, gains)
    budget = 2000
    ch = quant.reservoir_pick(bits, noise, budget, reservoir_max=6000)
    used = bits[ch, torch.arange(40)]
    assert int(used.sum()) <= budget * 40                      # never overspends on average
    assert used.max() > budget                                 # reservoir lets the transient borrow


def test_thresholds_shape_and_positive():
    c = torch.randn(2, 3, T.M)
    edges = psycho.band_edges()
    thr, energy = psycho.thresholds(c, edges)
    assert thr.shape == (2, 3, len(edges) - 1) and (thr >= 0).all()
    assert torch.allclose(energy.sum(-1), (c ** 2).sum(-1), rtol=1e-4)


def test_end_to_end_quality_improves_with_bitrate():
    sr = 48_000
    t = torch.arange(sr // 2) / sr
    x = (0.5 * torch.sin(2 * math.pi * 440 * t) + 0.2 * torch.sin(2 * math.pi * 3000 * t)
         + 0.01 * torch.randn_like(t)).unsqueeze(0) * 32767
    low = codec.encode(x, bitrate=48_000)
    high = codec.encode(x, bitrate=192_000)
    s_low, s_high = codec.snr_db(x, codec.decode(low)), codec.snr_db(x, codec.decode(high))
    assert s_high > s_low > 10
    assert high["nbits"] > low["nbits"]
