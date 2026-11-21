"""Benchmark: per-stage time on a batch of streams, and parallel vs sequential bit packing.
Runs on CUDA if available, otherwise CPU. Usage: python bench.py [--streams 64] [--seconds 10]"""
import argparse, math, time, torch
from gpuaac import mdct as T, psycho, quant, bitpack, codec

p = argparse.ArgumentParser()
p.add_argument("--streams", type=int, default=32)
p.add_argument("--seconds", type=float, default=10)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
name = torch.cuda.get_device_name() if dev == "cuda" else f"CPU ({torch.get_num_threads()} threads)"


def timed(fn, reps=5):
    fn(); sync()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync()
    return (time.perf_counter() - t) / reps, out


sr = 48_000
x = torch.randn(a.streams, 2, int(a.seconds * sr), device=dev) * 3000
audio_s = a.streams * a.seconds
print(f"device: {name} | {a.streams} stereo streams x {a.seconds:.0f}s = {audio_s:.0f}s of audio\n")

t_mdct, coeffs = timed(lambda: T.mdct(T.frame(x)))
edges = psycho.band_edges()
t_psy, _ = timed(lambda: psycho.thresholds(coeffs, edges))
flat = coeffs.reshape(-1, T.M)
gains = torch.arange(-40, 81, 4, device=dev).float()
t_trial, (bits, noise) = timed(lambda: quant.trial_quantize(flat, gains), reps=2)
t_res, choice = timed(lambda: quant.reservoir_pick(bits[:, :2000], noise[:, :2000], 1300, 6144), reps=1)

print(f"{'stage':38s}{'time':>10s}{'x realtime':>12s}")
for label, t in [("MDCT (batched FFT)", t_mdct), ("psychoacoustic thresholds", t_psy),
                 (f"trial quantization ({len(gains)} gains)", t_trial)]:
    print(f"{label:38s}{t*1e3:8.1f}ms{audio_s / t:11.0f}x")
print(f"{'reservoir pass (2,000 frames, sequential)':38s}{t_res*1e3:8.1f}ms")

q = quant.quantize(flat[:2000], 20.0).long()
codes, lengths = bitpack.encode_symbols(q)
t_par, (d1, _) = timed(lambda: bitpack.pack_parallel(codes, lengths), reps=2)
cc, ll = codes.cpu(), lengths.cpu()
t_seq, (d2, _) = timed(lambda: bitpack.pack_sequential(cc, ll), reps=1)
assert d1 == d2
print(f"\nbit packing, {codes.numel():,} codewords: parallel {t_par*1e3:.1f}ms vs "
      f"sequential {t_seq*1e3:.1f}ms ({t_seq / t_par:.1f}x), outputs identical")

# quality vs bitrate on a test tone (CPU, one stream)
t = torch.arange(sr * 2) / sr
tone = ((0.5 * torch.sin(2 * math.pi * 440 * t) + 0.2 * torch.sin(2 * math.pi * 3000 * t)
         + 0.01 * torch.randn_like(t)) * 32767).unsqueeze(0)
print("\nquality vs bitrate (2s mono test signal):")
for br in [32_000, 64_000, 128_000, 256_000]:
    enc = codec.encode(tone, bitrate=br)
    real = enc["nbits"] / 2.0
    print(f"  target {br//1000:>3d} kbps -> actual {real/1000:6.1f} kbps, SNR {codec.snr_db(tone, codec.decode(enc)):5.1f} dB")
