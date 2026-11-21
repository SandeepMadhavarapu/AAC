# gpu-aac-lab

A small, tested prototype of the **parallel core of an AAC-style audio encoder** in PyTorch, built to answer one question: *which parts of an AAC encoder actually parallelize on a GPU, and which don't?*

It is **not** a conformant AAC encoder (no official Huffman tables, scale-factor bands, TNS, M/S or ADTS framing). It implements the parts that decide the GPU design and measures them.

```
audio [streams, channels, samples]
  -> frame (unfold view, hop 1024)                    parallel, no copy
  -> MDCT via batched FFT                             parallel over every frame/channel/stream
  -> psychoacoustic thresholds (band energy, spread)  parallel per frame
  -> trial quantization at G candidate gains          parallel over (gain, frame)
  -> bit-reservoir choice                             SEQUENTIAL, but only reads a [G, F] table
  -> codeword lengths -> prefix sum -> bit scatter    parallel packing, one stream
decode: sequential prefix-code read -> dequantize -> IMDCT -> overlap-add (parallel)
```

## Design decisions

- **Batch everything.** Signals are `[stream, channel, frame, 1024]` contiguous float32, so neighbouring threads read neighbouring coefficients. One FFT call covers every frame of every stream.
- **Isolate the sequential steps instead of fighting them.** The bit reservoir makes each frame's budget depend on the frames before it. Instead of running the rate loop frame by frame, every frame is quantized at all candidate gains in parallel. A short sequential pass then picks one candidate per frame from the precomputed bits/noise table, without ever re-quantizing.
- **Variable-length codes without a serial writer.** Bit positions come from an exclusive prefix sum over codeword lengths, so every bit has a unique address and all of them are written in one scatter with no conflicts. A sequential reference packer is kept, and tests check that both produce identical bytes.
- **fp32 in the signal path.** The FFT math runs in complex64; quantization error is audible, so no fp16.

## Tests (11, `pytest`)

- FFT MDCT matches the O(N·M) textbook definition
- **Perfect reconstruction (TDAC)**: MDCT → IMDCT → overlap-add returns the input to < 1e-4
- Batched results equal per-frame results
- Parallel packer output is byte-identical to the sequential packer (1, 7 and 5,000 symbols)
- Bitstream decodes back to the exact quantized values; the bit estimate used for rate control is exact
- Finer gain means more bits and less noise; the reservoir never overspends on average and lets a transient borrow bits
- End to end: higher bitrate gives higher SNR

## Results (CPU, 2 threads, 80 s of stereo audio in one batch)

| stage | time | × realtime |
|---|---|---|
| MDCT (batched FFT) | 143 ms | ~560× |
| psychoacoustic thresholds | 18 ms | ~4,400× |
| trial quantization, 31 gains | 4.1 s | ~20× |
| reservoir pass, 2,000 frames (sequential) | 30 ms | n/a |

| target bitrate | actual | SNR |
|---|---|---|
| 32 kbps | 49.5 kbps | 16.4 dB |
| 64 kbps | 64.8 kbps | 31.9 dB |
| 128 kbps | 129.5 kbps | 38.7 dB |
| 256 kbps | 259.3 kbps | 52.3 dB |

## What the numbers taught me

1. **Trial quantization is the bottleneck, not the MDCT.** It does G× the work of everything else (pow, log2 per value per gain). That makes it the first candidate for a fused Triton kernel that computes bits and noise without writing the `[G, F, M]` intermediate to memory. The prototype already processes frames in chunks to bound memory after the first version ran out.
2. **The parallel packer is about 3× *slower* than the sequential one on a 2-thread CPU.** It does more total work (it expands every codeword to bits) and only pays off with thousands of GPU threads. I haven't measured it on a GPU yet; `python bench.py` reruns everything on CUDA automatically.
3. **The code, not the rate loop, sets the low-bitrate floor.** At 32 kbps the encoder can't get below ~49 kbps, because my Exp-Golomb code spends at least 1 bit per coefficient. Real AAC goes lower by coding whole sections as zero and packing 2 or 4 values per Huffman codeword. That's the next thing I'd add.

## Run it

```
pip install torch numpy pytest
python -m pytest -q
python bench.py --streams 16 --seconds 5
```

## Next steps
- Fused Triton kernel for trial quantization + bit counting
- Per-band scalefactors driven by the psychoacoustic thresholds (currently one global gain per frame)
- Real AAC spectral Huffman codebooks with 2/4-tuples and zero sections
- GPU benchmark on many concurrent streams; check output against FFmpeg's AAC encoder
