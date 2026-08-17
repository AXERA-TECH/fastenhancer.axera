"""FastEnhancer streaming speech enhancement SDK for AX650 NPU.

Split pipeline — everything with FFT / wide dynamic range runs on the host CPU,
the NPU runs only the well-bounded neural core (compressed spec -> mask):

  host:  STFT  ->  magnitude compression (mag^-0.7)
  NPU:   neural core  (compressed spec -> complex mask)     [model.axmodel]
  host:  apply complex mask  ->  decompression (mag^2.333)  ->  ISTFT

STFT/ISTFT and compression/decompression are the model's own math, re-implemented
in pure numpy. The SDK depends only on numpy + pyaxengine (no torch / onnxruntime).

The compression/decompression are kept on the host because `mag^2.333` amplifies
any quantization error; keeping them off the NPU makes U16 quantization near-lossless.

Config is read from model_meta.json (n_fft/hop/freq/channels) so one SDK serves
both the 16k and 48k models.
"""
import json
import os
import numpy as np

try:
    import axengine as axe
    _HAS_AXE = True
except Exception:                       # pragma: no cover - board only
    _HAS_AXE = False

_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_COMPRESSION = 0.3
_COMP_POW = _INPUT_COMPRESSION - 1.0            # -0.7  (compress)
_DECOMP_POW = 1.0 / _INPUT_COMPRESSION - 1.0    # +2.333 (decompress)


class _HostSTFT:
    """Pure-numpy replica of the model's ONNXSTFT (cached streaming windowed FFT)."""

    def __init__(self, n_fft, hop):
        self.n_fft = n_fft
        self.hop = hop
        self.cache_len = n_fft - hop
        win_dir = os.path.join(_DIR, "16k" if n_fft == 512 else "48k")
        self.window = np.load(os.path.join(win_dir, "window.npy")).astype(np.float32)
        self.window_istft = np.load(os.path.join(win_dir, "window_istft.npy")).astype(np.float32)

    def forward(self, wav_hop, cache):
        x = np.concatenate([cache, wav_hop.reshape(1, -1)], axis=1)
        new_cache = x[:, -self.cache_len:].copy()
        X = np.fft.rfft(x * self.window, axis=1)
        spec = np.stack([X.real, X.imag], axis=-1)[:, :, None, :]   # [1,F+1,1,2]
        return spec.astype(np.float32), new_cache.astype(np.float32)

    def inverse(self, spec, cache):
        n_fft = self.n_fft
        x_0 = spec[:, 0:1, 0, 0]
        x_last = spec[:, -1:, 0, 0]
        s = np.pad(spec[:, :, 0, :], ((0, 0), (0, n_fft // 2 - 1), (0, 0)))
        x = np.fft.ifft(s[..., 0] + 1j * s[..., 1], axis=1).real
        x = x.reshape(-1, n_fft // 2, 2)
        x = 2 * x - np.stack([x_0 + x_last, x_0 - x_last], axis=2) / n_fft
        x = x.reshape(-1, n_fft) * self.window_istft
        x[:, :self.cache_len] = x[:, :self.cache_len] + cache
        out = x[:, :self.hop]
        return out.reshape(-1).astype(np.float32), x[:, self.hop:].astype(np.float32)

    def init_cache(self):
        return (np.zeros((1, self.cache_len), np.float32),
                np.zeros((1, self.cache_len), np.float32))


class FastEnhancer:
    """End-to-end streaming denoiser: host STFT/compress + NPU core + host decompress/ISTFT."""

    def __init__(self, model_path, meta_path=None):
        if not _HAS_AXE:
            raise RuntimeError("pyaxengine (axengine) not available — run on the AX650 board.")
        meta_path = meta_path or os.path.join(os.path.dirname(model_path), "model_meta.json")
        meta = json.load(open(meta_path))
        self.sr = meta["sample_rate"]
        self.n_fft = meta["n_fft"]
        self.hop = meta["hop_size"]
        self.F = self.n_fft // 2
        gru = meta["cache_info"]["shape"]        # [1, gf, gc]
        self.gru_shape = tuple(gru)
        self.n_cache = meta["cache_info"]["count"]
        self.host = _HostSTFT(self.n_fft, self.hop)
        self.session = axe.InferenceSession(model_path)

    def enhance(self, wav):
        """wav: np.ndarray [T] float32 mono in [-1,1] at model sr -> enhanced [T]."""
        wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        length = len(wav)
        wav = np.pad(wav, (0, self.n_fft))
        cs, ci = self.host.init_cache()
        gru = [np.zeros(self.gru_shape, np.float32) for _ in range(self.n_cache)]

        out = []
        for i in range(0, length + self.n_fft - self.hop, self.hop):
            hop = wav[i:i + self.hop]
            if len(hop) < self.hop:
                hop = np.pad(hop, (0, self.hop - len(hop)))
            spec, cs = self.host.forward(hop, cs)               # [1,F+1,1,2]
            # compress on host
            s = spec[:, :-1, :, :]
            mag = np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), 1e-5)
            comp = (s * mag ** _COMP_POW).astype(np.float32)    # [1,F,1,2]
            # NPU core: compressed spec -> mask
            feed = {"comp_spec": comp}
            for j, c in enumerate(gru):
                feed["cache_in_%d" % j] = c
            outs = self.session.run(None, feed)
            mask, gru = outs[0], list(outs[1:])
            # apply complex mask + decompress on host
            sh = np.stack([comp[..., 0] * mask[..., 0] - comp[..., 1] * mask[..., 1],
                           comp[..., 0] * mask[..., 1] + comp[..., 1] * mask[..., 0]], axis=3)
            mc = np.linalg.norm(sh, axis=3, keepdims=True)
            sh = sh * mc ** _DECOMP_POW
            sh = np.pad(sh, ((0, 0), (0, 1), (0, 0), (0, 0)))   # add back last bin -> [1,F+1,1,2]
            wav_out, ci = self.host.inverse(sh.astype(np.float32), ci)
            out.append(wav_out)

        wav_hat = np.concatenate(out)
        start = self.n_fft - self.hop
        return np.clip(wav_hat[start:start + length], -1.0, 1.0)


# Backward-compatible alias
FastEnhancer16k = FastEnhancer
