"""Generate calibration data for Pulsar2 compile.

Runs real audio streaming inference through the ONNXModel to collect compressed
STFT frames and GRU cache states, saving them as .npy tar.gz for Pulsar2 MinMax.

Usage (from origin/):
    python generate_calib.py --model 16k --audio onnx/p232_013.wav onnx/p232_001-009.wav
    python generate_calib.py --model 48k --audio onnx/p232_013.wav onnx/p232_001-009.wav
"""
import sys, os, argparse, torch, yaml, librosa, numpy as np, tarfile, shutil
sys.path.insert(0, ".")

def generate(tag, audio_files):
    log_dir = "logs/fastenhancer_t" if tag == "16k" else "logs/fastenhancer_t48"
    with open(os.path.join(log_dir, "config.yaml")) as f: cfg = yaml.safe_load(f)
    mk, rf = dict(cfg["model_kwargs"]), dict(cfg["model_kwargs"]["rnnformer_kwargs"])
    n_fft, hop = mk["n_fft"], mk["hop_size"]
    F, gf, gc = n_fft // 2, rf["freq"], rf["channels"]
    sr = cfg["data"]["sampling_rate"]
    comp_pow = mk["input_compression"] - 1.0

    from models.fastenhancer.default.model import ONNXModel
    base = ONNXModel(**mk); base.eval()
    ckpt = torch.load(os.path.join(log_dir, "00500.pth"), map_location="cpu", weights_only=False)
    base.load_state_dict(ckpt["model"], strict=False)
    base.remove_weight_reparameterizations(); base.flatten_parameters()
    stft = base.stft

    cs_s, c0_s, c1_s = [], [], []
    for af in audio_files:
        wav, _ = librosa.load(af, sr=sr)
        wav = torch.from_numpy(wav).view(1, -1).clamp(-1, 1)
        length = wav.size(-1)
        wav = torch.nn.functional.pad(wav, (0, n_fft))
        cst, _ = stft.initialize_cache(torch.randn(1))
        cm = base.initialize_cache(torch.randn(1))
        with torch.no_grad():
            for idx, i in enumerate(range(0, length + n_fft - hop, hop)):
                wi = wav[:, i:i+hop]
                if wi.size(1) < hop: wi = torch.nn.functional.pad(wi, (0, hop - wi.size(1)))
                spec_in, cst = stft(wi, cst)
                s = spec_in[:, :-1, :, :]
                mag = torch.linalg.norm(s, dim=-1, keepdim=True).clamp(min=1e-5)
                comp = (s * mag.pow(comp_pow)).numpy().astype(np.float32)
                if idx % (40 if tag == "48k" else 30) == 0:
                    cs_s.append(comp); c0_s.append(cm[0].numpy().astype(np.float32)); c1_s.append(cm[1].numpy().astype(np.float32))
                spec_out, *cm = base(spec_in, *cm)

    od = f"../model_convert/{tag}/calib"
    if os.path.exists(od): shutil.rmtree(od)
    N = min(16, len(cs_s))
    for name, samples in [("comp_spec", cs_s), ("cache_in_0", c0_s), ("cache_in_1", c1_s)]:
        d = os.path.join(od, name); os.makedirs(d)
        for i in range(N): np.save(os.path.join(d, f"{i:04d}.npy"), samples[i])
        with tarfile.open(os.path.join(od, f"{name}.tar.gz"), "w:gz") as tar:
            for npy in sorted(os.listdir(d)): tar.add(os.path.join(d, npy), arcname=npy)
        print(f"  {name}.tar.gz: {N} samples, {os.path.getsize(os.path.join(od, f'{name}.tar.gz')):,}B")
    print(f"{tag} calib: comp range [{min(s.min() for s in cs_s):.2f},{max(s.max() for s in cs_s):.2f}]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["16k","48k"])
    ap.add_argument("--audio", nargs="+", required=True)
    generate(ap.parse_args().model, ap.parse_args().audio)
