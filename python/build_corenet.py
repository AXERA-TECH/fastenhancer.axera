"""Build CoreNet ONNX for both 16kHz and 48kHz FastEnhancer models.

CoreNet architecture: compressed spec -> complex mask (no Pow/ReduceL2).
Run this from the origin/ directory to export model.onnx.

Usage:
    python build_corenet.py --model 16k   # exports to ../model_convert/16k/
    python build_corenet.py --model 48k   # exports to ../model_convert/48k/
"""
import sys, os, argparse, torch, onnx, yaml, numpy as np
sys.path.insert(0, ".")

def build(tag):
    log_dir = "logs/fastenhancer_t" if tag == "16k" else "logs/fastenhancer_t48"
    with open(os.path.join(log_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    mk, rf = dict(cfg["model_kwargs"]), dict(cfg["model_kwargs"]["rnnformer_kwargs"])
    F, gf, gc = mk["n_fft"] // 2, rf["freq"], rf["channels"]

    from models.fastenhancer.default.model import ONNXModel
    base = ONNXModel(**mk); base.eval()
    ckpt = torch.load(os.path.join(log_dir, "00500.pth"), map_location="cpu", weights_only=False)
    base.load_state_dict(ckpt["model"], strict=False)
    base.remove_weight_reparameterizations(); base.flatten_parameters()

    class CoreNet(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, comp, c0, c1):
            mask, co = self.m.model_forward(comp, c0, c1)
            return mask, co[0], co[1]

    net = CoreNet(base); net.eval()
    comp = torch.randn(1, F, 1, 2)
    c0 = torch.zeros(1, gf, gc); c1 = torch.zeros(1, gf, gc)
    od = f"../model_convert/{tag}"
    os.makedirs(od, exist_ok=True)
    torch.onnx.export(net, (comp, c0, c1), f"{od}/model_raw.onnx",
        input_names=["comp_spec", "cache_in_0", "cache_in_1"],
        output_names=["mask", "cache_out_0", "cache_out_1"], dynamo=False, opset_version=17)
    from onnxsim import simplify; import onnxslim
    ms, ok = simplify(onnx.load(f"{od}/model_raw.onnx")); assert ok
    onnx.save(onnxslim.slim(ms), f"{od}/model.onnx")
    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{od}/model.onnx", providers=["CPUExecutionProvider"])
    o = sess.run(None, {"comp_spec": np.random.randn(1,F,1,2).astype(np.float32),
        "cache_in_0": np.zeros((1,gf,gc),np.float32), "cache_in_1": np.zeros((1,gf,gc),np.float32)})
    with torch.no_grad(): to = net(torch.randn(1,F,1,2), torch.zeros(1,gf,gc), torch.zeros(1,gf,gc))
    cos = np.dot(o[0].ravel(), to[0].numpy().ravel()) / (np.linalg.norm(o[0].ravel()) * np.linalg.norm(to[0].numpy().ravel()) + 1e-12)
    print(f"{tag}: ONNX {os.path.getsize(f'{od}/model.onnx'):,}B, ORT vs Torch cos={cos:.6f} {'PASS' if cos>0.999 else 'FAIL'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["16k","48k"])
    build(ap.parse_args().model)
