#!/usr/bin/env python3
"""FastEnhancer 16kHz AX650 demo — denoise a wav on the board.

Usage (on AX650 board):
    python3 demo.py --model ../models/model.axmodel --input noisy.wav --output enhanced.wav

If --input is omitted, a bundled noisy sample is used.
"""
import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fastenhancer_sdk import FastEnhancer


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path, data, sr):
    data = np.clip(data, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/model.axmodel")
    ap.add_argument("--input", default="sample_noisy.wav")
    ap.add_argument("--output", default="enhanced.wav")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("ERROR: model not found: %s" % args.model)
        sys.exit(1)

    wav, sr = read_wav(args.input)
    fe = FastEnhancer(args.model)
    if sr != fe.sr:
        print("WARNING: input sr=%d, model expects %d Hz. Resample first." % (sr, fe.sr))

    enhanced = fe.enhance(wav)

    rms_in = float(np.sqrt(np.mean(wav ** 2)))
    rms_out = float(np.sqrt(np.mean(enhanced ** 2)))
    write_wav(args.output, enhanced, fe.sr)
    print("FastEnhancer 16kHz done.")
    print("  input rms  = %.4f" % rms_in)
    print("  output rms = %.4f" % rms_out)
    print("  saved: %s" % args.output)


if __name__ == "__main__":
    main()
