#include "fastenhancer.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <ctime>
#include <chrono>

// Minimal WAV I/O (16-bit PCM mono).
static std::vector<float> read_wav(const char* path, int* sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return {}; }
    char hdr[44];
    f.read(hdr, 44);
    *sr = *reinterpret_cast<int32_t*>(hdr + 24);
    int16_t ch = *reinterpret_cast<int16_t*>(hdr + 22);
    // Read raw PCM data
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    int16_t* p = reinterpret_cast<int16_t*>(raw.data());
    size_t n = raw.size() / 2;
    std::vector<float> out;
    for (size_t i = 0; i < n; i += ch) out.push_back(p[i] / 32768.0f);
    return out;
}

static void write_wav(const char* path, const std::vector<float>& data, int sr) {
    std::ofstream f(path, std::ios::binary);
    int32_t n = static_cast<int32_t>(data.size());
    int32_t byte_rate = sr * 2, data_bytes = n * 2, chunk = 36 + data_bytes;
    int16_t one = 1, two = 2, sixteen = 16;
    f.write("RIFF", 4); f.write(reinterpret_cast<char*>(&chunk), 4); f.write("WAVE", 4);
    f.write("fmt ", 4); int32_t sz = 16; f.write(reinterpret_cast<char*>(&sz), 4);
    f.write(reinterpret_cast<char*>(&one), 2); f.write(reinterpret_cast<char*>(&one), 2);
    f.write(reinterpret_cast<char*>(&sr), 4); f.write(reinterpret_cast<char*>(&byte_rate), 4);
    f.write(reinterpret_cast<char*>(&two), 2); f.write(reinterpret_cast<char*>(&sixteen), 2);
    f.write("data", 4); f.write(reinterpret_cast<char*>(&data_bytes), 4);
    for (float v : data) {
        int16_t s = static_cast<int16_t>((v > 1 ? 1 : (v < -1 ? -1 : v)) * 32767);
        f.write(reinterpret_cast<char*>(&s), 2);
    }
}

static float rms(const std::vector<float>& x) {
    double s = 0; for (float v : x) s += double(v) * v;
    return x.empty() ? 0.f : std::sqrt(s / x.size());
}

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] : "../models/model.axmodel";
    const char* input = argc > 2 ? argv[2] : "sample_noisy.wav";
    const char* output = argc > 3 ? argv[3] : "enhanced.wav";

    int sr = 0;
    std::vector<float> wav = read_wav(input, &sr);
    if (wav.empty()) return 1;
    if (sr != FastEnhancer::kSampleRate)
        std::fprintf(stderr, "WARNING: sr=%d, model expects %d Hz\n", sr, FastEnhancer::kSampleRate);

    bool bench = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--bench") == 0) bench = true;

    FastEnhancer fe;
    if (!fe.load(model)) return 1;

    std::vector<float> enhanced = fe.enhance(wav);  // warmup

    if (bench) {
        const int N = 3;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) enhanced = fe.enhance(wav);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count() / N;
        double dur = double(wav.size()) / FastEnhancer::kSampleRate;
        std::printf("Benchmark %dkHz: %.3fs / %.1fs audio, RTF=%.4f\n",
                    FastEnhancer::kSampleRate / 1000, sec, dur, sec / dur);
    } else {
        write_wav(output, enhanced, FastEnhancer::kSampleRate);
        std::printf("FastEnhancer %dkHz done.\n  input rms  = %.4f\n  output rms = %.4f\n  saved: %s\n",
                    FastEnhancer::kSampleRate / 1000, rms(wav), rms(enhanced), output);
    }
    return 0;
}
