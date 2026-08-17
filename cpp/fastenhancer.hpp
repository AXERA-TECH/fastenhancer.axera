#pragma once
// Model: default 16kHz. Build with -DFE_MODEL_48K for 48kHz model.
#include "config.h"
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

// FastEnhancer streaming speech enhancement (AX650 NPU).
//
// Split pipeline — host does FFT + magnitude compression, NPU does the neural
// core (compressed spec -> complex mask), host does mask + decompression + ISTFT.
// Compression/decompression stay on host because mag^2.333 amplifies quant error.
//
// Dimensions from config.h (switched via -DFE_MODEL_48K at build time).
class FastEnhancer {
public:
    static constexpr int kNfft = FE_NFFT;
    static constexpr int kHop = FE_HOP;
    static constexpr int kCacheLen = kNfft - kHop;
    static constexpr int kFreq = kNfft / 2 + 1;
    static constexpr int kF = kNfft / 2;
    static constexpr int kGruF = FE_GRU_F;
    static constexpr int kGruC = FE_GRU_C;
    static constexpr int kSampleRate = FE_SAMPLE_RATE;

    FastEnhancer();
    ~FastEnhancer();

    bool load(const char* model_path);
    std::vector<float> enhance(const std::vector<float>& wav);

private:
    bool run_core(const float* comp_spec, float* mask);
    void stft_forward(const float* hop, float* spec_out);
    void istft_inverse(const float* spec_in, float* hop_out);

    void* handle_;
    void* context_;              // AX_ENGINE context
    bool loaded_;
    void* model_data_;           // model file bytes
    size_t model_size_;
    void* io_data_;              // AX_ENGINE_IO_T* (CMM-allocated I/O buffers)
    void* io_info_;              // AX_ENGINE_IO_INFO_T*
    float cache_stft_[kCacheLen];
    float cache_istft_[kCacheLen];
    float gru0_[kGruF * kGruC];
    float gru1_[kGruF * kGruC];
};
