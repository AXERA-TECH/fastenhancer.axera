#include "fastenhancer.hpp"
#include "windows.h"
#include "ax_engine_type.h"
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr float kCompPow = 0.3f - 1.0f;
constexpr float kDecompPow = 1.0f / 0.3f - 1.0f;

// Radix-2 DIT FFT (same algorithm as kiss_fft / rnnoise, self-contained).
// Reference: Lightweight-Speech-Denoising.axera and ZipVoice.AXERA

static void fft_c2c(float* re, float* im, int N, int inv) {
    for (int i = 1, j = 0; i < N; ++i) {
        int m = N >> 1;
        for (; j & m; m >>= 1) j ^= m;
        j ^= m;
        if (i < j) { float t=re[i]; re[i]=re[j]; re[j]=t; t=im[i]; im[i]=im[j]; im[j]=t; }
    }
    for (int L = 2; L <= N; L <<= 1) {
        double a = (inv ? 2.0 : -2.0) * kPi / L;
        for (int k = 0; k < N; k += L) {
            double wr = 1.0, wi = 0.0, wdr = std::cos(a), wdi = std::sin(a);
            for (int j = 0; j < L/2; ++j) {
                int i1 = k+j, i2 = k+j+L/2;
                float tr = (float)(wr*re[i2] - wi*im[i2]);
                float ti = (float)(wr*im[i2] + wi*re[i2]);
                re[i2] = re[i1] - tr; im[i2] = im[i1] - ti;
                re[i1] += tr; im[i1] += ti;
                double nwr = wr*wdr - wi*wdi; wi = wr*wdi + wi*wdr; wr = nwr;
            }
        }
    }
    if (inv) { for (int i = 0; i < N; ++i) { re[i] /= N; im[i] /= N; } }
}

static void rfft(const float* x, float* re_out, float* im_out, int N) {
    float re[N], im[N];
    for (int i = 0; i < N; ++i) { re[i] = x[i]; im[i] = 0; }
    fft_c2c(re, im, N, 0);
    int H = N/2 + 1;
    for (int k = 0; k < H; ++k) { re_out[k] = re[k]; im_out[k] = im[k]; }
}

static void ifft_real(const float* re_half, const float* im_half, float* out, int N) {
    float re[N], im[N]; int H = N/2;
    for (int k = 0; k <= H; ++k) { re[k] = re_half[k]; im[k] = im_half[k]; }
    for (int k = 1; k < H; ++k) { re[N-k] = re_half[k]; im[N-k] = -im_half[k]; }
    fft_c2c(re, im, N, 1);
    for (int i = 0; i < N; ++i) out[i] = re[i];
}
}  // namespace

FastEnhancer::FastEnhancer()
    : handle_(0), context_(0), loaded_(false),
      model_data_(nullptr), model_size_(0), io_data_(nullptr), io_info_(nullptr) {
    std::memset(cache_stft_, 0, sizeof(cache_stft_));
    std::memset(cache_istft_, 0, sizeof(cache_istft_));
    std::memset(gru0_, 0, sizeof(gru0_));
    std::memset(gru1_, 0, sizeof(gru1_));
}

FastEnhancer::~FastEnhancer() {
    if (io_data_) {
        AX_ENGINE_IO_T* io = (AX_ENGINE_IO_T*)io_data_;
        for (AX_U32 i = 0; i < io->nInputSize; ++i)
            AX_SYS_MemFree(io->pInputs[i].phyAddr, io->pInputs[i].pVirAddr);
        for (AX_U32 i = 0; i < io->nOutputSize; ++i)
            AX_SYS_MemFree(io->pOutputs[i].phyAddr, io->pOutputs[i].pVirAddr);
        delete[] io->pInputs;
        delete[] io->pOutputs;
        delete io;
    }
    if (handle_) AX_ENGINE_DestroyHandle(handle_);
    std::free(model_data_);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
}

bool FastEnhancer::load(const char* model_path) {
    // 0. read model file into memory
    FILE* f = std::fopen(model_path, "rb");
    if (!f) { std::fprintf(stderr, "FastEnhancer: cannot open %s\n", model_path); return false; }
    std::fseek(f, 0, SEEK_END); model_size_ = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    model_data_ = std::malloc(model_size_);
    if (!model_data_ || std::fread(model_data_, 1, model_size_, f) != model_size_) {
        std::fclose(f); return false;
    }
    std::fclose(f);

    // 1. init SYS + ENGINE
    if (AX_SYS_Init() != 0) { std::fprintf(stderr, "FastEnhancer: AX_SYS_Init failed\n"); return false; }
    AX_ENGINE_NPU_ATTR_T npu_attr;
    std::memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&npu_attr) != 0) { std::fprintf(stderr, "FastEnhancer: AX_ENGINE_Init failed\n"); return false; }

    // 2. create handle + context
    if (AX_ENGINE_CreateHandle(&handle_, model_data_, model_size_) != 0 || !handle_) {
        std::fprintf(stderr, "FastEnhancer: CreateHandle failed\n"); return false;
    }
    if (AX_ENGINE_CreateContext(handle_) != 0) {
        std::fprintf(stderr, "FastEnhancer: CreateContext failed\n"); return false;
    }

    // 3. get IO info
    AX_ENGINE_IO_INFO_T* info = nullptr;
    if (AX_ENGINE_GetIOInfo(handle_, &info) != 0 || !info) {
        std::fprintf(stderr, "FastEnhancer: GetIOInfo failed\n"); return false;
    }
    io_info_ = info;

    // 4. allocate CMM I/O buffers (physically contiguous, DMA-capable)
    AX_ENGINE_IO_T* io = new AX_ENGINE_IO_T;
    std::memset(io, 0, sizeof(*io));
    io->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    std::memset(io->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
    io->nInputSize = info->nInputSize;
    for (AX_U32 i = 0; i < info->nInputSize; ++i) {
        AX_ENGINE_IO_BUFFER_T* b = &io->pInputs[i];
        b->nSize = info->pInputs[i].nSize;
        if (AX_SYS_MemAlloc((AX_U64*)&b->phyAddr, &b->pVirAddr, b->nSize, 128,
                            (const AX_S8*)"fastenhancer") != 0) {
            std::fprintf(stderr, "FastEnhancer: MemAlloc input %u failed\n", i); return false;
        }
        std::memset(b->pVirAddr, 0, b->nSize);
    }
    io->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    std::memset(io->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
    io->nOutputSize = info->nOutputSize;
    for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
        AX_ENGINE_IO_BUFFER_T* b = &io->pOutputs[i];
        b->nSize = info->pOutputs[i].nSize;
        if (AX_SYS_MemAlloc((AX_U64*)&b->phyAddr, &b->pVirAddr, b->nSize, 128,
                            (const AX_S8*)"fastenhancer") != 0) {
            std::fprintf(stderr, "FastEnhancer: MemAlloc output %u failed\n", i); return false;
        }
    }
    io_data_ = io;
    loaded_ = true;
    return true;
}

void FastEnhancer::stft_forward(const float* hop, float* spec_out) {
    float x[kNfft];
    std::memcpy(x, cache_stft_, kCacheLen * sizeof(float));
    std::memcpy(x + kCacheLen, hop, kHop * sizeof(float));
    std::memcpy(cache_stft_, x + (kNfft - kCacheLen), kCacheLen * sizeof(float));
    for (int i = 0; i < kNfft; ++i) x[i] *= kWindow[i];
    float re[kFreq], im[kFreq];
    rfft(x, re, im, kNfft);
    for (int k = 0; k < kFreq; ++k) {
        spec_out[k * 2 + 0] = re[k];
        spec_out[k * 2 + 1] = im[k];
    }
}

void FastEnhancer::istft_inverse(const float* spec_in, float* hop_out) {
    float re[kNfft] = {0}, im[kNfft] = {0};
    for (int k = 0; k < kFreq; ++k) {
        re[k] = spec_in[k * 2 + 0];
        im[k] = spec_in[k * 2 + 1];
    }
    float x_0 = spec_in[0 * 2 + 0];
    float x_last = spec_in[(kFreq - 1) * 2 + 0];
    float t[kNfft];
    ifft_real(re, im, t, kNfft);
    for (int j = 0; j < kNfft / 2; ++j) {
        t[j * 2 + 0] = 2.0f * t[j * 2 + 0] - (x_0 + x_last) / kNfft;
        t[j * 2 + 1] = 2.0f * t[j * 2 + 1] - (x_0 - x_last) / kNfft;
    }
    for (int i = 0; i < kNfft; ++i) t[i] *= kWindowIstft[i];
    for (int i = 0; i < kCacheLen; ++i) t[i] += cache_istft_[i];
    std::memcpy(hop_out, t, kHop * sizeof(float));
    std::memcpy(cache_istft_, t + kHop, kCacheLen * sizeof(float));
}

bool FastEnhancer::run_core(const float* comp_spec, float* mask) {
    const int specBytes = kF * 2 * sizeof(float);
    const int gruBytes = kGruF * kGruC * sizeof(float);
    AX_ENGINE_IO_T* io = (AX_ENGINE_IO_T*)io_data_;

    // Fill CMM input buffers (order: comp_spec, cache_in_0, cache_in_1)
    std::memcpy(io->pInputs[0].pVirAddr, comp_spec, specBytes);
    std::memcpy(io->pInputs[1].pVirAddr, gru0_, gruBytes);
    std::memcpy(io->pInputs[2].pVirAddr, gru1_, gruBytes);

    if (AX_ENGINE_RunSync(handle_, io) != 0) return false;

    // Read CMM output buffers (order: mask, cache_out_0, cache_out_1)
    std::memcpy(mask, io->pOutputs[0].pVirAddr, specBytes);
    std::memcpy(gru0_, io->pOutputs[1].pVirAddr, gruBytes);
    std::memcpy(gru1_, io->pOutputs[2].pVirAddr, gruBytes);
    return true;
}

std::vector<float> FastEnhancer::enhance(const std::vector<float>& wav_in) {
    int length = static_cast<int>(wav_in.size());
    std::vector<float> wav(length + kNfft, 0.0f);
    for (int i = 0; i < length; ++i) {
        float v = wav_in[i];
        wav[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
    }
    std::memset(cache_stft_, 0, sizeof(cache_stft_));
    std::memset(cache_istft_, 0, sizeof(cache_istft_));
    std::memset(gru0_, 0, sizeof(gru0_));
    std::memset(gru1_, 0, sizeof(gru1_));

    std::vector<float> out;
    out.reserve(length + kNfft);
    float spec[kFreq * 2], comp[kF * 2], mask[kF * 2], sh[kFreq * 2], hop_out[kHop];
    for (int i = 0; i + kHop <= static_cast<int>(wav.size()); i += kHop) {
        stft_forward(&wav[i], spec);
        // compress: drop last bin, comp = s * mag^-0.7
        for (int k = 0; k < kF; ++k) {
            float re = spec[k * 2 + 0], im = spec[k * 2 + 1];
            float mag = std::sqrt(re * re + im * im);
            if (mag < 1e-5f) mag = 1e-5f;
            float scale = std::pow(mag, kCompPow);
            comp[k * 2 + 0] = re * scale;
            comp[k * 2 + 1] = im * scale;
        }
        if (!run_core(comp, mask)) break;
        // apply complex mask + decompress
        for (int k = 0; k < kF; ++k) {
            float cr = comp[k * 2 + 0], cimg = comp[k * 2 + 1];
            float mr = mask[k * 2 + 0], mi = mask[k * 2 + 1];
            float hr = cr * mr - cimg * mi;
            float hi = cr * mi + cimg * mr;
            float mag = std::sqrt(hr * hr + hi * hi);
            float scale = std::pow(mag, kDecompPow);
            sh[k * 2 + 0] = hr * scale;
            sh[k * 2 + 1] = hi * scale;
        }
        sh[kF * 2 + 0] = 0.0f;  // padded last bin
        sh[kF * 2 + 1] = 0.0f;
        istft_inverse(sh, hop_out);
        out.insert(out.end(), hop_out, hop_out + kHop);
    }
    int start = kNfft - kHop;
    std::vector<float> result(length, 0.0f);
    for (int i = 0; i < length && start + i < static_cast<int>(out.size()); ++i) {
        float v = out[start + i];
        result[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
    }
    return result;
}
