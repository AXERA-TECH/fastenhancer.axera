#pragma once
// Model selection: build with -DFE_MODEL_48K for 48kHz, default is 16kHz.

#ifdef FE_MODEL_48K
  #define FE_NFFT 1024
  #define FE_HOP 512
  #define FE_GRU_F 24
  #define FE_GRU_C 20
  #define FE_SAMPLE_RATE 48000
#else
  #define FE_NFFT 512
  #define FE_HOP 256
  #define FE_GRU_F 16
  #define FE_GRU_C 20
  #define FE_SAMPLE_RATE 16000
#endif
