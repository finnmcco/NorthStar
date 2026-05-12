#pragma once
#include <alsa/asoundlib.h>

// Hardware constants — established empirically with ICS43432 + googlevoicehat
// on Raspberry Pi 5. Change here and everything downstream updates.

namespace Config {

inline constexpr const char*       ALSA_DEVICE    = "hw:2,0";

// Capture — googlevoicehat driver native format
inline constexpr unsigned int      CAP_RATE       = 48000;
inline constexpr unsigned int      CAP_CHANNELS   = 2;       // stereo capture, left valid only
inline constexpr snd_pcm_uframes_t PERIOD_FRAMES  = 4800;    // 100ms per chunk @ 48kHz
inline constexpr snd_pcm_uframes_t BUFFER_FRAMES  = PERIOD_FRAMES * 4;

// ICS43432: left-justified 24-bit in a 32-bit word, >> 16 gives top 16 bits.
// Mic capsule is quiet — 8x gain established by record_prepare testing.
inline constexpr float             MIC_GAIN       = 12.0f;

// Vosk requires 16kHz mono int16 PCM
inline constexpr unsigned int      VOSK_RATE      = 16000;
inline constexpr double            RESAMPLE_RATIO = static_cast<double>(VOSK_RATE) / CAP_RATE;

// Pipeline tuning
inline constexpr int               QUEUE_MAX      = 16;      // ~1.6s of 16kHz chunks
inline constexpr int               DEBOUNCE_COUNT = 2;       // consecutive partials required

} // namespace Config
