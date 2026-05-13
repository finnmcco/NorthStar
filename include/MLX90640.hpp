#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  MLX90640 32×24 IR Array Driver  –  Raspberry Pi 5 (lgpio / I²C)
//
//  Datasheet rev.12 – December 2019
//  Default I²C address: 0x33
// ─────────────────────────────────────────────────────────────────────────────

namespace MLX90640 {

// ── Constants ────────────────────────────────────────────────────────────────

constexpr uint8_t  DEFAULT_I2C_ADDR   = 0x33;
constexpr int      PIXEL_COUNT        = 768;   // 32 × 24
constexpr int      EEPROM_SIZE        = 832;   // words
constexpr int      FRAME_DATA_SIZE    = 832;   // 768 pixels + 64 aux words

// Register addresses
constexpr uint16_t REG_STATUS         = 0x8000;
constexpr uint16_t REG_CONTROL        = 0x800D;
constexpr uint16_t REG_I2C_CONFIG     = 0x800F;
constexpr uint16_t REG_RAM_BASE       = 0x0400;
constexpr uint16_t REG_EEPROM_BASE    = 0x2400;

// Control register bit fields
constexpr uint16_t CTRL_REFRESH_MASK  = 0x0380;  // bits [9:7]
constexpr uint16_t CTRL_REFRESH_SHIFT = 7;
constexpr uint16_t CTRL_CHESS_MODE    = 0x1000;  // bit 12: 1=chess, 0=interleaved
constexpr uint16_t CTRL_SUBPAGE_MASK  = 0x0010;  // bit 4

// Status register bits
constexpr uint16_t STATUS_NEW_DATA    = 0x0008;  // bit 3
constexpr uint16_t STATUS_SUBPAGE     = 0x0001;  // bit 0

// ── Enumerations ─────────────────────────────────────────────────────────────

enum class RefreshRate : uint8_t {
    Hz0_5 = 0b000,   // 0.5 Hz
    Hz1   = 0b001,   // 1 Hz   (default)
    Hz2   = 0b010,
    Hz4   = 0b011,
    Hz8   = 0b100,
    Hz16  = 0b101,
    Hz32  = 0b110,
    Hz64  = 0b111,
};

enum class ReadPattern : uint8_t {
    Chess      = 1,   // recommended (lower noise)
    Interleaved = 0,
};

// ── EEPROM calibration data (parsed) ────────────────────────────────────────

struct CalibrationData {
    // Pixel gain / offset
    int16_t  pixOs[PIXEL_COUNT];
    int16_t  pixGain[PIXEL_COUNT];
    uint8_t  pixSensitivity[PIXEL_COUNT];  // packed 4-bit pairs in EEPROM
    float    alphaCorrR[4];

    // Compensation coefficients
    float    kVdd;
    float    vdd25;
    float    KvPTAT;
    float    KtPTAT;
    float    vPTAT25;
    float    alphaPTAT;

    float    gainEE;

    float    tgc;
    float    cpKv;
    float    cpKta;
    float    KsTa;

    float    ksTo[5];         // extended range corners
    float    ct[5];           // corner temperatures (°C)

    float    alpha[PIXEL_COUNT];
    float    offset[PIXEL_COUNT];
    float    kta[PIXEL_COUNT];
    float    kv[PIXEL_COUNT];

    float    cpAlpha[2];
    int16_t  cpOffset[2];

    float    ilChessC[3];

    int      resolutionEE;
    uint8_t  calibrationModeEE;  // 0=interleaved, 1=chess
};

// ── Main driver class ────────────────────────────────────────────────────────

class Driver {
public:
    // Open I²C bus (e.g. "/dev/i2c-1") and initialise the sensor.
    // Throws std::runtime_error on failure.
    explicit Driver(const std::string& i2cDevice,
                    uint8_t            address     = DEFAULT_I2C_ADDR,
                    RefreshRate        rate        = RefreshRate::Hz4,
                    ReadPattern        pattern     = ReadPattern::Chess);

    ~Driver();
    // Non-copyable
    Driver(const Driver&)            = delete;
    Driver& operator=(const Driver&) = delete;

    // ── Configuration ──────────────────────────────────────────────────────

    void setRefreshRate(RefreshRate rate);
    void setReadPattern(ReadPattern pattern);
    RefreshRate getRefreshRate() const;
    ReadPattern getReadPattern() const;

    // ── Data acquisition ───────────────────────────────────────────────────

    // Block until a new frame is available, then compute pixel temperatures.
    // Output: 768 floats in row-major order [row=0..23][col=0..31].
    // Returns the subpage index (0 or 1) just captured.
    int  getFrame(float* temperatures);

    // Lower-level: read raw frame words into caller-supplied buffer[834].
    // Returns subpage index, or -1 if no new data within timeoutMs.
    int  getRawFrame(uint16_t* frameData, int timeoutMs = 2000);

    // Convert a previously captured raw frame to temperatures.
    // emissivity: surface emissivity [0.0 – 1.0], typically 0.95.
    void rawToTemperatures(const uint16_t* frameData,
                           float*          temperatures,
                           float           emissivity = 0.95f) const;

    // ── Accessors ──────────────────────────────────────────────────────────

    const CalibrationData& calibration() const { return cal_; }

    // Ambient (chip) temperature from last frame [°C]
    float ambientTemperature() const { return Ta_; }

private:
    // I²C helpers
    void     writeWord(uint16_t reg, uint16_t value);
    uint16_t readWord(uint16_t reg);
    void     readWords(uint16_t reg, uint16_t* buf, int count);

    // Initialisation
    void     loadEEPROM();
    void     extractCalibration(const uint16_t* ee);
    void     applyRefreshRate();
    void     applyReadPattern();

    // Calculation helpers
    float    computeTa(const uint16_t* frameData) const;
    float    computeVdd(const uint16_t* frameData) const;

    // Members
    int          fd_  = -1;
    uint8_t      addr_;
    RefreshRate  rate_;
    ReadPattern  pattern_;
    CalibrationData cal_{};
    mutable float   Ta_  = 25.0f;  // last computed ambient temp
};

} // namespace MLX90640