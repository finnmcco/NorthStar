// MLX90640.cpp
#include "MLX90640.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>

namespace {

inline int16_t signExtend(uint16_t val, int bits) {
    uint16_t mask = 1u << (bits - 1);
    return static_cast<int16_t>((val ^ mask) - mask);
}

} // anonymous namespace

namespace MLX90640 {

Driver::Driver(const std::string& i2cDevice,
               uint8_t            address,
               RefreshRate        rate,
               ReadPattern        pattern)
    : addr_(address), rate_(rate), pattern_(pattern)
{
    fd_ = ::open(i2cDevice.c_str(), O_RDWR);
    if (fd_ < 0)
        throw std::runtime_error(
            "MLX90640: cannot open " + i2cDevice + " (" + strerror(errno) + ")");

    if (::ioctl(fd_, I2C_SLAVE, addr_) < 0)
        throw std::runtime_error(
            "MLX90640: cannot set I2C slave address (" + std::string(strerror(errno)) + ")");

    loadEEPROM();
    applyReadPattern();
    applyRefreshRate();
}

Driver::~Driver() {
    if (fd_ >= 0) ::close(fd_);
}

void Driver::writeWord(uint16_t reg, uint16_t value) {
    uint8_t buf[4] = {
        uint8_t(reg   >> 8), uint8_t(reg   & 0xFF),
        uint8_t(value >> 8), uint8_t(value & 0xFF)
    };
    if (::write(fd_, buf, 4) != 4)
        throw std::runtime_error(
            "MLX90640: I2C write failed (" + std::string(strerror(errno)) + ")");
}

uint16_t Driver::readWord(uint16_t reg) {
    uint16_t val;
    readWords(reg, &val, 1);
    return val;
}

void Driver::readWords(uint16_t reg, uint16_t* buf, int count) {
    int bytes = count * 2;
    auto rxBuf = std::make_unique<uint8_t[]>(bytes);

    uint8_t regBuf[2] = { uint8_t(reg >> 8), uint8_t(reg & 0xFF) };

    struct i2c_msg msgs[2];
    msgs[0].addr  = addr_;  msgs[0].flags = 0;        msgs[0].len = 2;     msgs[0].buf = regBuf;
    msgs[1].addr  = addr_;  msgs[1].flags = I2C_M_RD; msgs[1].len = bytes; msgs[1].buf = rxBuf.get();

    struct i2c_rdwr_ioctl_data ioctl_data;
    ioctl_data.msgs  = msgs;
    ioctl_data.nmsgs = 2;

    if (::ioctl(fd_, I2C_RDWR, &ioctl_data) < 0)
        throw std::runtime_error(
            "MLX90640: I2C read failed (" + std::string(strerror(errno)) + ")");

    for (int i = 0; i < count; ++i)
        buf[i] = (static_cast<uint16_t>(rxBuf[2*i]) << 8) | rxBuf[2*i+1];
}

void Driver::applyRefreshRate() {
    uint16_t ctrl = readWord(REG_CONTROL);
    ctrl &= ~CTRL_REFRESH_MASK;
    ctrl |= (static_cast<uint16_t>(rate_) << CTRL_REFRESH_SHIFT) & CTRL_REFRESH_MASK;
    writeWord(REG_CONTROL, ctrl);
}

void Driver::applyReadPattern() {
    uint16_t ctrl = readWord(REG_CONTROL);
    if (pattern_ == ReadPattern::Chess)
        ctrl |=  CTRL_CHESS_MODE;
    else
        ctrl &= ~CTRL_CHESS_MODE;
    writeWord(REG_CONTROL, ctrl);
}

void Driver::setRefreshRate(RefreshRate rate) { rate_ = rate; applyRefreshRate(); }
void Driver::setReadPattern(ReadPattern p)    { pattern_ = p; applyReadPattern(); }
RefreshRate Driver::getRefreshRate() const    { return rate_; }
ReadPattern Driver::getReadPattern()  const   { return pattern_; }

void Driver::loadEEPROM() {
    uint16_t ee[EEPROM_SIZE];
    readWords(REG_EEPROM_BASE, ee, EEPROM_SIZE);
    extractCalibration(ee);
}
void Driver::extractCalibration(const uint16_t* ee) {
    CalibrationData& c = cal_;

    // VDD
    int16_t kVddEE  = static_cast<int8_t>((ee[0x0033] & 0xFF00) >> 8);
    c.kVdd  = 32.0f * kVddEE;
    int16_t vdd25EE = static_cast<int16_t>(ee[0x0033] & 0x00FF);
    c.vdd25 = static_cast<float>(((vdd25EE - 256) << 5) - 8192);

    // PTAT
    c.KvPTAT    = static_cast<float>(signExtend((ee[0x0032] & 0xFC00) >> 10, 6)) / 4096.0f;
    c.KtPTAT    = static_cast<float>(signExtend( ee[0x0032] & 0x03FF,        10)) / 8.0f;
    c.vPTAT25   = static_cast<float>(static_cast<int16_t>(ee[0x0031]));
    c.alphaPTAT = static_cast<float>((ee[0x0010] & 0xF000) >> 12) / 4.0f + 8.0f;

    // Gain
    c.gainEE = static_cast<float>(static_cast<int16_t>(ee[0x0030]));

    // TGC — ee[0x003C] bits[7:0]
    c.tgc = static_cast<float>(static_cast<int8_t>(ee[0x003C] & 0x00FF)) / 32.0f;

    // KsTa — ee[0x003C] bits[15:8]
    c.KsTa = static_cast<float>(static_cast<int8_t>((ee[0x003C] & 0xFF00) >> 8)) / 8192.0f;

    // KsTo ranges — ee[0x003F]
    int step = ((ee[0x003F] & 0x3000) >> 12) * 10;
    c.ct[0] = -40.0f;
    c.ct[1] =   0.0f;
    c.ct[2] = static_cast<float>((ee[0x003F] & 0x00F0) >> 4) * step;
    c.ct[3] = static_cast<float>((ee[0x003F] & 0x0F00) >> 8) * step + c.ct[2];
    c.ct[4] = 400.0f;

    int ksToScale = (ee[0x003F] & 0x000F) + 8;
    c.ksTo[0] = static_cast<float>(static_cast<int8_t>( ee[0x003D] & 0x00FF))       / std::pow(2.0f, ksToScale);
    c.ksTo[1] = static_cast<float>(static_cast<int8_t>((ee[0x003D] & 0xFF00) >> 8)) / std::pow(2.0f, ksToScale);
    c.ksTo[2] = static_cast<float>(static_cast<int8_t>( ee[0x003E] & 0x00FF))       / std::pow(2.0f, ksToScale);
    c.ksTo[3] = static_cast<float>(static_cast<int8_t>((ee[0x003E] & 0xFF00) >> 8)) / std::pow(2.0f, ksToScale);
    c.ksTo[4] = -0.0002f;

    // alphaCorrR
    c.alphaCorrR[0] = 1.0f / (1.0f + c.ksTo[0] * 40.0f);
    c.alphaCorrR[1] = 1.0f;
    c.alphaCorrR[2] = 1.0f + c.ksTo[2] * c.ct[2];
    c.alphaCorrR[3] = c.alphaCorrR[2] * (1.0f + c.ksTo[3] * (c.ct[3] - c.ct[2]));

    // CP alpha — ee[0x0039]
    int cpAlphaScale = ((ee[0x0020] & 0xF000) >> 12) + 27;
    c.cpAlpha[0] = static_cast<float>(ee[0x0039] & 0x03FF) / std::pow(2.0f, cpAlphaScale);
    float cpAlphaRatio = 1.0f + static_cast<float>(signExtend((ee[0x0039] & 0xFC00) >> 10, 6)) / 128.0f;
    c.cpAlpha[1] = c.cpAlpha[0] * cpAlphaRatio;

    // CP offset — ee[0x003A] bits[9:0] SP0, bits[15:10] SP1 delta
    c.cpOffset[0] = signExtend(ee[0x003A] & 0x03FF, 10);
    c.cpOffset[1] = signExtend((ee[0x003A] & 0xFC00) >> 10, 6) + c.cpOffset[0];

    // CP Kta — ee[0x003A] bits ... actually from ee[0x003B]
    // Per Melexis ref: cpKta = ee[0x003B] bits[7:0] signed / 2^(cpKtaScale)
    // cpKtaScale from ee[0x003B] bits[11:8]+8 ... wait that's kvScale
    // From reference ExtractCPParameters:
    //   KtaCP = eeData[55] >> 8  (ee[0x0037] high byte? no...)
    // Let's use the reference indexing: eeData[N] where N is 0-based from 0x2400
    // eeData[54] = ee[0x0036], eeData[55] = ee[0x0037]
    // cpKta from eeData[59] = ee[0x003B] bits[7:0] / 2^(scale from eeData[56])
    int cpKtaScale = ((ee[0x003B] & 0x0F00) >> 8) + 8;
    c.cpKta = static_cast<float>(static_cast<int8_t>(ee[0x003B] & 0x00FF)) / std::pow(2.0f, cpKtaScale);

    // CP Kv — ee[0x003B] bits[15:8] / 2^(kvScale+8)
    int cpKvScale = ((ee[0x003B] & 0xF000) >> 12) + 8;
    c.cpKv = static_cast<float>(static_cast<int8_t>((ee[0x003B] & 0xFF00) >> 8)) / std::pow(2.0f, cpKvScale);

    // Resolution — ee[0x0038] bits[11:10]
    c.resolutionEE      = (ee[0x0038] & 0x0C00) >> 10;
    c.calibrationModeEE = (ee[0x0010] & 0x0800) >> 11;

    // IL chess correction — ee[0x0007], ee[0x0008], ee[0x0009]
    // Per Melexis ref ExtractIlChessC: eeData[7..9]
    c.ilChessC[0] = static_cast<float>(signExtend( ee[0x0007] & 0x003F,        6)) / 16.0f;
    c.ilChessC[1] = static_cast<float>(signExtend((ee[0x0007] & 0x07C0) >> 6,  5)) / 2.0f;
    c.ilChessC[2] = static_cast<float>(signExtend((ee[0x0007] & 0xF800) >> 11, 5)) / 8.0f;

    // Per-pixel scales
    int occScaleRow = (ee[0x0010] & 0x0F00) >> 8;
    int occScaleCol = (ee[0x0010] & 0x00F0) >> 4;
    int occScaleRem = (ee[0x0010] & 0x000F);

    int accScaleRem = (ee[0x0020] & 0x000F);
    int accScaleCol = (ee[0x0020] & 0x00F0) >> 4;
    int accScaleRow = (ee[0x0020] & 0x0F00) >> 8;
    int alphaScale  = ((ee[0x0020] & 0xF000) >> 12) + 30;
    int alphaRef    = static_cast<int>(static_cast<int16_t>(ee[0x0021]));

    // pixOsAverage — ee[0x0011]
    int16_t pixOsAverage = static_cast<int16_t>(ee[0x0011]);

    // occRow: ee[0x0012..0x0017]
    int occRow[24], occCol[32];
    for (int i = 0; i < 6; ++i) {
        uint16_t w = ee[0x0012 + i];
        occRow[i*4+0] = signExtend( w        & 0x000F, 4);
        occRow[i*4+1] = signExtend((w >>  4) & 0x000F, 4);
        occRow[i*4+2] = signExtend((w >>  8) & 0x000F, 4);
        occRow[i*4+3] = signExtend((w >> 12) & 0x000F, 4);
    }

    // occCol: ee[0x0018..0x001F]
    for (int i = 0; i < 8; ++i) {
        uint16_t w = ee[0x0018 + i];
        occCol[i*4+0] = signExtend( w        & 0x000F, 4);
        occCol[i*4+1] = signExtend((w >>  4) & 0x000F, 4);
        occCol[i*4+2] = signExtend((w >>  8) & 0x000F, 4);
        occCol[i*4+3] = signExtend((w >> 12) & 0x000F, 4);
    }

    // accRow: ee[0x0022..0x0027]
    int accRow[24], accCol[32];
    for (int i = 0; i < 6; ++i) {
        uint16_t w = ee[0x0022 + i];
        accRow[i*4+0] = signExtend( w        & 0x000F, 4);
        accRow[i*4+1] = signExtend((w >>  4) & 0x000F, 4);
        accRow[i*4+2] = signExtend((w >>  8) & 0x000F, 4);
        accRow[i*4+3] = signExtend((w >> 12) & 0x000F, 4);
    }

    // accCol: ee[0x0028..0x002F]
    for (int i = 0; i < 8; ++i) {
        uint16_t w = ee[0x0028 + i];
        accCol[i*4+0] = signExtend( w        & 0x000F, 4);
        accCol[i*4+1] = signExtend((w >>  4) & 0x000F, 4);
        accCol[i*4+2] = signExtend((w >>  8) & 0x000F, 4);
        accCol[i*4+3] = signExtend((w >> 12) & 0x000F, 4);
    }

    // Kta/Kv scales — ee[0x003B]
    int ktaScale1 = ((ee[0x003B] & 0x00F0) >> 4) + 8;
    int ktaScale2 =  (ee[0x003B] & 0x000F);
    int kvScale   = ((ee[0x003B] & 0xF000) >> 12) + 8;

    // KtaRC: ee[0x0036..0x0037]
    float ktaRC[4];
    ktaRC[0] = static_cast<float>(static_cast<int8_t>((ee[0x0036] & 0xFF00) >> 8)) / std::pow(2.0f, ktaScale1);
    ktaRC[2] = static_cast<float>(static_cast<int8_t>( ee[0x0036] & 0x00FF))       / std::pow(2.0f, ktaScale1);
    ktaRC[1] = static_cast<float>(static_cast<int8_t>((ee[0x0037] & 0xFF00) >> 8)) / std::pow(2.0f, ktaScale1);
    ktaRC[3] = static_cast<float>(static_cast<int8_t>( ee[0x0037] & 0x00FF))       / std::pow(2.0f, ktaScale1);

    // KvRC: ee[0x0034..0x0035]
    float kvRC[4];
    kvRC[0] = static_cast<float>(static_cast<int8_t>((ee[0x0034] & 0xFF00) >> 8)) / std::pow(2.0f, kvScale);
    kvRC[2] = static_cast<float>(static_cast<int8_t>( ee[0x0034] & 0x00FF))       / std::pow(2.0f, kvScale);
    kvRC[1] = static_cast<float>(static_cast<int8_t>((ee[0x0035] & 0xFF00) >> 8)) / std::pow(2.0f, kvScale);
    kvRC[3] = static_cast<float>(static_cast<int8_t>( ee[0x0035] & 0x00FF))       / std::pow(2.0f, kvScale);

    // Per-pixel loop
    for (int i = 0; i < PIXEL_COUNT; ++i) {
        int row = i / 32;
        int col = i % 32;
        uint16_t w = ee[0x0040 + i];

        int pixOsEE = signExtend((w & 0xFC00) >> 10, 6);
        c.offset[i] = static_cast<float>(
            pixOsEE       * (1 << occScaleRem)
            + occRow[row] * (1 << occScaleRow)
            + occCol[col] * (1 << occScaleCol)
            + pixOsAverage);

        int pixAlphaEE = (w & 0x03F0) >> 4;
        float alphaTemp = static_cast<float>(
            pixAlphaEE    * (1 << accScaleRem)
            + accRow[row] * (1 << accScaleRow)
            + accCol[col] * (1 << accScaleCol)
            + alphaRef);
        alphaTemp /= std::pow(2.0f, alphaScale);
        alphaTemp -= c.tgc * (c.cpAlpha[0] + c.cpAlpha[1]) / 2.0f;
        c.alpha[i] = alphaTemp;

        int ktaEE = signExtend((w & 0x000E) >> 1, 3);
        int rc    = (col & 1) | ((row & 1) << 1);
        c.kta[i]  = ktaRC[rc] + static_cast<float>(ktaEE) / std::pow(2.0f, ktaScale1 + ktaScale2);

        c.kv[i] = kvRC[rc];

        c.pixSensitivity[i] = w & 0x0001;
    }
}

int Driver::getRawFrame(uint16_t* frameData, int timeoutMs) {
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(timeoutMs);

    writeWord(REG_STATUS, 0x0000);

    while (clock::now() < deadline) {
        uint16_t status = readWord(REG_STATUS);
        if (status & STATUS_NEW_DATA) {
            int subpage = status & STATUS_SUBPAGE;

            readWords(0x0400, frameData,       768);
            readWords(0x0700, frameData + 768,  64);
            frameData[832] = readWord(REG_CONTROL);
            frameData[833] = static_cast<uint16_t>(subpage);

            writeWord(REG_STATUS, 0x0000);
            return subpage;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return -1;
}

int Driver::getFrame(float* temperatures) {
    uint16_t frameData[FRAME_DATA_SIZE];
    int subpage = getRawFrame(frameData);
    if (subpage < 0)
        throw std::runtime_error("MLX90640: timeout waiting for frame");
    rawToTemperatures(frameData, temperatures);
    return subpage;
}

float Driver::computeVdd(const uint16_t* frameData) const {
    int   resolutionRAM = (frameData[832] >> 10) & 0x3;
    float resCorr       = std::pow(2.0f, cal_.resolutionEE) / std::pow(2.0f, resolutionRAM);
    float vddRam        = static_cast<float>(static_cast<int16_t>(frameData[810]));
    return (resCorr * vddRam - cal_.vdd25) / cal_.kVdd + 3.3f;
}

float Driver::computeTa(const uint16_t* frameData) const {
    float vdd      = computeVdd(frameData);
    float vbeTa    = static_cast<float>(static_cast<int16_t>(frameData[768]));
    float ptatRam  = static_cast<float>(static_cast<int16_t>(frameData[800]));
    float vPTATart = (ptatRam / (ptatRam * cal_.alphaPTAT + vbeTa)) * 262144.0f;
    return (vPTATart / (1.0f + cal_.KvPTAT * (vdd - 3.3f)) - cal_.vPTAT25) / cal_.KtPTAT + 25.0f;
}

void Driver::rawToTemperatures(const uint16_t* frameData,
                               float*          temperatures,
                               float           emissivity) const
{
    const CalibrationData& c = cal_;

    float vdd = computeVdd(frameData);
    float Ta  = computeTa(frameData);
    Ta_       = Ta;

    float ta4 = (Ta + 273.15f); ta4 *= ta4; ta4 *= ta4;

    float gainRam = static_cast<float>(static_cast<int16_t>(frameData[778]));
    float gain    = c.gainEE / gainRam;

    int     subPage = frameData[833] & 0x1;
    uint8_t mode    = (frameData[832] & CTRL_CHESS_MODE) ? 1 : 0;

    // CP
    float irDataCP[2];
    irDataCP[0] = static_cast<float>(static_cast<int16_t>(frameData[776])) * gain;
    irDataCP[1] = static_cast<float>(static_cast<int16_t>(frameData[777])) * gain;

    if (mode == c.calibrationModeEE) {
        irDataCP[0] -= c.cpOffset[0] * (1.0f + c.cpKta * (Ta - 25.0f)) * (1.0f + c.cpKv * (vdd - 3.3f));
        irDataCP[1] -= c.cpOffset[1] * (1.0f + c.cpKta * (Ta - 25.0f)) * (1.0f + c.cpKv * (vdd - 3.3f));
    } else {
        irDataCP[0] -= c.cpOffset[0] * (1.0f + c.cpKta * (Ta - 25.0f)) * (1.0f + c.cpKv * (vdd - 3.3f));
        irDataCP[1] -= (c.cpOffset[1] + c.ilChessC[0]) * (1.0f + c.cpKta * (Ta - 25.0f)) * (1.0f + c.cpKv * (vdd - 3.3f));
    }

    for (int pixelNumber = 0; pixelNumber < PIXEL_COUNT; ++pixelNumber) {
        int row = pixelNumber / 32;
        int col = pixelNumber % 32;

        int8_t ilPattern    = (row >> 1) - ((row >> 2) << 1);
        int8_t chessPattern = ilPattern ^ (col & 1);
        int8_t pattern      = (mode == 0) ? ilPattern : chessPattern;

        // Only update pixels belonging to this subpage
        if (pattern != subPage) continue;

        float irData = static_cast<float>(static_cast<int16_t>(frameData[pixelNumber])) * gain;

        irData -= c.offset[pixelNumber]
                  * (1.0f + c.kta[pixelNumber] * (Ta  - 25.0f))
                  * (1.0f + c.kv[pixelNumber]  * (vdd - 3.3f));

        if (mode != c.calibrationModeEE) {
            int8_t ilChessPattern = (row & 1) ^ (col & 1);
            irData += c.ilChessC[2] * (2.0f * ilChessPattern - 1.0f);
        }

        irData -= c.tgc * irDataCP[subPage];
        irData /= emissivity;

        float alphaCompensated = c.alpha[pixelNumber] * (1.0f + c.KsTa * (Ta - 25.0f));

        float Sx = alphaCompensated * alphaCompensated * alphaCompensated
                 * (irData + alphaCompensated * ta4);
        Sx = std::sqrt(std::sqrt(std::abs(Sx))) * c.ksTo[1];
        if (irData + alphaCompensated * ta4 < 0.0f) Sx = -Sx;

        float denom = alphaCompensated * (1.0f - c.ksTo[1] * 273.15f) + Sx;
        float To4   = irData / denom + ta4;
        float To    = (To4 > 0.0f) ? std::sqrt(std::sqrt(To4)) - 273.15f : -273.15f;

        int8_t range;
        if      (To < c.ct[1]) range = 0;
        else if (To < c.ct[2]) range = 1;
        else if (To < c.ct[3]) range = 2;
        else                   range = 3;

        float To4ref = irData / (alphaCompensated * c.alphaCorrR[range]
                                 * (1.0f + c.ksTo[range] * (To - c.ct[range])))
                       + ta4;
        To = (To4ref > 0.0f) ? std::sqrt(std::sqrt(To4ref)) - 273.15f : -273.15f;

        temperatures[pixelNumber] = To;
    }
}

} // namespace MLX90640