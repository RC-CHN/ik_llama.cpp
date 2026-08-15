#include "iqk_gemm_amx.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#if defined(GGML_AMX_INT8) && defined(__AMX_TILE__) && defined(__AMX_INT8__) && defined(__x86_64__)

#include <algorithm>
#include <array>
#include <atomic>
#include <cpuid.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

#define IQK_ARCH_REQ_XCOMP_PERM 0x1023
#define IQK_XFEATURE_XTILEDATA  18

struct amx_tile_config {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

static_assert(sizeof(amx_tile_config) == 64, "AMX tile configuration must be 64 bytes");

struct amx_int8_stats {
    std::atomic<uint64_t> q4_calls  {0};
    std::atomic<uint64_t> q4_tiles  {0};
    std::atomic<uint64_t> q5_calls  {0};
    std::atomic<uint64_t> q5_tiles  {0};
    std::atomic<uint64_t> q6_calls  {0};
    std::atomic<uint64_t> q6_tiles  {0};
    std::atomic<uint64_t> iq3_calls {0};
    std::atomic<uint64_t> iq3_tiles {0};
    std::atomic<uint64_t> q8_calls  {0};
    std::atomic<uint64_t> q8_tiles  {0};
    std::atomic<uint64_t> iq4_calls {0};
    std::atomic<uint64_t> iq4_tiles {0};
    std::atomic<uint64_t> prepack_builds {0};
    std::atomic<uint64_t> prepack_bytes  {0};

    ~amx_int8_stats() {
        const char * enabled = std::getenv("GGML_AMX_STATS");
        if (!enabled || enabled[0] == '\0' || enabled[0] == '0') {
            return;
        }
        std::fprintf(stderr,
                "ggml_amx_int8: q4_k_r4 calls=%llu tiles=%llu; q5_k_r4 calls=%llu tiles=%llu; q6_k_r4 calls=%llu tiles=%llu; iq3_s_r4 calls=%llu tiles=%llu; q8_0_r8 calls=%llu tiles=%llu; iq4_nl_r4 calls=%llu tiles=%llu; prepack builds=%llu bytes=%llu\n",
                (unsigned long long) q4_calls.load(),
                (unsigned long long) q4_tiles.load(),
                (unsigned long long) q5_calls.load(),
                (unsigned long long) q5_tiles.load(),
                (unsigned long long) q6_calls.load(),
                (unsigned long long) q6_tiles.load(),
                (unsigned long long) iq3_calls.load(),
                (unsigned long long) iq3_tiles.load(),
                (unsigned long long) q8_calls.load(),
                (unsigned long long) q8_tiles.load(),
                (unsigned long long) iq4_calls.load(),
                (unsigned long long) iq4_tiles.load(),
                (unsigned long long) prepack_builds.load(),
                (unsigned long long) prepack_bytes.load());
    }
};

amx_int8_stats & stats() {
    static amx_int8_stats value;
    return value;
}

static bool stats_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_AMX_STATS");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

thread_local bool transient_weights = false;

struct weight_origin_context {
    const void * base = nullptr;
    int row_offset = 0;
    int total_rows = 0;
};

thread_local weight_origin_context weight_origin;

static inline bool env_is_set(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

static int iq4_min_nrc_y() {
    static const int value = [] {
        const char * text = std::getenv("GGML_AMX_IQ4_NL_MIN_NY");
        return text && text[0] ? std::max(4, std::atoi(text)) : 16;
    }();
    return value;
}

static int iq4_min_k() {
    static const int value = [] {
        const char * text = std::getenv("GGML_AMX_IQ4_NL_MIN_K");
        return text && text[0] ? std::max(QK4_NL, std::atoi(text)) : 1024;
    }();
    return value;
}

static int iq3_min_nrc_y() {
    static const int value = [] {
        const char * text = std::getenv("GGML_AMX_IQ3_S_MIN_NY");
        return text && text[0] ? std::max(1, std::atoi(text)) : 8;
    }();
    return value;
}

static int q8_0_min_nrc_y() {
    static const int value = [] {
        const char * text = std::getenv("GGML_AMX_Q8_0_R8_MIN_NY");
        return text && text[0] ? std::max(1, std::atoi(text)) : 16;
    }();
    return value;
}

static int q6_min_nrc_y() {
    static const int value = [] {
        const char * text = std::getenv("GGML_AMX_Q6_K_MIN_NY");
        return text && text[0] ? std::max(4, std::atoi(text)) : 16;
    }();
    return value;
}

static int q5_min_nrc_y() {
    static const int value = [] {
        const char * text = std::getenv("GGML_AMX_Q5_K_MIN_NY");
        return text && text[0] ? std::max(4, std::atoi(text)) : 16;
    }();
    return value;
}

static bool q4_claim_expanded_cache(size_t extra_bytes) {
    if (env_is_set("GGML_AMX_Q4_K_EXPANDED")) {
        return true;
    }
    static const uint64_t budget = [] {
        const char * text = std::getenv("GGML_AMX_Q4_K_EXPANDED_BUDGET_MB");
        if (!text || !text[0]) return uint64_t{0};
        return static_cast<uint64_t>(std::strtoull(text, nullptr, 10))*1024u*1024u;
    }();
    if (budget == 0) {
        return false;
    }
    static std::atomic<uint64_t> claimed {0};
    uint64_t current = claimed.load(std::memory_order_relaxed);
    while (current + extra_bytes <= budget) {
        if (claimed.compare_exchange_weak(current, current + extra_bytes,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

// R4 stores four logical rows together. Recover one 32-value row from the
// nibble-interleaved representation produced by repack_iq4_nl/repack_q4_k.
static inline void unpack_r4_nibbles(
        const uint8_t * src, int row, uint8_t * dst, const uint8_t * values) {
    for (int i = 0; i < 4; ++i) {
        const uint8_t b0 = src[4*row + i +  0];
        const uint8_t b1 = src[4*row + i + 16];
        const uint8_t b2 = src[4*row + i + 32];
        const uint8_t b3 = src[4*row + i + 48];
        dst[i +  0] = values[b0 & 0x0f];
        dst[i +  8] = values[b0 >> 4];
        dst[i + 16] = values[b1 & 0x0f];
        dst[i + 24] = values[b1 >> 4];
        dst[i +  4] = values[b2 & 0x0f];
        dst[i + 12] = values[b2 >> 4];
        dst[i + 20] = values[b3 & 0x0f];
        dst[i + 28] = values[b3 >> 4];
    }
}

static inline void unpack_q6_r4_32(
        const block_q6_k_r4 & block, int ib32, int row, uint8_t * dst) {
    const uint8_t * ql = block.ql + 64*ib32 + 4*row;
    const uint8_t * qh = block.qh + 32*ib32 + 4*row;
    for (int i = 0; i < 4; ++i) {
        const uint8_t l0 = ql[i +  0];
        const uint8_t l1 = ql[i + 16];
        const uint8_t l2 = ql[i + 32];
        const uint8_t l3 = ql[i + 48];
        const uint8_t h0 = qh[i +  0];
        const uint8_t h1 = qh[i + 16];
        dst[i +  0] = (l0 & 0x0f) | ((h0 & 0x03) << 4);
        dst[i +  8] = (l0 >> 4)   | (((h0 >> 2) & 0x03) << 4);
        dst[i +  4] = (l2 & 0x0f) | (((h0 >> 4) & 0x03) << 4);
        dst[i + 12] = (l2 >> 4)   | (((h0 >> 6) & 0x03) << 4);
        dst[i + 16] = (l1 & 0x0f) | ((h1 & 0x03) << 4);
        dst[i + 24] = (l1 >> 4)   | (((h1 >> 2) & 0x03) << 4);
        dst[i + 20] = (l3 & 0x0f) | (((h1 >> 4) & 0x03) << 4);
        dst[i + 28] = (l3 >> 4)   | (((h1 >> 6) & 0x03) << 4);
    }
}

static inline void unpack_q5_r4_32(
        const block_q5_k_r4 & block, int ib32, int row, uint8_t * dst) {
    const uint8_t * ql = block.qs + 64*ib32 + 4*row;
    const uint8_t * qh = block.qh + 16*ib32 + 4*row;
    for (int i = 0; i < 4; ++i) {
        const uint8_t l0 = ql[i +  0];
        const uint8_t l1 = ql[i + 16];
        const uint8_t l2 = ql[i + 32];
        const uint8_t l3 = ql[i + 48];
        const uint8_t h = qh[i];
        dst[i +  0] = (l0 & 0x0f) | ((h << 4) & 0x10);
        dst[i +  8] = (l0 >> 4)   | ((h << 3) & 0x10);
        dst[i +  4] = (l2 & 0x0f) | ((h << 2) & 0x10);
        dst[i + 12] = (l2 >> 4)   | ((h << 1) & 0x10);
        dst[i + 16] = (l1 & 0x0f) | ((h >> 0) & 0x10);
        dst[i + 24] = (l1 >> 4)   | ((h >> 1) & 0x10);
        dst[i + 20] = (l3 & 0x0f) | ((h >> 2) & 0x10);
        dst[i + 28] = (l3 >> 4)   | ((h >> 3) & 0x10);
    }
}

// AMX weights stay at four bits in memory.  Two logical rows are unpacked at
// once so the extra byte traffic exists only between L1 and the tile register.
static inline void unpack_q4_tile(
        const uint8_t * src, int nrows, uint8_t * dst) {
    const __m256i mask = _mm256_set1_epi8(0x0f);
    for (int row = 0; row < nrows; row += 2) {
        const __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(src + static_cast<size_t>(row)*16));
        const __m256i lo = _mm256_and_si256(packed, mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst + static_cast<size_t>(row + 0)*32),
                           _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst + static_cast<size_t>(row + 1)*32),
                           _mm256_permute2x128_si256(lo, hi, 0x31));
    }
}

static inline void unpack_iq4_tile(
        const uint8_t * src, int nrows, uint8_t * dst) {
    alignas(16) static constexpr uint8_t values[16] = {
        1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241,
    };
    const __m256i table = _mm256_broadcastsi128_si256(
        _mm_load_si128(reinterpret_cast<const __m128i *>(values)));
    const __m256i mask = _mm256_set1_epi8(0x0f);
    for (int row = 0; row < nrows; row += 2) {
        const __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(src + static_cast<size_t>(row)*16));
        const __m256i lo = _mm256_shuffle_epi8(table, _mm256_and_si256(packed, mask));
        const __m256i hi = _mm256_shuffle_epi8(table,
            _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask));
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst + static_cast<size_t>(row + 0)*32),
                           _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst + static_cast<size_t>(row + 1)*32),
                           _mm256_permute2x128_si256(lo, hi, 0x31));
    }
}

static inline void pack_b_vnni(
        const int8_t * const * rows, int nrc_y, int8_t * packed) {
    for (int ik = 0; ik < 8; ++ik) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            std::memcpy(packed + (ik*nrc_y + iy)*4, rows[iy] + 4*ik, 4);
        }
    }
}

static inline int sum_i8_32(const int8_t * values) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_set1_epi8(1), q);
    const __m256i quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(quads),
                               _mm256_extracti128_si256(quads, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline void configure_tiles(int nrc_x, int nrc_y) {
    alignas(64) amx_tile_config config = {};
    config.palette_id = 1;
    config.rows[0] = nrc_x;
    config.colsb[0] = std::min(16, nrc_y) * sizeof(int32_t); // C0: M x N0
    config.rows[1] = nrc_x;
    config.colsb[1] = 32;                      // A: M x K uint8
    config.rows[2] = 8;
    config.colsb[2] = std::min(16, nrc_y) * 4; // B0: K/4 x N0 x 4
    if (nrc_y > 16) {
        config.rows[3] = nrc_x;
        config.colsb[3] = (nrc_y - 16) * sizeof(int32_t); // C1
        config.rows[4] = 8;
        config.colsb[4] = (nrc_y - 16) * 4;               // B1
    }
    _tile_loadconfig(&config);
}

static inline void amx_dot_tile(
        const uint8_t * a, const int8_t * b0, const int8_t * b1,
        int32_t * c, int nrc_y) {
    _tile_zero(0);
    _tile_loadd(1, a, 32);
    _tile_loadd(2, b0, std::min(16, nrc_y)*4);
    _tile_dpbusd(0, 1, 2);
    _tile_stored(0, c, nrc_y*sizeof(int32_t));
    if (nrc_y > 16) {
        _tile_zero(3);
        _tile_loadd(4, b1, (nrc_y - 16)*4);
        _tile_dpbusd(3, 1, 4);
        _tile_stored(3, c + 16, nrc_y*sizeof(int32_t));
    }
}

static inline void configure_tiles_q4_transposed(int nrows0, int nrows1, int ncols) {
    alignas(64) amx_tile_config config = {};
    config.palette_id = 1;
    config.rows[0] = nrows0; config.colsb[0] = ncols*4; // C0
    config.rows[2] = nrows0; config.colsb[2] = 32;      // activation A0
    config.rows[3] = 8;      config.colsb[3] = ncols*4; // VNNI weight B
    if (nrows1) {
        config.rows[1] = nrows1; config.colsb[1] = ncols*4; // C1
        config.rows[4] = nrows1; config.colsb[4] = 32;      // activation A1
    }
    _tile_loadconfig(&config);
}

static inline void unpack_q4_vnni(const uint8_t * src, uint8_t * dst) {
    const __m512i mask = _mm512_set1_epi8(0x0f);
    for (int pair = 0; pair < 4; ++pair) {
        const __m512i packed = _mm512_loadu_si512(
            reinterpret_cast<const __m512i *>(src + pair*64));
        _mm512_store_si512(reinterpret_cast<__m512i *>(dst + (2*pair + 0)*64),
                           _mm512_and_si512(packed, mask));
        _mm512_store_si512(reinterpret_cast<__m512i *>(dst + (2*pair + 1)*64),
                           _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask));
    }
}

static inline void unpack_q4_high2_vnni(const uint8_t * src, uint8_t * dst) {
    const __m512i one = _mm512_set1_epi8(1);
    const __m512i two = _mm512_set1_epi8(2);
    for (int row = 0; row < 16; ++row) {
        uint64_t bits0;
        uint64_t bits1;
        std::memcpy(&bits0, src + row*16 + 0, sizeof(bits0));
        std::memcpy(&bits1, src + row*16 + 8, sizeof(bits1));
        const __m512i unpacked = _mm512_or_si512(
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits0), one),
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits1), two));
        _mm512_store_si512(reinterpret_cast<__m512i *>(dst + row*64), unpacked);
    }
}

static inline void unpack_q5_high3_vnni(const uint8_t * src, uint8_t * dst) {
    const __m512i one  = _mm512_set1_epi8(1);
    const __m512i two  = _mm512_set1_epi8(2);
    const __m512i four = _mm512_set1_epi8(4);
    for (int row = 0; row < 16; ++row) {
        uint64_t bits0;
        uint64_t bits1;
        uint64_t bits2;
        std::memcpy(&bits0, src + row*24 +  0, sizeof(bits0));
        std::memcpy(&bits1, src + row*24 +  8, sizeof(bits1));
        std::memcpy(&bits2, src + row*24 + 16, sizeof(bits2));
        __m512i unpacked = _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits0), one);
        unpacked = _mm512_or_si512(unpacked,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits1), two));
        unpacked = _mm512_or_si512(unpacked,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits2), four));
        _mm512_store_si512(reinterpret_cast<__m512i *>(dst + row*64), unpacked);
    }
}

static inline void unpack_q6_high5_vnni(const uint8_t * src, uint8_t * dst) {
    const __m512i bit0 = _mm512_set1_epi8(0x01);
    const __m512i bit1 = _mm512_set1_epi8(0x02);
    const __m512i bit2 = _mm512_set1_epi8(0x04);
    const __m512i bit3 = _mm512_set1_epi8(0x08);
    const __m512i bit4 = _mm512_set1_epi8(0x10);
    for (int row = 0; row < 8; ++row) {
        uint64_t bits[5];
        std::memcpy(bits, src + row*5*sizeof(uint64_t), sizeof(bits));
        __m512i unpacked = _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits[0]), bit0);
        unpacked = _mm512_or_si512(unpacked,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits[1]), bit1));
        unpacked = _mm512_or_si512(unpacked,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits[2]), bit2));
        unpacked = _mm512_or_si512(unpacked,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits[3]), bit3));
        unpacked = _mm512_or_si512(unpacked,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits[4]), bit4));
        _mm512_store_si512(reinterpret_cast<__m512i *>(dst + row*64), unpacked);
    }
}

static inline __m512i pack_q6_u16_pair(__m512i lo, __m512i hi) {
    __m512i result = _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(lo));
    return _mm512_inserti64x4(result, _mm512_cvtusepi16_epi8(hi), 1);
}

static inline __m512i pack_i16_pair(__m512i lo, __m512i hi) {
    __m512i result = _mm512_castsi256_si512(_mm512_cvtsepi16_epi8(lo));
    return _mm512_inserti64x4(result, _mm512_cvtsepi16_epi8(hi), 1);
}

static inline void encode_iq3_scaled_vnni(
        const int8_t * q, const int8_t * scales, int8_t * low, int8_t * high) {
    alignas(64) static constexpr uint8_t repeat_scale[64] = {
         0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,
         4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7,
         8, 8, 8, 8,  9, 9, 9, 9, 10,10,10,10, 11,11,11,11,
        12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15,
    };
    const __m512i shuffle = _mm512_load_si512(
        reinterpret_cast<const __m512i *>(repeat_scale));

    for (int group = 0; group < 2; ++group) {
        const __m128i scale16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(scales + 16*group));
        const __m512i scale8 = _mm512_shuffle_epi8(
            _mm512_broadcast_i32x4(scale16), shuffle);
        const __m512i scale_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(scale8));
        const __m512i scale_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(scale8, 1));
        for (int kr = 0; kr < 8; ++kr) {
            const int row = 8*group + kr;
            const __m512i q8 = _mm512_loadu_si512(
                reinterpret_cast<const __m512i *>(q + 64*row));
            const __m512i product_lo = _mm512_mullo_epi16(
                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(q8)), scale_lo);
            const __m512i product_hi = _mm512_mullo_epi16(
                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q8, 1)), scale_hi);
            // floor(product/128) leaves an exact signed-byte remainder in
            // [0, 127], so product = low + 128*high without a correction term.
            const __m512i high_lo = _mm512_srai_epi16(product_lo, 7);
            const __m512i high_hi = _mm512_srai_epi16(product_hi, 7);
            const __m512i low_lo = _mm512_sub_epi16(product_lo, _mm512_slli_epi16(high_lo, 7));
            const __m512i low_hi = _mm512_sub_epi16(product_hi, _mm512_slli_epi16(high_hi, 7));
            _mm512_store_si512(reinterpret_cast<__m512i *>(low + 64*row),
                pack_i16_pair(low_lo, low_hi));
            _mm512_store_si512(reinterpret_cast<__m512i *>(high + 64*row),
                pack_i16_pair(high_lo, high_hi));
        }
    }
}

static inline void encode_q6_scaled_vnni(
        const int8_t * q, const int8_t * scales, uint8_t * low, uint8_t * high) {
    alignas(64) static constexpr uint8_t repeat_scale[64] = {
         0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,
         4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7,
         8, 8, 8, 8,  9, 9, 9, 9, 10,10,10,10, 11,11,11,11,
        12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15,
    };
    const __m512i shuffle = _mm512_load_si512(
        reinterpret_cast<const __m512i *>(repeat_scale));
    const __m512i bias = _mm512_set1_epi16(255);
    const __m512i byte_mask = _mm512_set1_epi16(255);
    const __m512i high_bias = _mm512_set1_epi16(15);

    for (int k16 = 0; k16 < 4; ++k16) {
        const __m128i scale16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(scales + 16*k16));
        const __m512i scale8 = _mm512_shuffle_epi8(
            _mm512_broadcast_i32x4(scale16), shuffle);
        const __m512i scale_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(scale8));
        const __m512i scale_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(scale8, 1));

        for (int kr = 0; kr < 4; ++kr) {
            const int row = 4*k16 + kr;
            const __m512i q8 = _mm512_loadu_si512(
                reinterpret_cast<const __m512i *>(q + 64*row));
            const __m512i product_lo = _mm512_mullo_epi16(
                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(q8)), scale_lo);
            const __m512i product_hi = _mm512_mullo_epi16(
                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q8, 1)), scale_hi);
            const __m512i biased_lo = _mm512_add_epi16(product_lo, bias);
            const __m512i biased_hi = _mm512_add_epi16(product_hi, bias);
            _mm512_store_si512(reinterpret_cast<__m512i *>(low + 64*row),
                pack_q6_u16_pair(_mm512_and_si512(biased_lo, byte_mask),
                                  _mm512_and_si512(biased_hi, byte_mask)));
            _mm512_store_si512(reinterpret_cast<__m512i *>(high + 64*row),
                pack_q6_u16_pair(_mm512_add_epi16(_mm512_srai_epi16(biased_lo, 8), high_bias),
                                  _mm512_add_epi16(_mm512_srai_epi16(biased_hi, 8), high_bias)));
        }
    }
}

static inline void amx_compute_q4_transposed(
        const int8_t * a0, const int8_t * a1, const uint8_t * b,
        int nrows1, int a_stride) {
    _tile_zero(0);
    _tile_loadd(2, a0, a_stride);
    _tile_loadd(3, b, 64);
    _tile_dpbsud(0, 2, 3);
    if (nrows1) {
        _tile_zero(1);
        _tile_loadd(4, a1, a_stride);
        _tile_dpbsud(1, 4, 3);
    }
}

static inline void amx_store_q4_transposed(int32_t * c, int nrows1) {
    _tile_stored(0, c, 16*sizeof(int32_t));
    if (nrows1) {
        _tile_stored(1, c + 16*16, 16*sizeof(int32_t));
    }
}

static inline void store_q4_scaled_parts(
        __m512i q, __m512i scale_lo, __m512i scale_hi,
        uint8_t * low, uint8_t * high) {
    const __m512i p0 = _mm512_mullo_epi16(
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(q)),
        scale_lo);
    const __m512i p1 = _mm512_mullo_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(q, 1)),
        scale_hi);
    const __m512i low0 = _mm512_and_si512(p0, _mm512_set1_epi16(0xff));
    const __m512i low1 = _mm512_and_si512(p1, _mm512_set1_epi16(0xff));
    __m512i low8 = _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(low0));
    low8 = _mm512_inserti64x4(low8, _mm512_cvtusepi16_epi8(low1), 1);
    __m512i high8 = _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(_mm512_srli_epi16(p0, 8)));
    high8 = _mm512_inserti64x4(high8, _mm512_cvtusepi16_epi8(_mm512_srli_epi16(p1, 8)), 1);
    _mm512_store_si512(reinterpret_cast<__m512i *>(low), low8);
    _mm512_store_si512(reinterpret_cast<__m512i *>(high), high8);
}

static inline void unpack_scaled_q4_vnni(
        const uint8_t * src, const uint8_t * scales, uint8_t * low, uint8_t * high) {
    alignas(64) static constexpr uint8_t repeat_scale[64] = {
         0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,
         4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7,
         8, 8, 8, 8,  9, 9, 9, 9, 10,10,10,10, 11,11,11,11,
        12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15,
    };
    const __m512i shuffle = _mm512_load_si512(
        reinterpret_cast<const __m512i *>(repeat_scale));
    const __m512i vscale = _mm512_permutexvar_epi8(shuffle,
        _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(scales))));
    const __m512i scale_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(vscale));
    const __m512i scale_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(vscale, 1));
    const __m512i mask = _mm512_set1_epi8(0x0f);
    for (int pair = 0; pair < 4; ++pair) {
        const __m512i packed = _mm512_loadu_si512(
            reinterpret_cast<const __m512i *>(src + pair*64));
        const __m512i q0 = _mm512_and_si512(packed, mask);
        const __m512i q1 = _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask);
        store_q4_scaled_parts(q0, scale_lo, scale_hi,
            low + (2*pair + 0)*64, high + (2*pair + 0)*64);
        store_q4_scaled_parts(q1, scale_lo, scale_hi,
            low + (2*pair + 1)*64, high + (2*pair + 1)*64);
    }
}

static inline void configure_tiles_q4_scaled(int nrows0, int nrows1, int ncols) {
    alignas(64) amx_tile_config config = {};
    config.palette_id = 1;
    config.rows[0] = nrows0; config.colsb[0] = ncols*4; // low C0
    config.rows[2] = nrows0; config.colsb[2] = ncols*4; // high C0
    config.rows[4] = nrows0; config.colsb[4] = 64;      // activation A0, K64
    config.rows[6] = 16;     config.colsb[6] = ncols*4; // VNNI B, K64/4
    if (nrows1) {
        config.rows[1] = nrows1; config.colsb[1] = ncols*4; // low C1
        config.rows[3] = nrows1; config.colsb[3] = ncols*4; // high C1
        config.rows[5] = nrows1; config.colsb[5] = 64;      // activation A1, K64
    }
    _tile_loadconfig(&config);
}

static inline void configure_tiles_iq3_unscaled(int nrows0, int nrows1, int ncols) {
    alignas(64) amx_tile_config config = {};
    config.palette_id = 1;
    config.rows[0] = nrows0; config.colsb[0] = ncols*4; // C0
    config.rows[4] = nrows0; config.colsb[4] = 32;      // activation A0, K32
    config.rows[6] = 8;      config.colsb[6] = ncols*4; // VNNI B, K32/4
    if (nrows1) {
        config.rows[1] = nrows1; config.colsb[1] = ncols*4; // C1
        config.rows[5] = nrows1; config.colsb[5] = 32;      // activation A1, K32
    }
    _tile_loadconfig(&config);
}

struct prepack_key {
    const void * vx;
    size_t bx;
    int n;
    int nrc_x;

    bool operator==(const prepack_key & other) const {
        return vx == other.vx && bx == other.bx && n == other.n && nrc_x == other.nrc_x;
    }
};

struct prepack_key_hash {
    size_t operator()(const prepack_key & key) const {
        size_t h = std::hash<const void *>{}(key.vx);
        h ^= std::hash<size_t>{}(key.bx) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.nrc_x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct iq3_prepack_key {
    prepack_key base;
    uint64_t content_tag;

    bool operator==(const iq3_prepack_key & other) const {
        return base == other.base && content_tag == other.content_tag;
    }
};

struct iq3_prepack_key_hash {
    size_t operator()(const iq3_prepack_key & key) const {
        size_t h = prepack_key_hash{}(key.base);
        h ^= std::hash<uint64_t>{}(key.content_tag) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class amx_hugepage_arena {
    struct chunk {
        uint8_t * data;
        size_t size;
        size_t used;
    };

public:
    ~amx_hugepage_arena() {
        for (const chunk & value : chunks_) {
            munmap(value.data, value.size);
        }
    }

    void * allocate(size_t size, size_t alignment = 64) {
        GGML_ASSERT(alignment && (alignment & (alignment - 1)) == 0);
        if (chunks_.empty() || align_up(chunks_.back().used, alignment) + size > chunks_.back().size) {
            if (!chunks_.empty()) {
                chunk & previous = chunks_.back();
                const size_t populated = previous.used & ~(hugepage_size - 1);
                if (populated) {
                    (void) madvise(previous.data, populated, MADV_COLLAPSE);
                }
            }
            add_chunk(std::max(default_chunk_size, align_up(size, hugepage_size)));
        }
        chunk & value = chunks_.back();
        value.used = align_up(value.used, alignment);
        void * result = value.data + value.used;
        value.used += size;
        return result;
    }

private:
    static constexpr size_t hugepage_size = 2u*1024u*1024u;
    // There is one arena per worker thread.  Keeping chunks moderately sized
    // limits the aggregate unused tail (important on 64 GiB HBM systems)
    // while still giving khugepaged four contiguous 2 MiB huge pages at a
    // time to work with.
    static constexpr size_t default_chunk_size = 8u*1024u*1024u;

    static size_t align_up(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    void add_chunk(size_t size) {
        size = align_up(size, hugepage_size);
        const size_t reserve = size + hugepage_size;
        void * raw = mmap(nullptr, reserve, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            std::fprintf(stderr, "ggml_amx_int8: mmap failed for %zu-byte prepack arena\n", size);
            std::abort();
        }
        const uintptr_t raw_address = reinterpret_cast<uintptr_t>(raw);
        const uintptr_t aligned_address = align_up(raw_address, hugepage_size);
        const size_t prefix = aligned_address - raw_address;
        const size_t suffix = reserve - prefix - size;
        if (prefix) {
            munmap(raw, prefix);
        }
        if (suffix) {
            munmap(reinterpret_cast<void *>(aligned_address + size), suffix);
        }
        auto * data = reinterpret_cast<uint8_t *>(aligned_address);
        (void) madvise(data, size, MADV_HUGEPAGE);
        chunks_.push_back({data, size, 0});
    }

    std::vector<chunk> chunks_;
};

amx_hugepage_arena & q4_prepack_arena() {
    static thread_local amx_hugepage_arena arena;
    return arena;
}

void * iq3_shared_allocate(size_t size) {
    static std::mutex mutex;
    static amx_hugepage_arena arena;
    std::lock_guard<std::mutex> lock(mutex);
    return arena.allocate(size);
}

struct q4_prepack {
    bool compact = true;
    int stride = 0;
    uint8_t * q = nullptr;
    uint8_t * scales = nullptr;
    uint8_t * scaled_low = nullptr;
    uint8_t * scaled_high = nullptr;
    uint8_t * minq = nullptr;
    uint8_t * mins = nullptr;
    float * d = nullptr;
    float * dmin = nullptr;
    size_t q_size = 0;
    size_t scales_size = 0;
    size_t scaled_low_size = 0;
    size_t scaled_high_size = 0;
    size_t minq_size = 0;
    size_t mins_size = 0;
    size_t d_size = 0;
    size_t dmin_size = 0;
};

struct q5_prepack {
    int stride = 0;
    uint8_t * scaled_low = nullptr;
    uint8_t * scaled_high = nullptr;
    uint8_t * minq = nullptr;
    float * d = nullptr;
    float * dmin = nullptr;
    size_t scaled_low_size = 0;
    size_t scaled_high_size = 0;
    size_t minq_size = 0;
    size_t d_size = 0;
    size_t dmin_size = 0;
};

struct q6_prepack {
    int stride = 0;
    int8_t * q = nullptr;
    int8_t * scales = nullptr;
    float * d = nullptr;
    size_t q_size = 0;
    size_t scales_size = 0;
    size_t d_size = 0;
};

struct iq3_prepack {
    bool expanded = false;
    int stride = 0;
    int8_t * q = nullptr;
    int8_t * scales = nullptr;
    int8_t * low = nullptr;
    int8_t * high = nullptr;
    float * d = nullptr;
    size_t q_size = 0;
    size_t scales_size = 0;
    size_t low_size = 0;
    size_t high_size = 0;
    size_t d_size = 0;
};

struct iq4_prepack {
    std::vector<uint8_t> q;
    std::vector<float> dscale;
};

static const q4_prepack & get_q4_prepack(int n, const void * vx, size_t bx, int nrc_x) {
    static thread_local std::unordered_map<prepack_key, q4_prepack, prepack_key_hash> cache;
    const prepack_key key{vx, bx, n, nrc_x};
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    q4_prepack packed;
    const int nk32 = n/32;
    packed.stride = 16*((nrc_x + 15)/16);
    const int stride = packed.stride;
    const int ntiles = stride/16;
    const size_t block_tiles = static_cast<size_t>(n/QK_K)*ntiles;
    const size_t compact_payload = block_tiles*(8*256 + 8*16 + 128);
    const size_t expanded_payload = block_tiles*(8*512 + 4*256);
    packed.compact = !q4_claim_expanded_cache(expanded_payload - compact_payload);
    if (packed.compact) {
        // Retain four-bit weights in a compact VNNI-oriented layout.  The
        // current 16x32 tile is multiplied by its scale and expanded only in
        // L1, so persistent storage stays close to the source quant size.
        packed.q_size = static_cast<size_t>(n/QK_K)*ntiles*8*256;
        packed.scales_size = static_cast<size_t>(n/QK_K)*ntiles*8*16;
        packed.q = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.q_size));
        packed.scales = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.scales_size));
    } else {
        // Expanded comparison path: exact 10-bit dscale*q as low8 plus two
        // compact high bit planes.
        packed.scaled_low_size = static_cast<size_t>(n/QK_K)*ntiles*8*512;
        packed.scaled_high_size = static_cast<size_t>(n/QK_K)*ntiles*4*256;
        packed.scaled_low = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.scaled_low_size));
        packed.scaled_high = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.scaled_high_size));
        std::memset(packed.scaled_high, 0, packed.scaled_high_size);
    }
    // Only two VNNI rows carry the eight K32 minimum scales.  Keep those 128
    // bytes compact in the persistent cache and add the K64 zero padding in a
    // small L1-resident scratch tile at execution time.
    packed.minq_size = static_cast<size_t>(n/QK_K)*ntiles*128;
    packed.d_size = static_cast<size_t>(n/QK_K)*stride;
    packed.dmin_size = static_cast<size_t>(n/QK_K)*stride;
    packed.minq = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.minq_size));
    if (packed.compact) {
        packed.mins_size = packed.minq_size;
        packed.mins = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.mins_size));
    }
    packed.d = static_cast<float *>(q4_prepack_arena().allocate(packed.d_size*sizeof(float)));
    packed.dmin = static_cast<float *>(q4_prepack_arena().allocate(packed.dmin_size*sizeof(float)));
    std::memset(packed.minq, 0, packed.minq_size);
    static constexpr uint8_t values[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    for (int kb32 = 0; kb32 < nk32; ++kb32) {
        const int ibl = kb32/8;
        const int ib32 = kb32%8;
        for (int row = 0; row < nrc_x; ++row) {
            const int group_row = row & ~3;
            const int lane = row & 3;
            const auto * blocks = reinterpret_cast<const block_q4_k_r4 *>(
                    static_cast<const char *>(vx) + static_cast<size_t>(group_row)*bx);
            const block_q4_k_r4 & block = blocks[ibl];
            alignas(32) uint8_t unpacked[32];
            unpack_r4_nibbles(block.qs + 64*ib32, lane, unpacked, values);
            const int tile = row/16;
            const int col = row%16;
            const int si = 4*ib32 + lane;
            const uint8_t low = block.scales_l[si];
            const uint8_t high = (block.scales_h[si%16] >> (4*(si/16))) & 0x0f;
            const int ds = (low & 0x0f) | ((high & 0x03) << 4);
            const int ms = (low >> 4) | ((high >> 2) << 4);
            const size_t group = (static_cast<size_t>(ibl)*ntiles + tile)*8 + ib32;
            if (packed.compact) {
                uint8_t * q = packed.q + group*256;
                for (int pair = 0; pair < 4; ++pair) {
                    for (int j = 0; j < 4; ++j) {
                        q[pair*64 + col*4 + j] =
                            unpacked[8*pair + j] | (unpacked[8*pair + 4 + j] << 4);
                    }
                }
                packed.scales[group*16 + col] = ds;
            } else {
                const size_t li = group*512;
                for (int ik = 0; ik < 8; ++ik) {
                    for (int j = 0; j < 4; ++j) {
                        const int product = ds*unpacked[4*ik + j];
                        packed.scaled_low[li + ik*64 + col*4 + j] = product & 0xff;
                    }
                }
                const size_t hi = (static_cast<size_t>(ibl)*ntiles + tile)*1024;
                for (int k32 = 0; k32 < 32; ++k32) {
                    const int ik = 32*ib32 + k32;
                    const uint8_t upper = (ds*unpacked[k32]) >> 8;
                    const int bit_index = col*4 + ik%4;
                    uint8_t * planes = packed.scaled_high + hi + (ik/4)*16;
                    if (upper & 1) {
                        planes[bit_index/8] |= 1u << (bit_index%8);
                    }
                    if (upper & 2) {
                        planes[8 + bit_index/8] |= 1u << (bit_index%8);
                    }
                }
            }
            const size_t mi = (static_cast<size_t>(ibl)*ntiles + tile)*128 +
                (ib32/4)*64 + col*4 + ib32%4;
            packed.minq[mi] = ms;
            if (packed.compact) {
                packed.mins[group*16 + col] = ms;
            }
            if (ib32 == 0) {
                const size_t di = static_cast<size_t>(ibl)*stride + row;
                packed.d[di] = GGML_FP16_TO_FP32(block.d[lane]);
                packed.dmin[di] = GGML_FP16_TO_FP32(block.d[lane + 4]);
            }
        }
    }
    const uint64_t bytes = packed.q_size + packed.scales_size + packed.mins_size +
        packed.scaled_low_size + packed.scaled_high_size + packed.minq_size +
        (packed.d_size + packed.dmin_size)*sizeof(float);
    if (stats_enabled()) {
        stats().prepack_builds.fetch_add(1, std::memory_order_relaxed);
        stats().prepack_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return cache.emplace(key, std::move(packed)).first->second;
}

static const q5_prepack & get_q5_prepack(int n, const void * vx, size_t bx, int nrc_x) {
    static thread_local std::unordered_map<prepack_key, q5_prepack, prepack_key_hash> cache;
    const prepack_key key{vx, bx, n, nrc_x};
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    q5_prepack packed;
    const int nk32 = n/32;
    packed.stride = 16*((nrc_x + 15)/16);
    const int stride = packed.stride;
    const int ntiles = stride/16;
    // Q5's exact scale*q product needs eleven bits.  Keep the low byte in
    // AMX VNNI-B order and pack the remaining three bits into bit planes.
    packed.scaled_low_size = static_cast<size_t>(n/QK_K)*ntiles*8*512;
    packed.scaled_high_size = static_cast<size_t>(n/QK_K)*ntiles*6*256;
    packed.scaled_low = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.scaled_low_size));
    packed.scaled_high = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.scaled_high_size));
    std::memset(packed.scaled_high, 0, packed.scaled_high_size);
    packed.minq_size = static_cast<size_t>(n/QK_K)*ntiles*128;
    packed.d_size = static_cast<size_t>(n/QK_K)*stride;
    packed.dmin_size = static_cast<size_t>(n/QK_K)*stride;
    packed.minq = static_cast<uint8_t *>(q4_prepack_arena().allocate(packed.minq_size));
    packed.d = static_cast<float *>(q4_prepack_arena().allocate(packed.d_size*sizeof(float)));
    packed.dmin = static_cast<float *>(q4_prepack_arena().allocate(packed.dmin_size*sizeof(float)));
    std::memset(packed.minq, 0, packed.minq_size);

    for (int kb32 = 0; kb32 < nk32; ++kb32) {
        const int ibl = kb32/8;
        const int ib32 = kb32%8;
        for (int row = 0; row < nrc_x; ++row) {
            const int group_row = row & ~3;
            const int lane = row & 3;
            const auto * blocks = reinterpret_cast<const block_q5_k_r4 *>(
                    static_cast<const char *>(vx) + static_cast<size_t>(group_row)*bx);
            const block_q5_k_r4 & block = blocks[ibl];
            alignas(32) uint8_t unpacked[32];
            unpack_q5_r4_32(block, ib32, lane, unpacked);
            const int tile = row/16;
            const int col = row%16;
            const int si = 4*ib32 + lane;
            const uint8_t low = block.scales_l[si];
            const uint8_t high = (block.scales_h[si%16] >> (4*(si/16))) & 0x0f;
            const int ds = (low & 0x0f) | ((high & 0x03) << 4);
            const int ms = (low >> 4) | ((high >> 2) << 4);
            const size_t group = (static_cast<size_t>(ibl)*ntiles + tile)*8 + ib32;
            const size_t li = group*512;
            for (int ik = 0; ik < 8; ++ik) {
                for (int j = 0; j < 4; ++j) {
                    const int product = ds*unpacked[4*ik + j];
                    packed.scaled_low[li + ik*64 + col*4 + j] = product & 0xff;
                }
            }
            const size_t hi = (static_cast<size_t>(ibl)*ntiles + tile)*1536;
            for (int k32 = 0; k32 < 32; ++k32) {
                const int ik = 32*ib32 + k32;
                const uint8_t upper = (ds*unpacked[k32]) >> 8;
                const int bit_index = col*4 + ik%4;
                uint8_t * planes = packed.scaled_high + hi + (ik/4)*24;
                if (upper & 1) {
                    planes[bit_index/8] |= 1u << (bit_index%8);
                }
                if (upper & 2) {
                    planes[8 + bit_index/8] |= 1u << (bit_index%8);
                }
                if (upper & 4) {
                    planes[16 + bit_index/8] |= 1u << (bit_index%8);
                }
            }
            const size_t mi = (static_cast<size_t>(ibl)*ntiles + tile)*128 +
                (ib32/4)*64 + col*4 + ib32%4;
            packed.minq[mi] = ms;
            if (ib32 == 0) {
                const size_t di = static_cast<size_t>(ibl)*stride + row;
                packed.d[di] = GGML_FP16_TO_FP32(block.d[lane]);
                packed.dmin[di] = GGML_FP16_TO_FP32(block.d[lane + 4]);
            }
        }
    }
    const uint64_t bytes = packed.scaled_low_size + packed.scaled_high_size + packed.minq_size +
        (packed.d_size + packed.dmin_size)*sizeof(float);
    if (stats_enabled()) {
        stats().prepack_builds.fetch_add(1, std::memory_order_relaxed);
        stats().prepack_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return cache.emplace(key, std::move(packed)).first->second;
}

static const q6_prepack & get_q6_prepack(int n, const void * vx, size_t bx, int nrc_x) {
    static thread_local std::unordered_map<prepack_key, q6_prepack, prepack_key_hash> cache;
    const prepack_key key{vx, bx, n, nrc_x};
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    q6_prepack packed;
    const int nbl = n/QK_K;
    packed.stride = 16*((nrc_x + 15)/16);
    const int stride = packed.stride;
    const int ntiles = stride/16;
    // One signed byte per Q6 weight in AMX VNNI-B order, plus one signed
    // scale per K16 group.  This is 1.0625 bytes/weight instead of the old
    // exact-product cache's 1.640625 bytes/weight.
    packed.q_size = static_cast<size_t>(nbl)*ntiles*4096;
    packed.scales_size = static_cast<size_t>(nbl)*ntiles*256;
    packed.d_size = static_cast<size_t>(nbl)*stride;
    packed.q = static_cast<int8_t *>(q4_prepack_arena().allocate(packed.q_size));
    packed.scales = static_cast<int8_t *>(q4_prepack_arena().allocate(packed.scales_size));
    packed.d = static_cast<float *>(q4_prepack_arena().allocate(packed.d_size*sizeof(float)));
    std::memset(packed.q, 0, packed.q_size);
    std::memset(packed.scales, 0, packed.scales_size);

    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int row = 0; row < nrc_x; ++row) {
            const int group_row = row & ~3;
            const int lane = row & 3;
            const auto * blocks = reinterpret_cast<const block_q6_k_r4 *>(
                static_cast<const char *>(vx) + static_cast<size_t>(group_row)*bx);
            const block_q6_k_r4 & block = blocks[ibl];
            const int tile = row/16;
            const int col = row%16;
            const size_t tile_index = static_cast<size_t>(ibl)*ntiles + tile;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                alignas(32) uint8_t quants[32];
                unpack_q6_r4_32(block, ib32, lane, quants);
                for (int k32 = 0; k32 < 32; ++k32) {
                    const int ik = 32*ib32 + k32;
                    const size_t offset = tile_index*4096 +
                        (ik/4)*64 + col*4 + ik%4;
                    packed.q[offset] = static_cast<int8_t>(quants[k32] - 32);
                }
                packed.scales[tile_index*256 + (2*ib32 + 0)*16 + col] =
                    block.scales[8*ib32 + lane + 0];
                packed.scales[tile_index*256 + (2*ib32 + 1)*16 + col] =
                    block.scales[8*ib32 + lane + 4];
            }
            packed.d[static_cast<size_t>(ibl)*stride + row] =
                GGML_FP16_TO_FP32(block.d[lane]);
        }
    }
    const uint64_t bytes = packed.q_size + packed.scales_size +
        packed.d_size*sizeof(float);
    if (stats_enabled()) {
        stats().prepack_builds.fetch_add(1, std::memory_order_relaxed);
        stats().prepack_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return cache.emplace(key, std::move(packed)).first->second;
}

static inline void decode_iq3_group_vnni(
        const block_iq3_s_r4 & block, int col, int8_t * q, int8_t * scales) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i one = _mm_set1_epi32(1);
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        uint32_t packed_scales = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const int scale_index = 4*ib32 + lane;
            const uint8_t scale = static_cast<uint8_t>(
                1 + 2*((block.scales[scale_index%16] >> (4*(scale_index/16))) & 0x0f));
            packed_scales |= static_cast<uint32_t>(scale) << (8*lane);
        }
        std::memcpy(scales + ib32*16 + col, &packed_scales, sizeof(packed_scales));

        const __m128i sign_bits = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(block.signs + 16*ib32));
        const __m128i qh_group = _mm_cvtepu8_epi32(
            _mm_loadu_si32(block.qh + 4*ib32));
        for (int i = 0; i < 4; ++i) {
            const __m128i index0_low = _mm_cvtepu8_epi32(
                _mm_loadu_si32(block.qs + 32*ib32 + 8*i + 0));
            const __m128i index1_low = _mm_cvtepu8_epi32(
                _mm_loadu_si32(block.qs + 32*ib32 + 8*i + 4));
            const __m128i index0 = _mm_or_si128(index0_low,
                _mm_slli_epi32(_mm_and_si128(_mm_srlv_epi32(qh_group,
                    _mm_set1_epi32(i)), one), 8));
            const __m128i index1 = _mm_or_si128(index1_low,
                _mm_slli_epi32(_mm_and_si128(_mm_srlv_epi32(qh_group,
                    _mm_set1_epi32(i + 4)), one), 8));
            const __m256i indices = _mm256_inserti128_si256(
                _mm256_castsi128_si256(index0), index1, 1);
            const __m256i magnitudes = _mm256_i32gather_epi32(
                reinterpret_cast<const int *>(iq3s_grid), indices, 4);
            const __m128i magnitude0 = _mm256_castsi256_si128(magnitudes);
            const __m128i magnitude1 = _mm256_extracti128_si256(magnitudes, 1);
            const __mmask16 mask0 = _mm_test_epi8_mask(sign_bits, _mm_set1_epi8(1 << i));
            const __mmask16 mask1 = _mm_test_epi8_mask(sign_bits, _mm_set1_epi8(1 << (i + 4)));
            const __m128i signed0 = _mm_mask_sub_epi8(magnitude0, mask0, zero, magnitude0);
            const __m128i signed1 = _mm_mask_sub_epi8(magnitude1, mask1, zero, magnitude1);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(q + (8*ib32 + i + 0)*64 + 4*col), signed0);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(q + (8*ib32 + i + 4)*64 + 4*col), signed1);
        }
    }
}

static const iq3_prepack & get_iq3_prepack(int n, const void * vx, size_t bx, int nrc_x) {
    struct cache_shard {
        std::mutex mutex;
        std::unordered_map<iq3_prepack_key, iq3_prepack, iq3_prepack_key_hash> values;
    };
    static std::array<cache_shard, 256> cache;
    const bool expanded = env_is_set("GGML_AMX_IQ3_S_EXPANDED");
    uint64_t tag0;
    uint64_t tag1;
    std::memcpy(&tag0, vx, sizeof(tag0));
    std::memcpy(&tag1, static_cast<const char *>(vx) + static_cast<size_t>(nrc_x - 4)*bx,
        sizeof(tag1));
    const iq3_prepack_key key{{vx, bx, n, nrc_x},
        tag0 ^ (tag1 + 0x9e3779b97f4a7c15ULL) ^ (expanded ? 0xd6e8feb86659fd93ULL : 0)};
    cache_shard & shard = cache[iq3_prepack_key_hash{}(key) % cache.size()];
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto found = shard.values.find(key);
    if (found != shard.values.end()) {
        return found->second;
    }

    iq3_prepack packed;
    packed.expanded = expanded;
    const int nbl = n/QK_K;
    packed.stride = 16*((nrc_x + 15)/16);
    const int stride = packed.stride;
    const int ntiles = stride/16;
    const size_t weight_bytes = static_cast<size_t>(nbl)*ntiles*4096;
    if (expanded) {
        packed.low_size = weight_bytes;
        packed.high_size = weight_bytes;
    } else {
        // Decode the IQ3 codebook once, but keep per-K32 scales separate.  The
        // exact scale*q products are formed in L1 immediately before tile loads.
        // Persistent cost is 1.046875 bytes/weight rather than two expanded
        // signed bytes per weight.
        packed.q_size = weight_bytes;
        packed.scales_size = static_cast<size_t>(nbl)*ntiles*128;
    }
    packed.d_size = static_cast<size_t>(nbl)*stride;
    const size_t total_size = packed.q_size + packed.scales_size + packed.low_size +
        packed.high_size + packed.d_size*sizeof(float);
    auto * storage = static_cast<uint8_t *>(iq3_shared_allocate(total_size));
    size_t offset = 0;
    if (packed.q_size) {
        packed.q = reinterpret_cast<int8_t *>(storage + offset);
        offset += packed.q_size;
        packed.scales = reinterpret_cast<int8_t *>(storage + offset);
        offset += packed.scales_size;
        std::memset(packed.q, 0, packed.q_size);
        std::memset(packed.scales, 0, packed.scales_size);
    } else {
        packed.low = reinterpret_cast<int8_t *>(storage + offset);
        offset += packed.low_size;
        packed.high = reinterpret_cast<int8_t *>(storage + offset);
        offset += packed.high_size;
        std::memset(packed.low, 0, packed.low_size);
        std::memset(packed.high, 0, packed.high_size);
    }
    packed.d = reinterpret_cast<float *>(storage + offset);
    std::memset(packed.d, 0, packed.d_size*sizeof(float));

    GGML_ASSERT(nrc_x % 4 == 0);
    alignas(64) int8_t scratch_q[4096];
    alignas(64) int8_t scratch_scales[128];
    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int tile = 0; tile < ntiles; ++tile) {
            const size_t tile_index = static_cast<size_t>(ibl)*ntiles + tile;
            int8_t * q = expanded ? scratch_q : packed.q + tile_index*4096;
            int8_t * scales = expanded ? scratch_scales : packed.scales + tile_index*128;
            const int first_row = 16*tile;
            const int rows = std::min(16, nrc_x - first_row);
            for (int local_row = 0; local_row < rows; local_row += 4) {
                const int row = first_row + local_row;
                const auto * blocks = reinterpret_cast<const block_iq3_s_r4 *>(
                    static_cast<const char *>(vx) + static_cast<size_t>(row)*bx);
                const block_iq3_s_r4 & block = blocks[ibl];
                decode_iq3_group_vnni(block, local_row, q, scales);
                _mm_storeu_ps(packed.d + static_cast<size_t>(ibl)*stride + row,
                    _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.d))));
            }
            if (expanded) {
                for (int k64 = 0; k64 < QK_K/64; ++k64) {
                    encode_iq3_scaled_vnni(q + k64*1024, scales + 2*k64*16,
                        packed.low + tile_index*4096 + k64*1024,
                        packed.high + tile_index*4096 + k64*1024);
                }
            }
        }
    }
    const uint64_t bytes = packed.q_size + packed.scales_size + packed.low_size +
        packed.high_size + packed.d_size*sizeof(float);
    if (stats_enabled()) {
        stats().prepack_builds.fetch_add(1, std::memory_order_relaxed);
        stats().prepack_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return shard.values.emplace(key, std::move(packed)).first->second;
}

static void decode_iq3_tile(
        const void * vx, size_t bx, int ibl, int ix0, int nrows,
        int8_t * q, int8_t * scales, float * d) {
    GGML_ASSERT(ix0 % 4 == 0 && nrows % 4 == 0);
    for (int local_row = 0; local_row < nrows; local_row += 4) {
        const int row = ix0 + local_row;
        const auto * blocks = reinterpret_cast<const block_iq3_s_r4 *>(
            static_cast<const char *>(vx) + static_cast<size_t>(row)*bx);
        const block_iq3_s_r4 & block = blocks[ibl];
        decode_iq3_group_vnni(block, local_row, q, scales);
        _mm_storeu_ps(d + local_row,
            _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.d))));
    }
}

static const iq4_prepack & get_iq4_prepack(int n, const void * vx, size_t bx, int nrc_x) {
    static thread_local std::unordered_map<prepack_key, iq4_prepack, prepack_key_hash> cache;
    const prepack_key key{vx, bx, n, nrc_x};
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    iq4_prepack packed;
    const int nblocks = n/QK4_NL;
    packed.q.resize(static_cast<size_t>(nblocks)*nrc_x*16);
    packed.dscale.resize(static_cast<size_t>(nblocks)*nrc_x);
    static constexpr uint8_t values[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    };
    for (int ib = 0; ib < nblocks; ++ib) {
        for (int row = 0; row < nrc_x; ++row) {
            const int group_row = row & ~3;
            const int lane = row & 3;
            const auto * blocks = reinterpret_cast<const block_iq4_nl_r4 *>(
                    static_cast<const char *>(vx) + static_cast<size_t>(group_row)*bx);
            const block_iq4_nl_r4 & block = blocks[ib];
            alignas(32) uint8_t unpacked[32];
            unpack_r4_nibbles(block.qs, lane, unpacked, values);
            const size_t qi = (static_cast<size_t>(ib)*nrc_x + row)*16;
            for (int j = 0; j < 16; ++j) {
                packed.q[qi + j] = unpacked[j] | (unpacked[j + 16] << 4);
            }
            packed.dscale[static_cast<size_t>(ib)*nrc_x + row] =
                GGML_FP16_TO_FP32(block.d[lane]);
        }
    }
    const uint64_t bytes = packed.q.size() + packed.dscale.size()*sizeof(float);
    if (stats_enabled()) {
        stats().prepack_builds.fetch_add(1, std::memory_order_relaxed);
        stats().prepack_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return cache.emplace(key, std::move(packed)).first->second;
}

static inline void q8_2_block(
        const char * base, int nblocks, int ib, const int8_t *& quants, float & d, int & sum) {
    const int nblocks4 = 4*(nblocks/4);
    if (ib < nblocks4) {
        const auto * blocks = reinterpret_cast<const block_q8_2_x4 *>(base);
        const block_q8_2_x4 & block = blocks[ib/4];
        const int lane = ib % 4;
        quants = block.qs + 32*lane;
        d = GGML_BF16_TO_FP32(ggml_bf16_t{block.d[lane]});
        sum = reinterpret_cast<const int16_t *>(block.d)[lane + 4];
    } else {
        const char * tail = base + static_cast<size_t>(nblocks4)*sizeof(block_q8_2);
        const auto * block = reinterpret_cast<const block_q8_2 *>(tail) + (ib - nblocks4);
        quants = block->qs;
        d = GGML_BF16_TO_FP32(ggml_bf16_t{block->d});
        sum = *reinterpret_cast<const int16_t *>(&block->s);
    }
}

} // namespace

bool iqk_amx_int8_runtime_available() {
    static thread_local int available = -1;
    if (available >= 0) {
        return available != 0;
    }
    if (env_is_set("GGML_AMX_DISABLE") || env_is_set("GGML_AMX_INT8_DISABLE")) {
        available = 0;
        return false;
    }

    unsigned int eax, ebx, ecx, edx;
    constexpr unsigned int cpuid_amx_tile = 1u << 24;
    constexpr unsigned int cpuid_amx_int8 = 1u << 25;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) ||
        (edx & (cpuid_amx_tile | cpuid_amx_int8)) != (cpuid_amx_tile | cpuid_amx_int8)) {
        available = 0;
        return false;
    }

    available = syscall(SYS_arch_prctl, IQK_ARCH_REQ_XCOMP_PERM, IQK_XFEATURE_XTILEDATA) == 0;
    return available != 0;
}

void iqk_amx_set_transient_weights(bool transient) {
    transient_weights = transient;
}

void iqk_amx_set_weight_origin(const void * base, int row_offset, int total_rows) {
    weight_origin = {base, row_offset, total_rows};
}

static bool iqk_amx_mul_mat_q4_k_r4_compact(
        int n, const q4_prepack & packed, const DataInfo & info, int nrc_x, int nrc_y) {
    const int nbl = n/QK_K;
    const int ntiles = packed.stride/16;
    alignas(64) float accum[nrc_y*64];
    std::memset(accum, 0, static_cast<size_t>(nrc_y)*64*sizeof(float));
    const block_q8_K * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
    }

    alignas(64) int8_t activations[nrc_y][QK_K];
    alignas(64) int16_t qsums[nrc_y][8];
    alignas(64) float dy[nrc_y];
    alignas(64) uint8_t weights[8][8][64];
    alignas(64) int32_t dots[32][16];
    alignas(64) int32_t main_acc[32][16];
    alignas(64) int32_t min_acc[32][16];
    uint64_t tile_count = 0;
    int configured_nx = 0;
    int configured_ny0 = 0;
    int configured_ny1 = 0;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            dy[iy] = y[iy][ibl].d;
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                _mm512_store_si512(reinterpret_cast<__m512i *>(activations[iy] + 64*k64),
                    _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
                        y[iy][ibl].qs + 64*k64)));
            }
            for (int group = 0; group < 8; ++group) {
                qsums[iy][group] = static_cast<int16_t>(
                    sum_i8_32(y[iy][ibl].qs + 32*group));
            }
        }

        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            const size_t tile_base = (static_cast<size_t>(ibl)*ntiles + ix0/16)*8;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                unpack_q4_vnni(packed.q + (tile_base + ib32)*256,
                    &weights[ib32][0][0]);
            }
            const size_t di = static_cast<size_t>(ibl)*packed.stride + ix0;
            const __m512 vd = _mm512_maskz_loadu_ps(mask, packed.d + di);
            const __m512 vdmin = _mm512_maskz_loadu_ps(mask, packed.dmin + di);

            for (int iy0 = 0; iy0 < nrc_y; iy0 += 32) {
                const int ny = std::min(32, nrc_y - iy0);
                const int ny0 = std::min(16, ny);
                const int ny1 = std::max(0, ny - 16);
                if (nx != configured_nx || ny0 != configured_ny0 || ny1 != configured_ny1) {
                    configure_tiles_q4_transposed(ny0, ny1, nx);
                    configured_nx = nx;
                    configured_ny0 = ny0;
                    configured_ny1 = ny1;
                }
                std::memset(main_acc, 0, sizeof(main_acc));
                std::memset(min_acc, 0, sizeof(min_acc));

                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const int8_t * a1 = ny1 ? &activations[iy0 + 16][32*ib32] : nullptr;
                    amx_compute_q4_transposed(
                        &activations[iy0][32*ib32], a1,
                        &weights[ib32][0][0], ny1, QK_K);
                    amx_store_q4_transposed(&dots[0][0], ny1);
                    tile_count += ny1 ? 2 : 1;

                    const __m512i vscale = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(packed.scales + (tile_base + ib32)*16)));
                    const __m512i vmin = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(packed.mins + (tile_base + ib32)*16)));
                    for (int iy = 0; iy < ny; ++iy) {
                        const __m512i vdot = _mm512_maskz_loadu_epi32(mask, dots[iy]);
                        __m512i vmain = _mm512_load_si512(
                            reinterpret_cast<const __m512i *>(main_acc[iy]));
                        __m512i vmins = _mm512_load_si512(
                            reinterpret_cast<const __m512i *>(min_acc[iy]));
                        vmain = _mm512_add_epi32(vmain, _mm512_mullo_epi32(vdot, vscale));
                        vmins = _mm512_add_epi32(vmins, _mm512_mullo_epi32(
                            _mm512_set1_epi32(qsums[iy0 + iy][ib32]), vmin));
                        _mm512_store_si512(reinterpret_cast<__m512i *>(main_acc[iy]), vmain);
                        _mm512_store_si512(reinterpret_cast<__m512i *>(min_acc[iy]), vmins);
                    }
                }

                for (int iy = 0; iy < ny; ++iy) {
                    const __m512 vmain = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(mask, main_acc[iy]));
                    const __m512 vmins = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(mask, min_acc[iy]));
                    const __m512 vvalue = _mm512_sub_ps(_mm512_mul_ps(vd, vmain),
                                                       _mm512_mul_ps(vdmin, vmins));
                    const int output_y = iy0 + iy;
                    const size_t oi = static_cast<size_t>(output_y)*64 + ix0;
                    __m512 vout = _mm512_maskz_loadu_ps(mask, accum + oi);
                    vout = _mm512_fmadd_ps(_mm512_set1_ps(dy[output_y]), vvalue, vout);
                    _mm512_mask_storeu_ps(accum + oi, mask, vout);
                }
            }
        }
    }
    _tile_release();

    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            _mm512_mask_storeu_ps(info.dst_row(iy) + ix0, mask,
                _mm512_maskz_loadu_ps(mask, accum + static_cast<size_t>(iy)*64 + ix0));
        }
    }
    if (stats_enabled()) {
        stats().q4_calls.fetch_add(1, std::memory_order_relaxed);
        stats().q4_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

bool iqk_amx_mul_mat_q4_k_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y) {
    constexpr int max_nrc_y = 512;
    if (env_is_set("GGML_AMX_Q4_K_DISABLE") || !iqk_amx_int8_runtime_available() ||
        nrc_y < 4 || nrc_y > max_nrc_y ||
        nrc_x % 4 != 0 || nrc_x > 64 || n % QK_K != 0 || transient_weights) {
        return false;
    }

    const int nbl = n/QK_K;
    const q4_prepack & packed = get_q4_prepack(n, vx, bx, nrc_x);
    if (packed.compact && env_is_set("GGML_AMX_Q4_K_RAW_DOT")) {
        return iqk_amx_mul_mat_q4_k_r4_compact(n, packed, info, nrc_x, nrc_y);
    }
    const int stride = packed.stride;
    const int ntiles = stride/16;
    alignas(64) float accum[nrc_y*64];
    std::memset(accum, 0, static_cast<size_t>(nrc_y)*64*sizeof(float));
    const block_q8_K * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
    }

    alignas(64) int8_t activations[nrc_y][QK_K];
    alignas(64) uint8_t weights_low[4][16][64];
    alignas(64) uint8_t weights_high[4][16][64];
    alignas(64) int32_t main_low[32][64];
    alignas(64) int32_t main_high[32][64];
    alignas(64) int32_t cmin_low[32][16];
    alignas(64) int32_t cmin_high[32][16];
    alignas(64) uint8_t min_weights_tile[16][64] = {};
    alignas(64) int16_t qsums[nrc_y][8];
    alignas(64) float dy[nrc_y];
    alignas(64) int8_t min_low[nrc_y][64];
    alignas(64) int8_t min_high[nrc_y][64];
    std::memset(min_low, 0, static_cast<size_t>(nrc_y)*64);
    std::memset(min_high, 0, static_cast<size_t>(nrc_y)*64);
    uint64_t tile_count = 0;

    int configured_nx = 0;
    int configured_ny0 = 0;
    int configured_ny1 = 0;
    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            dy[iy] = y[iy][ibl].d;
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                _mm512_store_si512(reinterpret_cast<__m512i *>(activations[iy] + 64*k64),
                    _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
                        y[iy][ibl].qs + 64*k64)));
            }
            for (int group = 0; group < 8; ++group) {
                qsums[iy][group] = static_cast<int16_t>(
                    sum_i8_32(y[iy][ibl].qs + 32*group));
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            for (int group = 0; group < 8; ++group) {
                const int high = qsums[iy][group]/128;
                min_low[iy][group] = static_cast<int8_t>(qsums[iy][group] - 128*high);
                min_high[iy][group] = static_cast<int8_t>(high);
            }
        }

        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            if (packed.compact) {
                for (int k64 = 0; k64 < 4; ++k64) {
                    for (int half = 0; half < 2; ++half) {
                        const size_t group =
                            (static_cast<size_t>(ibl)*ntiles + ix0/16)*8 + 2*k64 + half;
                        unpack_scaled_q4_vnni(
                            packed.q + group*256, packed.scales + group*16,
                            &weights_low[k64][0][0] + half*512,
                            &weights_high[k64][0][0] + half*512);
                    }
                }
            } else {
                // Decode the compact high-two-bit planes once, then reuse the
                // resulting 4 KiB working set for every 32-token tile group.
                for (int k64 = 0; k64 < 4; ++k64) {
                    const size_t high_tile =
                        (static_cast<size_t>(ibl)*ntiles + ix0/16)*1024 + k64*256;
                    unpack_q4_high2_vnni(packed.scaled_high + high_tile,
                        &weights_high[k64][0][0]);
                }
            }

            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            const uint8_t * min_weights = packed.minq +
                (static_cast<size_t>(ibl)*ntiles + ix0/16)*128;
            std::memcpy(&min_weights_tile[0][0], min_weights, 128);
            const size_t di = static_cast<size_t>(ibl)*stride + ix0;
            const __m512 vd = _mm512_maskz_loadu_ps(mask, packed.d + di);
            const __m512 vdmin = _mm512_maskz_loadu_ps(mask, packed.dmin + di);

            for (int iy0 = 0; iy0 < nrc_y; iy0 += 32) {
                const int ny = std::min(32, nrc_y - iy0);
                const int ny0 = std::min(16, ny);
                const int ny1 = std::max(0, ny - 16);
                if (nx != configured_nx || ny0 != configured_ny0 || ny1 != configured_ny1) {
                    configure_tiles_q4_scaled(ny0, ny1, nx);
                    configured_nx = nx;
                    configured_ny0 = ny0;
                    configured_ny1 = ny1;
                }

                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_zero(2);
                if (ny1) _tile_zero(3);
                for (int k64 = 0; k64 < 4; ++k64) {
                    _tile_loadd(4, &activations[iy0][64*k64], QK_K);
                    if (ny1) _tile_loadd(5, &activations[iy0 + 16][64*k64], QK_K);
                    const size_t low_group =
                        (static_cast<size_t>(ibl)*ntiles + ix0/16)*8 + 2*k64;
                    const uint8_t * low_weights = packed.compact ?
                        &weights_low[k64][0][0] : packed.scaled_low + low_group*512;
                    _tile_loadd(6, low_weights, 64);
                    _tile_dpbsud(0, 4, 6);
                    if (ny1) _tile_dpbsud(1, 5, 6);
                    _tile_loadd(6, &weights_high[k64][0][0], 64);
                    _tile_dpbsud(2, 4, 6);
                    if (ny1) _tile_dpbsud(3, 5, 6);
                    tile_count += ny1 ? 4 : 2;
                }
                _tile_stored(0, &main_low[0][ix0], 64*sizeof(int32_t));
                if (ny1) _tile_stored(1, &main_low[16][ix0], 64*sizeof(int32_t));
                _tile_stored(2, &main_high[0][ix0], 64*sizeof(int32_t));
                if (ny1) _tile_stored(3, &main_high[16][ix0], 64*sizeof(int32_t));

                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_loadd(4, &min_low[iy0][0], 64);
                if (ny1) _tile_loadd(5, &min_low[iy0 + 16][0], 64);
                _tile_loadd(6, &min_weights_tile[0][0], 64);
                _tile_dpbsud(0, 4, 6);
                if (ny1) _tile_dpbsud(1, 5, 6);
                _tile_stored(0, &cmin_low[0][0], 16*sizeof(int32_t));
                if (ny1) _tile_stored(1, &cmin_low[16][0], 16*sizeof(int32_t));

                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_loadd(4, &min_high[iy0][0], 64);
                if (ny1) _tile_loadd(5, &min_high[iy0 + 16][0], 64);
                _tile_dpbsud(0, 4, 6);
                if (ny1) _tile_dpbsud(1, 5, 6);
                _tile_stored(0, &cmin_high[0][0], 16*sizeof(int32_t));
                if (ny1) _tile_stored(1, &cmin_high[16][0], 16*sizeof(int32_t));
                tile_count += ny1 ? 4 : 2;

                for (int iy = 0; iy < ny; ++iy) {
                    const __m512i vhigh =
                        _mm512_maskz_loadu_epi32(mask, &main_high[iy][ix0]);
                    const __m512i vmain = _mm512_add_epi32(
                        _mm512_maskz_loadu_epi32(mask, &main_low[iy][ix0]),
                        _mm512_slli_epi32(vhigh, 8));
                    const __m512i vmin = _mm512_add_epi32(
                        _mm512_maskz_loadu_epi32(mask, cmin_low[iy]),
                        _mm512_slli_epi32(_mm512_maskz_loadu_epi32(mask, cmin_high[iy]), 7));
                    const int output_y = iy0 + iy;
                    const size_t oi = static_cast<size_t>(output_y)*64 + ix0;
                    const __m512 vdy = _mm512_set1_ps(dy[output_y]);
                    __m512 vout = _mm512_maskz_loadu_ps(mask, accum + oi);
                    vout = _mm512_fmadd_ps(_mm512_mul_ps(vdy, _mm512_cvtepi32_ps(vmain)),
                                           vd, vout);
                    vout = _mm512_fnmadd_ps(_mm512_mul_ps(vdy, _mm512_cvtepi32_ps(vmin)),
                                            vdmin, vout);
                    _mm512_mask_storeu_ps(accum + oi, mask, vout);
                }
            }
        }
    }
    _tile_release();

    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            _mm512_mask_storeu_ps(info.dst_row(iy) + ix0, mask,
                _mm512_maskz_loadu_ps(mask, accum + static_cast<size_t>(iy)*64 + ix0));
        }
    }
    if (stats_enabled()) {
        stats().q4_calls.fetch_add(1, std::memory_order_relaxed);
        stats().q4_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

bool iqk_amx_mul_mat_q5_k_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y) {
    if (env_is_set("GGML_AMX_Q5_K_DISABLE") || !iqk_amx_int8_runtime_available() ||
        nrc_y < q5_min_nrc_y() || nrc_y > 512 ||
        nrc_x % 4 != 0 || nrc_x > 64 || n % QK_K != 0 || transient_weights) {
        return false;
    }

    const int nbl = n/QK_K;
    const q5_prepack & packed = get_q5_prepack(n, vx, bx, nrc_x);
    const int stride = packed.stride;
    const int ntiles = stride/16;
    alignas(64) float accum[nrc_y*64];
    std::memset(accum, 0, static_cast<size_t>(nrc_y)*64*sizeof(float));
    const block_q8_K * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
    }

    alignas(64) int8_t activations[nrc_y][QK_K];
    alignas(64) uint8_t weights_high[4][16][64];
    alignas(64) int32_t main_low[32][64];
    alignas(64) int32_t main_high[32][64];
    alignas(64) int32_t cmin_low[32][16];
    alignas(64) int32_t cmin_high[32][16];
    alignas(64) uint8_t min_weights_tile[16][64] = {};
    alignas(64) int16_t qsums[nrc_y][8];
    alignas(64) float dy[nrc_y];
    alignas(64) int8_t min_low[nrc_y][64];
    alignas(64) int8_t min_high[nrc_y][64];
    std::memset(min_low, 0, static_cast<size_t>(nrc_y)*64);
    std::memset(min_high, 0, static_cast<size_t>(nrc_y)*64);
    uint64_t tile_count = 0;

    int configured_nx = 0;
    int configured_ny0 = 0;
    int configured_ny1 = 0;
    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            dy[iy] = y[iy][ibl].d;
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                _mm512_store_si512(reinterpret_cast<__m512i *>(activations[iy] + 64*k64),
                    _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
                        y[iy][ibl].qs + 64*k64)));
            }
            for (int group = 0; group < 8; ++group) {
                qsums[iy][group] = static_cast<int16_t>(
                    sum_i8_32(y[iy][ibl].qs + 32*group));
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            for (int group = 0; group < 8; ++group) {
                const int high = qsums[iy][group]/128;
                min_low[iy][group] = static_cast<int8_t>(qsums[iy][group] - 128*high);
                min_high[iy][group] = static_cast<int8_t>(high);
            }
        }

        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            for (int k64 = 0; k64 < 4; ++k64) {
                const size_t high_tile =
                    (static_cast<size_t>(ibl)*ntiles + ix0/16)*1536 + k64*384;
                unpack_q5_high3_vnni(packed.scaled_high + high_tile,
                    &weights_high[k64][0][0]);
            }

            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            const uint8_t * min_weights = packed.minq +
                (static_cast<size_t>(ibl)*ntiles + ix0/16)*128;
            std::memcpy(&min_weights_tile[0][0], min_weights, 128);
            const size_t di = static_cast<size_t>(ibl)*stride + ix0;
            const __m512 vd = _mm512_maskz_loadu_ps(mask, packed.d + di);
            const __m512 vdmin = _mm512_maskz_loadu_ps(mask, packed.dmin + di);

            for (int iy0 = 0; iy0 < nrc_y; iy0 += 32) {
                const int ny = std::min(32, nrc_y - iy0);
                const int ny0 = std::min(16, ny);
                const int ny1 = std::max(0, ny - 16);
                if (nx != configured_nx || ny0 != configured_ny0 || ny1 != configured_ny1) {
                    configure_tiles_q4_scaled(ny0, ny1, nx);
                    configured_nx = nx;
                    configured_ny0 = ny0;
                    configured_ny1 = ny1;
                }

                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_zero(2);
                if (ny1) _tile_zero(3);
                for (int k64 = 0; k64 < 4; ++k64) {
                    _tile_loadd(4, &activations[iy0][64*k64], QK_K);
                    if (ny1) _tile_loadd(5, &activations[iy0 + 16][64*k64], QK_K);
                    const size_t low_group =
                        (static_cast<size_t>(ibl)*ntiles + ix0/16)*8 + 2*k64;
                    _tile_loadd(6, packed.scaled_low + low_group*512, 64);
                    _tile_dpbsud(0, 4, 6);
                    if (ny1) _tile_dpbsud(1, 5, 6);
                    _tile_loadd(6, &weights_high[k64][0][0], 64);
                    _tile_dpbsud(2, 4, 6);
                    if (ny1) _tile_dpbsud(3, 5, 6);
                    tile_count += ny1 ? 4 : 2;
                }
                _tile_stored(0, &main_low[0][ix0], 64*sizeof(int32_t));
                if (ny1) _tile_stored(1, &main_low[16][ix0], 64*sizeof(int32_t));
                _tile_stored(2, &main_high[0][ix0], 64*sizeof(int32_t));
                if (ny1) _tile_stored(3, &main_high[16][ix0], 64*sizeof(int32_t));

                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_loadd(4, &min_low[iy0][0], 64);
                if (ny1) _tile_loadd(5, &min_low[iy0 + 16][0], 64);
                _tile_loadd(6, &min_weights_tile[0][0], 64);
                _tile_dpbsud(0, 4, 6);
                if (ny1) _tile_dpbsud(1, 5, 6);
                _tile_stored(0, &cmin_low[0][0], 16*sizeof(int32_t));
                if (ny1) _tile_stored(1, &cmin_low[16][0], 16*sizeof(int32_t));

                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_loadd(4, &min_high[iy0][0], 64);
                if (ny1) _tile_loadd(5, &min_high[iy0 + 16][0], 64);
                _tile_dpbsud(0, 4, 6);
                if (ny1) _tile_dpbsud(1, 5, 6);
                _tile_stored(0, &cmin_high[0][0], 16*sizeof(int32_t));
                if (ny1) _tile_stored(1, &cmin_high[16][0], 16*sizeof(int32_t));
                tile_count += ny1 ? 4 : 2;

                for (int iy = 0; iy < ny; ++iy) {
                    const __m512i vmain = _mm512_add_epi32(
                        _mm512_maskz_loadu_epi32(mask, &main_low[iy][ix0]),
                        _mm512_slli_epi32(
                            _mm512_maskz_loadu_epi32(mask, &main_high[iy][ix0]), 8));
                    const __m512i vmin = _mm512_add_epi32(
                        _mm512_maskz_loadu_epi32(mask, cmin_low[iy]),
                        _mm512_slli_epi32(_mm512_maskz_loadu_epi32(mask, cmin_high[iy]), 7));
                    const int output_y = iy0 + iy;
                    const size_t oi = static_cast<size_t>(output_y)*64 + ix0;
                    const __m512 vdy = _mm512_set1_ps(dy[output_y]);
                    __m512 vout = _mm512_maskz_loadu_ps(mask, accum + oi);
                    vout = _mm512_fmadd_ps(_mm512_mul_ps(vdy, _mm512_cvtepi32_ps(vmain)),
                                           vd, vout);
                    vout = _mm512_fnmadd_ps(_mm512_mul_ps(vdy, _mm512_cvtepi32_ps(vmin)),
                                            vdmin, vout);
                    _mm512_mask_storeu_ps(accum + oi, mask, vout);
                }
            }
        }
    }
    _tile_release();

    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            _mm512_mask_storeu_ps(info.dst_row(iy) + ix0, mask,
                _mm512_maskz_loadu_ps(mask, accum + static_cast<size_t>(iy)*64 + ix0));
        }
    }
    if (stats_enabled()) {
        stats().q5_calls.fetch_add(1, std::memory_order_relaxed);
        stats().q5_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

bool iqk_amx_mul_mat_q6_k_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y) {
    if (env_is_set("GGML_AMX_Q6_K_DISABLE") || !iqk_amx_int8_runtime_available() ||
        nrc_y < q6_min_nrc_y() || nrc_y > 512 ||
        nrc_x % 4 != 0 || nrc_x > 64 || n % QK_K != 0 || transient_weights) {
        return false;
    }

    const int nbl = n/QK_K;
    const q6_prepack & packed = get_q6_prepack(n, vx, bx, nrc_x);
    const int ntiles = packed.stride/16;
    alignas(64) float accum[nrc_y*64];
    std::memset(accum, 0, static_cast<size_t>(nrc_y)*64*sizeof(float));
    const block_q8_K * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
    }

    alignas(64) int8_t activations[nrc_y][QK_K];
    // Keep the original six-bit weights compact in persistent memory.  Build
    // the exact scale*(q-32) low/high AMX tiles here, once per output tile,
    // and reuse them across every token in this call.
    alignas(64) uint8_t weights_low[4][16][64];
    alignas(64) uint8_t weights_high[4][16][64];
    alignas(64) int32_t main_low[32][64];
    alignas(64) int32_t main_high[32][64];
    alignas(64) float dy[nrc_y];
    alignas(64) int ysum[nrc_y];
    int configured_nx = 0;
    int configured_ny0 = 0;
    int configured_ny1 = 0;
    uint64_t tile_count = 0;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            dy[iy] = y[iy][ibl].d;
            ysum[iy] = 0;
            for (int group = 0; group < QK_K/32; ++group) {
                ysum[iy] += sum_i8_32(y[iy][ibl].qs + 32*group);
            }
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                _mm512_store_si512(reinterpret_cast<__m512i *>(activations[iy] + 64*k64),
                    _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
                        y[iy][ibl].qs + 64*k64)));
            }
        }

        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const size_t tile_index = static_cast<size_t>(ibl)*ntiles + ix0/16;
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                encode_q6_scaled_vnni(
                    packed.q + tile_index*4096 + k64*1024,
                    packed.scales + tile_index*256 + k64*64,
                    &weights_low[k64][0][0], &weights_high[k64][0][0]);
            }

            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            const __m512 vd = _mm512_maskz_loadu_ps(mask,
                packed.d + static_cast<size_t>(ibl)*packed.stride + ix0);
            for (int iy0 = 0; iy0 < nrc_y; iy0 += 32) {
                const int ny = std::min(32, nrc_y - iy0);
                const int ny0 = std::min(16, ny);
                const int ny1 = std::max(0, ny - 16);
                if (nx != configured_nx || ny0 != configured_ny0 || ny1 != configured_ny1) {
                    configure_tiles_q4_scaled(ny0, ny1, nx);
                    configured_nx = nx;
                    configured_ny0 = ny0;
                    configured_ny1 = ny1;
                }
                _tile_zero(0);
                if (ny1) _tile_zero(1);
                _tile_zero(2);
                if (ny1) _tile_zero(3);
                for (int k64 = 0; k64 < QK_K/64; ++k64) {
                    _tile_loadd(4, &activations[iy0][64*k64], QK_K);
                    if (ny1) _tile_loadd(5, &activations[iy0 + 16][64*k64], QK_K);
                    _tile_loadd(6, &weights_low[k64][0][0], 64);
                    _tile_dpbsud(0, 4, 6);
                    if (ny1) _tile_dpbsud(1, 5, 6);
                    _tile_loadd(6, &weights_high[k64][0][0], 64);
                    _tile_dpbsud(2, 4, 6);
                    if (ny1) _tile_dpbsud(3, 5, 6);
                    tile_count += ny1 ? 4 : 2;
                }
                _tile_stored(0, &main_low[0][ix0], 64*sizeof(int32_t));
                if (ny1) _tile_stored(1, &main_low[16][ix0], 64*sizeof(int32_t));
                _tile_stored(2, &main_high[0][ix0], 64*sizeof(int32_t));
                if (ny1) _tile_stored(3, &main_high[16][ix0], 64*sizeof(int32_t));

                for (int iy = 0; iy < ny; ++iy) {
                    const __m512i vmain = _mm512_add_epi32(
                        _mm512_maskz_loadu_epi32(mask, &main_low[iy][ix0]),
                        _mm512_slli_epi32(
                            _mm512_maskz_loadu_epi32(mask, &main_high[iy][ix0]), 8));
                    const int output_y = iy0 + iy;
                    const __m512i vcorrected = _mm512_sub_epi32(vmain,
                        _mm512_set1_epi32(4095*ysum[output_y]));
                    const size_t oi = static_cast<size_t>(output_y)*64 + ix0;
                    __m512 vout = _mm512_maskz_loadu_ps(mask, accum + oi);
                    vout = _mm512_fmadd_ps(
                        _mm512_mul_ps(_mm512_set1_ps(dy[output_y]), _mm512_cvtepi32_ps(vcorrected)),
                        vd, vout);
                    _mm512_mask_storeu_ps(accum + oi, mask, vout);
                }
            }
        }
    }
    _tile_release();

    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            _mm512_mask_storeu_ps(info.dst_row(iy) + ix0, mask,
                _mm512_maskz_loadu_ps(mask, accum + static_cast<size_t>(iy)*64 + ix0));
        }
    }
    if (stats_enabled()) {
        stats().q6_calls.fetch_add(1, std::memory_order_relaxed);
        stats().q6_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

bool iqk_amx_mul_mat_iq3_s_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y) {
    if (env_is_set("GGML_AMX_IQ3_S_DISABLE") || !iqk_amx_int8_runtime_available() ||
        nrc_y < iq3_min_nrc_y() || nrc_y > 32 ||
        nrc_x % 4 != 0 || nrc_x > 64 || n % QK_K != 0 || transient_weights) {
        return false;
    }

    const int nbl = n/QK_K;
    const bool use_prepack = env_is_set("GGML_AMX_IQ3_S_PREPACK");
    const bool use_full_origin = use_prepack && env_is_set("GGML_AMX_IQ3_S_EXPANDED") &&
        weight_origin.base && weight_origin.row_offset >= 0 &&
        weight_origin.row_offset + nrc_x <= weight_origin.total_rows &&
        static_cast<const char *>(weight_origin.base) + static_cast<size_t>(weight_origin.row_offset)*bx == vx;
    const void * prepack_vx = use_full_origin ? weight_origin.base : vx;
    const int prepack_rows = use_full_origin ? weight_origin.total_rows : nrc_x;
    const int prepack_row_offset = use_full_origin ? weight_origin.row_offset : 0;
    const iq3_prepack * packed = use_prepack ? &get_iq3_prepack(n, prepack_vx, bx, prepack_rows) : nullptr;
    const int ntiles = packed ? packed->stride/16 : 0;
    alignas(64) float accum[32*64] = {};
    const block_q8_K * y[32];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = reinterpret_cast<const block_q8_K *>(info.src1_row(iy));
    }

    alignas(64) int8_t activations[32][QK_K];
    alignas(64) int8_t decoded_q[4096];
    alignas(64) int8_t decoded_scales[8][16];
    alignas(64) float decoded_d[16];
    alignas(64) int32_t main_low[32][64];
    alignas(64) int32_t main_high[32][64];
    alignas(64) float dy[32];
    int configured_nx = 0;
    int configured_ny0 = 0;
    int configured_ny1 = 0;
    uint64_t tile_count = 0;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            dy[iy] = y[iy][ibl].d;
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                _mm512_store_si512(reinterpret_cast<__m512i *>(activations[iy] + 64*k64),
                    _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
                        y[iy][ibl].qs + 64*k64)));
            }
        }

        for (int ix0 = 0; ix0 < nrc_x;) {
            const int cache_row = prepack_row_offset + ix0;
            const int cache_col = use_full_origin ? cache_row%16 : 0;
            const int nx = std::min(16 - cache_col, nrc_x - ix0);
            const int8_t * q = nullptr;
            const int8_t * scales = nullptr;
            const int8_t * low_tiles[4];
            const int8_t * high_tiles[4];
            const float * d;
            size_t tile_index = 0;
            if (packed) {
                tile_index = static_cast<size_t>(ibl)*ntiles + cache_row/16;
                if (!packed->expanded) {
                    q = packed->q + tile_index*4096;
                    scales = packed->scales + tile_index*128;
                }
                d = packed->d + static_cast<size_t>(ibl)*packed->stride + cache_row;
            } else {
                decode_iq3_tile(vx, bx, ibl, ix0, nx,
                    decoded_q, &decoded_scales[0][0], decoded_d);
                q = decoded_q;
                scales = &decoded_scales[0][0];
                d = decoded_d;
            }

            // IQ3_S has one odd scale per K32 group and output row.  Applying
            // that scale after the signed AMX dot product is exact, avoids
            // materializing scale*q as low+128*high, and halves the number of
            // tile dot-product instructions.  This is particularly important
            // for MoE experts, whose typical token batch is only 8--32 rows.
            if (!packed || !packed->expanded) {
                const int ny0 = std::min(16, nrc_y);
                const int ny1 = std::max(0, nrc_y - 16);
                if (nx != configured_nx || ny0 != configured_ny0 || ny1 != configured_ny1) {
                    configure_tiles_iq3_unscaled(ny0, ny1, nx);
                    configured_nx = nx;
                    configured_ny0 = ny0;
                    configured_ny1 = ny1;
                }
                const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
                const __m512 vd = _mm512_maskz_loadu_ps(mask, d);
                for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                    _tile_zero(0);
                    if (ny1) _tile_zero(1);
                    _tile_loadd(4, &activations[0][32*ib32], QK_K);
                    if (ny1) _tile_loadd(5, &activations[16][32*ib32], QK_K);
                    _tile_loadd(6, q + ib32*512 + 4*cache_col, 64);
                    _tile_dpbssd(0, 4, 6);
                    if (ny1) _tile_dpbssd(1, 5, 6);
                    _tile_stored(0, &main_low[0][ix0], 64*sizeof(int32_t));
                    if (ny1) _tile_stored(1, &main_low[16][ix0], 64*sizeof(int32_t));
                    tile_count += ny1 ? 2 : 1;

                    const __m128i scale8 = _mm_maskz_loadu_epi8(mask,
                        scales + ib32*16 + cache_col);
                    const __m512 vdscale = _mm512_mul_ps(vd,
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(scale8)));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        const __m512i vdot = _mm512_maskz_loadu_epi32(mask,
                            &main_low[iy][ix0]);
                        float * out = accum + static_cast<size_t>(iy)*64 + ix0;
                        __m512 vout = _mm512_maskz_loadu_ps(mask, out);
                        vout = _mm512_fmadd_ps(
                            _mm512_mul_ps(_mm512_set1_ps(dy[iy]),
                                _mm512_cvtepi32_ps(vdot)),
                            vdscale, vout);
                        _mm512_mask_storeu_ps(out, mask, vout);
                    }
                }
                ix0 += nx;
                continue;
            }

            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                low_tiles[k64] = packed->low + tile_index*4096 + k64*1024 + 4*cache_col;
                high_tiles[k64] = packed->high + tile_index*4096 + k64*1024 + 4*cache_col;
            }

            const int ny0 = std::min(16, nrc_y);
            const int ny1 = std::max(0, nrc_y - 16);
            if (nx != configured_nx || ny0 != configured_ny0 || ny1 != configured_ny1) {
                configure_tiles_q4_scaled(ny0, ny1, nx);
                configured_nx = nx;
                configured_ny0 = ny0;
                configured_ny1 = ny1;
            }
            _tile_zero(0);
            if (ny1) _tile_zero(1);
            _tile_zero(2);
            if (ny1) _tile_zero(3);
            for (int k64 = 0; k64 < QK_K/64; ++k64) {
                _tile_loadd(4, &activations[0][64*k64], QK_K);
                if (ny1) _tile_loadd(5, &activations[16][64*k64], QK_K);
                _tile_loadd(6, low_tiles[k64], 64);
                _tile_dpbssd(0, 4, 6);
                if (ny1) _tile_dpbssd(1, 5, 6);
                _tile_loadd(6, high_tiles[k64], 64);
                _tile_dpbssd(2, 4, 6);
                if (ny1) _tile_dpbssd(3, 5, 6);
                tile_count += ny1 ? 4 : 2;
            }
            _tile_stored(0, &main_low[0][ix0], 64*sizeof(int32_t));
            if (ny1) _tile_stored(1, &main_low[16][ix0], 64*sizeof(int32_t));
            _tile_stored(2, &main_high[0][ix0], 64*sizeof(int32_t));
            if (ny1) _tile_stored(3, &main_high[16][ix0], 64*sizeof(int32_t));

            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            const __m512 vd = _mm512_maskz_loadu_ps(mask, d);
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m512i vdot = _mm512_add_epi32(
                    _mm512_maskz_loadu_epi32(mask, &main_low[iy][ix0]),
                    _mm512_slli_epi32(
                        _mm512_maskz_loadu_epi32(mask, &main_high[iy][ix0]), 7));
                float * out = accum + static_cast<size_t>(iy)*64 + ix0;
                __m512 vout = _mm512_maskz_loadu_ps(mask, out);
                vout = _mm512_fmadd_ps(
                    _mm512_mul_ps(_mm512_set1_ps(dy[iy]), _mm512_cvtepi32_ps(vdot)),
                    vd, vout);
                _mm512_mask_storeu_ps(out, mask, vout);
            }
            ix0 += nx;
        }
    }
    _tile_release();

    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int nx = std::min(16, nrc_x - ix0);
            const __mmask16 mask = static_cast<__mmask16>((1u << nx) - 1u);
            _mm512_mask_storeu_ps(info.dst_row(iy) + ix0, mask,
                _mm512_maskz_loadu_ps(mask, accum + static_cast<size_t>(iy)*64 + ix0));
        }
    }
    if (stats_enabled()) {
        stats().iq3_calls.fetch_add(1, std::memory_order_relaxed);
        stats().iq3_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

bool iqk_amx_mul_mat_q8_0_r8(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y) {
    if (env_is_set("GGML_AMX_Q8_0_R8_DISABLE") || !iqk_amx_int8_runtime_available() ||
        nrc_y < q8_0_min_nrc_y() || nrc_y > 32 ||
        nrc_x % 16 != 0 || nrc_x > 64 || n % QK8_0 != 0) {
        return false;
    }

    const int nblocks = n/QK8_0;
    const int ny0 = std::min(16, nrc_y);
    const int ny1 = std::max(0, nrc_y - 16);
    const char * y[32];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = info.src1_row(iy);
    }

    alignas(64) int8_t activations[32][QK8_0];
    alignas(64) int8_t packed_weights[8][64];
    alignas(64) int32_t dots[32][16];
    alignas(64) float dy[32];
    uint64_t tile_count = 0;
    configure_tiles_iq3_unscaled(ny0, ny1, 16);

    for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
        __m512 accum[32];
        for (int iy = 0; iy < nrc_y; ++iy) {
            accum[iy] = _mm512_setzero_ps();
        }
        const auto * weights0 = reinterpret_cast<const block_q8_0_r8 *>(
            static_cast<const char *>(vx) + static_cast<size_t>(ix0)*bx);
        const auto * weights1 = reinterpret_cast<const block_q8_0_r8 *>(
            static_cast<const char *>(vx) + static_cast<size_t>(ix0 + 8)*bx);
        for (int ib = 0; ib < nblocks; ++ib) {
            const int8_t * first_quants = nullptr;
            for (int iy = 0; iy < nrc_y; ++iy) {
                const int8_t * quants;
                int ignored_sum;
                q8_2_block(y[iy], nblocks, ib, quants, dy[iy], ignored_sum);
                if (!first_quants) first_quants = quants;
                if (info.row_mapping) {
                    _mm256_store_si256(reinterpret_cast<__m256i *>(activations[iy]),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(quants)));
                }
            }

            const int8_t * activation0 = info.row_mapping ? activations[0] : first_quants;
            const int activation_stride = info.row_mapping ? QK8_0 : static_cast<int>(info.by);
            for (int kr = 0; kr < QK8_0/4; ++kr) {
                __m512i values = _mm512_castsi256_si512(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(weights0[ib].qs + 32*kr)));
                values = _mm512_inserti64x4(values, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(weights1[ib].qs + 32*kr)), 1);
                _mm512_store_si512(reinterpret_cast<__m512i *>(packed_weights[kr]), values);
            }
            _tile_zero(0);
            if (ny1) _tile_zero(1);
            _tile_loadd(4, activation0, activation_stride);
            if (ny1) _tile_loadd(5, activation0 + 16*activation_stride, activation_stride);
            _tile_loadd(6, &packed_weights[0][0], 16*4);
            _tile_dpbssd(0, 4, 6);
            if (ny1) _tile_dpbssd(1, 5, 6);
            _tile_stored(0, &dots[0][0], 16*sizeof(int32_t));
            if (ny1) _tile_stored(1, &dots[16][0], 16*sizeof(int32_t));
            tile_count += ny1 ? 2 : 1;

            const __m512 vd = _mm512_insertf32x8(
                _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(weights0[ib].d)))),
                _mm256_cvtph_ps(_mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(weights1[ib].d))), 1);
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m512 vdot = _mm512_cvtepi32_ps(_mm512_load_si512(
                    reinterpret_cast<const __m512i *>(dots[iy])));
                accum[iy] = _mm512_fmadd_ps(
                    _mm512_mul_ps(vd, _mm512_set1_ps(dy[iy])), vdot, accum[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix0, iy, accum[iy]);
        }
    }
    _tile_release();

    if (stats_enabled()) {
        stats().q8_calls.fetch_add(1, std::memory_order_relaxed);
        stats().q8_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

bool iqk_amx_mul_mat_iq4_nl_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y) {
    if (env_is_set("GGML_AMX_IQ4_NL_DISABLE") || !iqk_amx_int8_runtime_available() ||
        nrc_y < iq4_min_nrc_y() || nrc_y > 32 || n < iq4_min_k() ||
        nrc_x % 4 != 0 || n % QK4_NL != 0 || transient_weights) {
        return false;
    }

    const int nblocks = n/QK4_NL;
    const iq4_prepack & packed = get_iq4_prepack(n, vx, bx, nrc_x);
    alignas(64) float stack_accum[64*32] = {};
    std::vector<float> heap_accum;
    float * accum = stack_accum;
    if (nrc_x > 64) {
        heap_accum.resize(static_cast<size_t>(nrc_x)*nrc_y, 0.0f);
        accum = heap_accum.data();
    }
    const char * y[32];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = info.src1_row(iy);
    }

    alignas(64) uint8_t a[16][32];
    alignas(64) int8_t b0[8*64];
    alignas(64) int8_t b1[8*64];
    alignas(64) int32_t c[16*32];
    const int8_t * b_rows[32];
    uint64_t tile_count = 0;
    int configured_m = 0;

    for (int ib = 0; ib < nblocks; ++ib) {
        alignas(64) float dy[32] = {};
        alignas(64) int ysum[32] = {};
        for (int iy = 0; iy < nrc_y; ++iy) {
            q8_2_block(y[iy], nblocks, ib, b_rows[iy], dy[iy], ysum[iy]);
        }
        const int ny0 = std::min(16, nrc_y);
        const int ny1 = std::max(0, nrc_y - 16);
        pack_b_vnni(b_rows, ny0, b0);
        if (ny1) {
            pack_b_vnni(b_rows + 16, ny1, b1);
        }

        for (int ix0 = 0; ix0 < nrc_x; ix0 += 16) {
            const int this_m = std::min(16, nrc_x - ix0);
            const size_t pi = static_cast<size_t>(ib)*nrc_x + ix0;
            const float * dscale = packed.dscale.data() + pi;
            unpack_iq4_tile(packed.q.data() + pi*16, this_m, &a[0][0]);

            if (this_m != configured_m) {
                configure_tiles(this_m, nrc_y);
                configured_m = this_m;
            }
            amx_dot_tile(&a[0][0], b0, b1, c, nrc_y);
            ++tile_count;
            const __mmask16 mask0 = static_cast<__mmask16>((1u << ny0) - 1u);
            const __mmask16 mask1 = static_cast<__mmask16>((1u << ny1) - 1u);
            const __m512 vdy0 = _mm512_maskz_loadu_ps(mask0, dy);
            const __m512i vysum0 = _mm512_maskz_loadu_epi32(mask0, ysum);
            const __m512 vdy1 = _mm512_maskz_loadu_ps(mask1, dy + 16);
            const __m512i vysum1 = _mm512_maskz_loadu_epi32(mask1, ysum + 16);
            for (int ir = 0; ir < this_m; ++ir) {
                float * out = accum + static_cast<size_t>(ix0 + ir)*nrc_y;
                const __m512 vdscale = _mm512_set1_ps(dscale[ir]);
                const __m512i vc0 = _mm512_maskz_loadu_epi32(mask0, c + ir*nrc_y);
                const __m512 vdot0 = _mm512_cvtepi32_ps(
                    _mm512_sub_epi32(vc0, _mm512_slli_epi32(vysum0, 7)));
                __m512 vout0 = _mm512_maskz_loadu_ps(mask0, out);
                vout0 = _mm512_fmadd_ps(_mm512_mul_ps(vdy0, vdot0), vdscale, vout0);
                _mm512_mask_storeu_ps(out, mask0, vout0);
                if (ny1) {
                    const __m512i vc1 = _mm512_maskz_loadu_epi32(mask1, c + ir*nrc_y + 16);
                    const __m512 vdot1 = _mm512_cvtepi32_ps(
                        _mm512_sub_epi32(vc1, _mm512_slli_epi32(vysum1, 7)));
                    __m512 vout1 = _mm512_maskz_loadu_ps(mask1, out + 16);
                    vout1 = _mm512_fmadd_ps(_mm512_mul_ps(vdy1, vdot1), vdscale, vout1);
                    _mm512_mask_storeu_ps(out + 16, mask1, vout1);
                }
            }
        }
    }
    _tile_release();

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accum[static_cast<size_t>(ix)*nrc_y + iy]);
        }
    }
    if (stats_enabled()) {
        stats().iq4_calls.fetch_add(1, std::memory_order_relaxed);
        stats().iq4_tiles.fetch_add(tile_count, std::memory_order_relaxed);
    }
    return true;
}

#else

bool iqk_amx_int8_runtime_available() {
    return false;
}

void iqk_amx_set_transient_weights(bool) {
}

void iqk_amx_set_weight_origin(const void *, int, int) {
}

bool iqk_amx_mul_mat_q4_k_r4(
        int, const void *, size_t, const DataInfo &, int, int) {
    return false;
}

bool iqk_amx_mul_mat_q5_k_r4(
        int, const void *, size_t, const DataInfo &, int, int) {
    return false;
}

bool iqk_amx_mul_mat_q6_k_r4(
        int, const void *, size_t, const DataInfo &, int, int) {
    return false;
}

bool iqk_amx_mul_mat_iq3_s_r4(
        int, const void *, size_t, const DataInfo &, int, int) {
    return false;
}

bool iqk_amx_mul_mat_q8_0_r8(
        int, const void *, size_t, const DataInfo &, int, int) {
    return false;
}

bool iqk_amx_mul_mat_iq4_nl_r4(
        int, const void *, size_t, const DataInfo &, int, int) {
    return false;
}

#endif

#endif
