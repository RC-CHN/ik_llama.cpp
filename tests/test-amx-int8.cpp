#include "ggml.h"
#include "ggml-common.h"
#include "iqk_mul_mat.h"
#include "iqk_quantize.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

void dequantize_r4(enum ggml_type type, const void * src, float * dst, int m, int k, size_t row_size) {
    static constexpr int8_t iq4_values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    if (type == GGML_TYPE_Q8_0_R8) {
        for (int row = 0; row < m; row += 8) {
            dequantize_row_q8_0_r8(
                reinterpret_cast<const block_q8_0_r8 *>(
                    static_cast<const char *>(src) + static_cast<size_t>(row)*row_size),
                dst + static_cast<size_t>(row)*k, 8LL*k);
        }
        return;
    }
    for (int row = 0; row < m; row += 4) {
        const char * group = static_cast<const char *>(src) + static_cast<size_t>(row)*row_size;
        if (type == GGML_TYPE_Q6_K_R4) {
            dequantize_row_q6_k_r4(
                reinterpret_cast<const block_q6_k_r4 *>(group),
                dst + static_cast<size_t>(row)*k, 4LL*k);
            continue;
        }
        if (type == GGML_TYPE_Q5_K_R4) {
            dequantize_row_q5_k_r4(
                reinterpret_cast<const block_q5_k_r4 *>(group),
                dst + static_cast<size_t>(row)*k, 4LL*k);
            continue;
        }
        if (type == GGML_TYPE_IQ3_S_R4) {
            dequantize_row_iq3_s_r4(
                reinterpret_cast<const block_iq3_s_r4 *>(group),
                dst + static_cast<size_t>(row)*k, 4LL*k);
            continue;
        }
        if (type == GGML_TYPE_Q4_K_R4) {
            const auto * blocks = reinterpret_cast<const block_q4_k_r4 *>(group);
            for (int ibl = 0; ibl < k/QK_K; ++ibl) {
                const block_q4_k_r4 & block = blocks[ibl];
                for (int lane = 0; lane < 4; ++lane) {
                    const float d = ggml_fp16_to_fp32(block.d[lane]);
                    const float dmin = ggml_fp16_to_fp32(block.d[lane + 4]);
                    float * out = dst + static_cast<size_t>(row + lane)*k + ibl*QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const int si = 4*ib32 + lane;
                        const uint8_t low = block.scales_l[si];
                        const uint8_t high = (block.scales_h[si%16] >> (4*(si/16))) & 0x0f;
                        const float ds = d*((low & 0x0f) | ((high & 0x03) << 4));
                        const float ms = dmin*((low >> 4) | ((high >> 2) << 4));
                        for (int i = 0; i < 4; ++i) {
                            const uint8_t b0 = block.qs[64*ib32 + 4*lane + i +  0];
                            const uint8_t b1 = block.qs[64*ib32 + 4*lane + i + 16];
                            const uint8_t b2 = block.qs[64*ib32 + 4*lane + i + 32];
                            const uint8_t b3 = block.qs[64*ib32 + 4*lane + i + 48];
                            out[32*ib32 + i +  0] = ds*(b0 & 15) - ms;
                            out[32*ib32 + i +  8] = ds*(b0 >> 4) - ms;
                            out[32*ib32 + i + 16] = ds*(b1 & 15) - ms;
                            out[32*ib32 + i + 24] = ds*(b1 >> 4) - ms;
                            out[32*ib32 + i +  4] = ds*(b2 & 15) - ms;
                            out[32*ib32 + i + 12] = ds*(b2 >> 4) - ms;
                            out[32*ib32 + i + 20] = ds*(b3 & 15) - ms;
                            out[32*ib32 + i + 28] = ds*(b3 >> 4) - ms;
                        }
                    }
                }
            }
        } else {
            const auto * blocks = reinterpret_cast<const block_iq4_nl_r4 *>(group);
            for (int ib = 0; ib < k/QK4_NL; ++ib) {
                const block_iq4_nl_r4 & block = blocks[ib];
                for (int lane = 0; lane < 4; ++lane) {
                    const float d = ggml_fp16_to_fp32(block.d[lane]);
                    float * out = dst + static_cast<size_t>(row + lane)*k + ib*QK4_NL;
                    for (int i = 0; i < 4; ++i) {
                        const uint8_t b0 = block.qs[4*lane + i +  0];
                        const uint8_t b1 = block.qs[4*lane + i + 16];
                        const uint8_t b2 = block.qs[4*lane + i + 32];
                        const uint8_t b3 = block.qs[4*lane + i + 48];
                        out[i +  0] = d*iq4_values[b0 & 15];
                        out[i +  8] = d*iq4_values[b0 >> 4];
                        out[i + 16] = d*iq4_values[b1 & 15];
                        out[i + 24] = d*iq4_values[b1 >> 4];
                        out[i +  4] = d*iq4_values[b2 & 15];
                        out[i + 12] = d*iq4_values[b2 >> 4];
                        out[i + 20] = d*iq4_values[b3 & 15];
                        out[i + 28] = d*iq4_values[b3 >> 4];
                    }
                }
            }
        }
    }
}

void dequantize_q8_k32(const void * src, float * dst, int n, int k, size_t row_size) {
    for (int row = 0; row < n; ++row) {
        const auto * blocks = reinterpret_cast<const block_q8_K *>(
                static_cast<const char *>(src) + static_cast<size_t>(row)*row_size);
        for (int ib = 0; ib < k/QK_K; ++ib) {
            for (int i = 0; i < QK_K; ++i) {
                dst[static_cast<size_t>(row)*k + ib*QK_K + i] = blocks[ib].d*blocks[ib].qs[i];
            }
        }
    }
}

void dequantize_q8_2_x4(const void * src, float * dst, int n, int k, size_t row_size) {
    const int nb = k/QK8_2;
    const int nb4 = 4*(nb/4);
    for (int row = 0; row < n; ++row) {
        const char * base = static_cast<const char *>(src) + static_cast<size_t>(row)*row_size;
        for (int ib = 0; ib < nb; ++ib) {
            const int8_t * qs;
            float d;
            if (ib < nb4) {
                const auto * blocks = reinterpret_cast<const block_q8_2_x4 *>(base);
                const int lane = ib%4;
                qs = blocks[ib/4].qs + lane*QK8_2;
                d = ggml_bf16_to_fp32(ggml_bf16_t{blocks[ib/4].d[lane]});
            } else {
                const char * tail = base + static_cast<size_t>(nb4)*sizeof(block_q8_2);
                const auto * block = reinterpret_cast<const block_q8_2 *>(tail) + (ib - nb4);
                qs = block->qs;
                d = ggml_bf16_to_fp32(ggml_bf16_t{block->d});
            }
            for (int i = 0; i < QK8_2; ++i) {
                dst[static_cast<size_t>(row)*k + ib*QK8_2 + i] = d*qs[i];
            }
        }
    }
}

bool run_case(enum ggml_type type_a, enum ggml_type type_b, int m, int n, int k, int nth) {
    std::mt19937 rng(0x9470cU + type_a + 17*n + 31*nth);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(static_cast<size_t>(m)*k);
    std::vector<float> b(static_cast<size_t>(n)*k);
    for (float & value : a) value = dist(rng);
    for (float & value : b) value = dist(rng);

    const size_t row_a = ggml_row_size(type_a, k);
    const size_t row_b = ggml_row_size(type_b, k);
    std::vector<uint8_t> qa(static_cast<size_t>(m)*row_a);
    std::vector<uint8_t> qb(static_cast<size_t>(n)*row_b);
    const size_t bytes_a = ggml_quantize_chunk(type_a, a.data(), qa.data(), 0, m, k, nullptr, nullptr);
    if (type_b == GGML_TYPE_Q8_K32 || type_b == GGML_TYPE_Q8_K) {
        if (type_b == GGML_TYPE_Q8_K) {
            iqk_quantize_row_q8_K(b.data(), qb.data(), static_cast<int64_t>(n)*k);
        } else {
        quantize_row_q8_K32(b.data(), qb.data(), static_cast<int64_t>(n)*k);
        }
    } else {
        quantize_row_q8_2_x4(b.data(), qb.data(), static_cast<int64_t>(n)*k);
    }
    const size_t bytes_b = qb.size();
    if (bytes_a != qa.size() || bytes_b != qb.size()) {
        std::fprintf(stderr, "unexpected quantized size for %s x %s: %zu/%zu, %zu/%zu\n",
                ggml_type_name(type_a), ggml_type_name(type_b),
                bytes_a, qa.size(), bytes_b, qb.size());
        return false;
    }

    std::vector<float> da(static_cast<size_t>(m)*k);
    std::vector<float> db(static_cast<size_t>(n)*k);
    dequantize_r4(type_a, qa.data(), da.data(), m, k, row_a);
    if (type_b == GGML_TYPE_Q8_K32 || type_b == GGML_TYPE_Q8_K) {
        dequantize_q8_k32(qb.data(), db.data(), n, k, row_b);
    } else {
        dequantize_q8_2_x4(qb.data(), db.data(), n, k, row_b);
    }
    float range_a = 0.0f;
    float range_b = 0.0f;
    for (float value : da) range_a = std::max(range_a, std::abs(value));
    for (float value : db) range_b = std::max(range_b, std::abs(value));
    if (range_a == 0.0f || range_b == 0.0f) {
        if (type_a == GGML_TYPE_Q4_K_R4) {
            const auto * block = reinterpret_cast<const block_q4_k_r4 *>(qa.data());
            std::fprintf(stderr, "first q4 scales bits: %u %u %u %u / %u %u %u %u\n",
                    block->d[0], block->d[1], block->d[2], block->d[3],
                    block->d[4], block->d[5], block->d[6], block->d[7]);
        } else {
            const auto * block = reinterpret_cast<const block_iq4_nl_r4 *>(qa.data());
            std::fprintf(stderr, "first iq4 scales bits: %u %u %u %u\n",
                    block->d[0], block->d[1], block->d[2], block->d[3]);
        }
        std::fprintf(stderr, "invalid dequantized ranges for %s x %s: %g/%g\n",
                ggml_type_name(type_a), ggml_type_name(type_b), range_a, range_b);
        return false;
    }

    std::vector<float> result(static_cast<size_t>(m)*n, 0.0f);
    std::vector<std::thread> workers;
    for (int ith = 0; ith < nth; ++ith) {
        workers.emplace_back([&, ith] {
            if (!iqk_mul_mat(m, n, k,
                        type_a, qa.data(), row_a,
                        type_b, qb.data(), row_b,
                        result.data(), m, ith, nth)) {
                std::fprintf(stderr, "iqk_mul_mat rejected %s x %s\n",
                        ggml_type_name(type_a), ggml_type_name(type_b));
                std::abort();
            }
        });
    }
    for (std::thread & worker : workers) worker.join();

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float max_expected = 0.0f;
    float max_actual = 0.0f;
    for (int iy = 0; iy < n; ++iy) {
        for (int ix = 0; ix < m; ++ix) {
            float expected = 0.0f;
            for (int ik = 0; ik < k; ++ik) {
                expected += da[static_cast<size_t>(ix)*k + ik]*db[static_cast<size_t>(iy)*k + ik];
            }
            const float actual = result[static_cast<size_t>(iy)*m + ix];
            max_expected = std::max(max_expected, std::abs(expected));
            max_actual = std::max(max_actual, std::abs(actual));
            const float abs_error = std::abs(expected - actual);
            const float rel_error = abs_error/std::max(1.0f, std::abs(expected));
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, rel_error);
            if (abs_error > 1.0e-2f && rel_error > 5.0e-4f) {
                std::fprintf(stderr,
                        "%s mismatch at (%d,%d): expected %.9g, got %.9g, abs=%g rel=%g\n",
                        ggml_type_name(type_a), ix, iy, expected, actual, abs_error, rel_error);
                return false;
            }
        }
    }
    std::printf("PASS %s x %s %dx%dx%d threads=%d max_abs=%g max_rel=%g range=%g/%g inputs=%g/%g\n",
            ggml_type_name(type_a), ggml_type_name(type_b), m, n, k, nth,
            max_abs, max_rel, max_expected, max_actual, range_a, range_b);
    return true;
}

bool bench_case(enum ggml_type type_a, enum ggml_type type_b, int n = 512, int reps = 32,
        int m = 1024, int k = 5120) {
    constexpr int warmup = 3;
    std::mt19937 rng(0x9470cU + type_a);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(static_cast<size_t>(m)*k);
    std::vector<float> b(static_cast<size_t>(n)*k);
    for (float & value : a) value = dist(rng);
    for (float & value : b) value = dist(rng);

    const size_t row_a = ggml_row_size(type_a, k);
    const size_t row_b = ggml_row_size(type_b, k);
    std::vector<uint8_t> qa(static_cast<size_t>(m)*row_a);
    std::vector<uint8_t> qb(static_cast<size_t>(n)*row_b);
    ggml_quantize_chunk(type_a, a.data(), qa.data(), 0, m, k, nullptr, nullptr);
    if (type_b == GGML_TYPE_Q8_K32 || type_b == GGML_TYPE_Q8_K) {
        if (type_b == GGML_TYPE_Q8_K) {
            iqk_quantize_row_q8_K(b.data(), qb.data(), static_cast<int64_t>(n)*k);
        } else {
        quantize_row_q8_K32(b.data(), qb.data(), static_cast<int64_t>(n)*k);
        }
    } else {
        quantize_row_q8_2_x4(b.data(), qb.data(), static_cast<int64_t>(n)*k);
    }
    a.clear();
    b.clear();
    std::vector<float> result(static_cast<size_t>(m)*n);
    auto invoke = [&] {
        return iqk_mul_mat(m, n, k, type_a, qa.data(), row_a,
            type_b, qb.data(), row_b, result.data(), m, 0, 1);
    };
    for (int i = 0; i < warmup; ++i) {
        if (!invoke()) return false;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        if (!invoke()) return false;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    for (float value : result) checksum += value;
    const double gmac_s = static_cast<double>(m)*n*k*reps/seconds/1.0e9;
    std::printf("BENCH %s x %s M=%d N=%d K=%d: %.3f ms, %.2f GMAC/s checksum=%.9g\n",
        ggml_type_name(type_a), ggml_type_name(type_b), m, n, k,
        1000.0*seconds/reps, gmac_s, checksum);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_init_params params = {};
    params.mem_size = 1024*1024;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 1;

    const bool benchmark = argc == 2 && std::string(argv[1]) == "--bench";
    const bool benchmark_iq4_moe = argc == 2 && std::string(argv[1]) == "--bench-iq4-moe";
    const bool benchmark_iq3_moe = argc == 2 && std::string(argv[1]) == "--bench-iq3-moe";
    const bool benchmark_q8 = argc == 2 && std::string(argv[1]) == "--bench-q8";
    const bool benchmark_q5 = argc == 2 && std::string(argv[1]) == "--bench-q5";
    const bool benchmark_q4 = argc == 2 && std::string(argv[1]) == "--bench-q4";
    if (!benchmark && !benchmark_iq4_moe && !benchmark_iq3_moe && !benchmark_q8 && !benchmark_q5 && !benchmark_q4) {
#ifdef _WIN32
        _putenv_s("GGML_AMX_Q4_K_MAX_NY", "512");
        _putenv_s("GGML_AMX_Q6_K_MAX_NY", "512");
#else
        setenv("GGML_AMX_Q4_K_MAX_NY", "512", 1);
        setenv("GGML_AMX_Q6_K_MAX_NY", "512", 1);
#endif
    }
    const bool ok = benchmark_q4 ?
        (bench_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32,  32, 512) &&
         bench_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32,  64, 256) &&
         bench_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 128, 128) &&
         bench_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 256,  64) &&
         bench_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 512,  32)) : benchmark_q5 ?
        (bench_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32,   8, 2048) &&
         bench_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32,  16, 1024) &&
         bench_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32,  32,  512) &&
         bench_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32,  64,  256) &&
         bench_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32, 512,   32)) : benchmark_q8 ?
        (bench_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4,  1, 32768, 2048, 2048) &&
         bench_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4,  8, 4096, 2048, 2048) &&
         bench_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 16, 2048, 2048, 2048) &&
         bench_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 32, 1024, 2048, 2048) &&
         bench_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 64,  512, 2048, 2048)) : benchmark_iq3_moe ?
        (bench_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K,  1, 32768, 512, 2048) &&
         bench_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K,  4, 20480, 512, 2048) &&
         bench_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K,  8, 10240, 512, 2048) &&
         bench_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K, 16,  5120, 512, 2048) &&
         bench_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K, 32,  2560, 512, 2048) &&
         bench_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K, 64,  1280, 512, 2048)) : benchmark_iq4_moe ?
        (bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4,  8, 10240, 2048, 512) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 16,  5120, 2048, 512) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 32,  2560, 2048, 512) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 64,  1280, 2048, 512)) : benchmark ?
        (bench_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32) &&
         bench_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   8, 2048) &&
         bench_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,  16, 1024) &&
         bench_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,  32,  512) &&
         bench_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,  64,  256) &&
         bench_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K, 512,   32) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4,   8, 2048) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4,  16, 1024) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4,  32,  512) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4,  64,  256) &&
         bench_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 512,   32)) :
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64, 512, 1024, 1) &&
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64, 256, 1024, 1) &&
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64, 128, 1024, 1) &&
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64, 64, 1024, 1) &&
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64, 32, 1024, 1) &&
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64, 16, 1024, 1) &&
        run_case(GGML_TYPE_Q4_K_R4, GGML_TYPE_Q8_K32, 64,  8, 1024, 4) &&
        run_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32, 64, 512, 1024, 1) &&
        run_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32, 64,  64, 1024, 1) &&
        run_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32, 64,  16, 1024, 1) &&
        run_case(GGML_TYPE_Q5_K_R4, GGML_TYPE_Q8_K32, 64,   8, 1024, 4) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64, 512, 1024, 1) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64, 256, 1024, 1) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64, 128, 1024, 1) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64, 64, 1024, 1) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64, 32, 1024, 1) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64, 16, 1024, 1) &&
        run_case(GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   64,  8, 1024, 4) &&
        run_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K,  64, 32, 1024, 1) &&
        run_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K,  64, 16, 1024, 1) &&
        run_case(GGML_TYPE_IQ3_S_R4, GGML_TYPE_Q8_K,  64,  8, 1024, 1) &&
        run_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 64, 32, 1024, 1) &&
        run_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 64, 16, 1024, 1) &&
        run_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 64,  8, 1024, 1) &&
        run_case(GGML_TYPE_Q8_0_R8, GGML_TYPE_Q8_2_X4, 64,  1, 1024, 1) &&
        run_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 64, 32, 1024, 1) &&
        run_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 64, 16, 1024, 1) &&
        run_case(GGML_TYPE_IQ4_NL_R4, GGML_TYPE_Q8_2_X4, 64,  8, 1024, 4);
    ggml_free(ctx);
    return ok ? 0 : 1;
}
