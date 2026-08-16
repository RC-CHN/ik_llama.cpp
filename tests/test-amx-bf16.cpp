#include "ggml.h"
#include "iqk_mul_mat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

bool iqk_fa_256_256(int int_type_k, int int_type_v, int nq, int nk,
        int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
        const float * q, const void * k, const void * v, const void * mask,
        float scale, float softcap, float * qkv, const float * sinkf,
        int sink_stride, float * M, float * S);

namespace {

struct matrix_case {
    int m;
    int n;
    int k;
    int nth;
};

bool run_case(const matrix_case & tc, bool report = true) {
    std::mt19937 rng(0x5a17u + tc.m + 17*tc.n + 31*tc.k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<ggml_bf16_t> a(static_cast<size_t>(tc.m)*tc.k);
    std::vector<ggml_bf16_t> b(static_cast<size_t>(tc.n)*tc.k);
    for (auto & value : a) value = ggml_fp32_to_bf16(dist(rng));
    for (auto & value : b) value = ggml_fp32_to_bf16(dist(rng));

    std::vector<float> result(static_cast<size_t>(tc.m)*tc.n, 0.0f);
    std::vector<std::thread> workers;
    for (int ith = 0; ith < tc.nth; ++ith) {
        workers.emplace_back([&, ith] {
            const bool handled = iqk_mul_mat(tc.m, tc.n, tc.k,
                GGML_TYPE_BF16, a.data(), tc.k*sizeof(ggml_bf16_t),
                GGML_TYPE_BF16, b.data(), tc.k*sizeof(ggml_bf16_t),
                result.data(), tc.m, ith, tc.nth);
            if (!handled) {
                std::fprintf(stderr, "iqk_mul_mat rejected BF16 case %dx%dx%d\n", tc.m, tc.n, tc.k);
                std::abort();
            }
        });
    }
    for (auto & worker : workers) worker.join();

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (int iy = 0; iy < tc.n; ++iy) {
        for (int ix = 0; ix < tc.m; ++ix) {
            float expected = 0.0f;
            for (int ik = 0; ik < tc.k; ++ik) {
                expected += ggml_bf16_to_fp32(a[static_cast<size_t>(ix)*tc.k + ik]) *
                            ggml_bf16_to_fp32(b[static_cast<size_t>(iy)*tc.k + ik]);
            }
            const float actual = result[static_cast<size_t>(iy)*tc.m + ix];
            const float abs_error = std::abs(expected - actual);
            const float rel_error = abs_error / std::max(1.0f, std::abs(expected));
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            if (abs_error > 2.0e-3f && rel_error > 2.0e-4f) {
                std::fprintf(stderr,
                    "BF16 mismatch for %dx%dx%d at (%d,%d): expected %.9g, got %.9g\n",
                    tc.m, tc.n, tc.k, ix, iy, expected, actual);
                return false;
            }
        }
    }
    if (report) {
        std::printf("PASS BF16 %dx%dx%d, threads=%d, max_abs=%g, max_rel=%g\n",
            tc.m, tc.n, tc.k, tc.nth, max_abs_error, max_rel_error);
    }
    return true;
}

bool run_fa_case(int nq, int nk) {
    constexpr int d = 256;

    std::vector<float> q(static_cast<size_t>(nq)*d);
    std::vector<ggml_bf16_t> k(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> v(static_cast<size_t>(nk)*d);
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(nq)*nk);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.25f*std::sin(0.013f*static_cast<float>(i));
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = ggml_fp32_to_bf16(0.25f*std::cos(0.007f*static_cast<float>(i)));
        v[i] = ggml_fp32_to_bf16(std::sin(0.009f*static_cast<float>(i)));
    }
    for (int iq = 0; iq < nq; ++iq) {
        const int visible = nk - (nq - 1 - iq)%17;
        for (int ik = 0; ik < nk; ++ik) {
            mask[static_cast<size_t>(iq)*nk + ik] =
                ggml_fp32_to_fp16(ik < visible ? 0.0f : -INFINITY);
        }
    }

    std::vector<float> avx(static_cast<size_t>(nq)*d);
    std::vector<float> amx(static_cast<size_t>(nq)*d);
    std::vector<float> avx_m(nq), avx_s(nq), amx_m(nq), amx_s(nq);
    bool avx_handled = false;

    // AMX availability is cached per thread. Run the forced-AVX reference on
    // a disposable thread, then let the main thread exercise AMX normally.
    setenv("GGML_AMX_FA_QK", "0", 1);
    std::thread avx_worker([&] {
        avx_handled = iqk_fa_256_256(GGML_TYPE_BF16, GGML_TYPE_BF16, nq, nk,
            d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
            nk*sizeof(ggml_fp16_t), d, q.data(), k.data(), v.data(), mask.data(),
            1.0f/std::sqrt(static_cast<float>(d)), 0.0f, avx.data(), nullptr, 0,
            avx_m.data(), avx_s.data());
    });
    avx_worker.join();
    unsetenv("GGML_AMX_FA_QK");

    const bool amx_handled = iqk_fa_256_256(GGML_TYPE_BF16, GGML_TYPE_BF16, nq, nk,
        d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
        nk*sizeof(ggml_fp16_t), d, q.data(), k.data(), v.data(), mask.data(),
        1.0f/std::sqrt(static_cast<float>(d)), 0.0f, amx.data(), nullptr, 0,
        amx_m.data(), amx_s.data());
    if (!avx_handled || !amx_handled) {
        std::fprintf(stderr, "flash attention rejected nq=%d, nk=%d\n", nq, nk);
        return false;
    }

    double squared_error = 0.0;
    double squared_reference = 0.0;
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < avx.size(); ++i) {
        if (!std::isfinite(avx[i]) || !std::isfinite(amx[i])) {
            std::fprintf(stderr, "non-finite flash attention result at %zu\n", i);
            return false;
        }
        const float error = amx[i] - avx[i];
        max_abs_error = std::max(max_abs_error, std::abs(error));
        squared_error += static_cast<double>(error)*error;
        squared_reference += static_cast<double>(avx[i])*avx[i];
    }
    const double relative_rms = std::sqrt(squared_error/std::max(1.0e-30, squared_reference));

    float max_stats_error = 0.0f;
    for (int i = 0; i < nq; ++i) {
        max_stats_error = std::max(max_stats_error, std::abs(amx_m[i] - avx_m[i]));
        max_stats_error = std::max(max_stats_error, std::abs(amx_s[i] - avx_s[i]));
    }
    if (max_abs_error > 0.08f || relative_rms > 0.04 || max_stats_error > 1.0e-3f) {
        std::fprintf(stderr,
            "AMX flash attention mismatch for nq=%d, nk=%d: max_abs=%g, rel_rms=%g, stats=%g\n",
            nq, nk, max_abs_error, relative_rms, max_stats_error);
        return false;
    }

    std::printf("PASS AMX FA nq=%d, nk=%d, max_abs=%g, rel_rms=%g, stats=%g\n",
        nq, nk, max_abs_error, relative_rms, max_stats_error);
    return true;
}

bool run_fa_repeated_mask_case(int n_tokens, int nk) {
    constexpr int d = 256;
    constexpr int gqa = 6;
    const int nq = n_tokens*gqa;

    std::vector<float> q(static_cast<size_t>(nq)*d);
    std::vector<ggml_bf16_t> k(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> v(static_cast<size_t>(nk)*d);
    std::vector<ggml_fp16_t> compact_mask(static_cast<size_t>(n_tokens)*nk);
    std::vector<ggml_fp16_t> expanded_mask(static_cast<size_t>(nq)*nk);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.25f*std::sin(0.011f*static_cast<float>(i));
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = ggml_fp32_to_bf16(0.25f*std::cos(0.005f*static_cast<float>(i)));
        v[i] = ggml_fp32_to_bf16(std::sin(0.017f*static_cast<float>(i)));
    }
    for (int token = 0; token < n_tokens; ++token) {
        const int visible = nk - 3*token;
        for (int ik = 0; ik < nk; ++ik) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(ik < visible ? 0.0f : -INFINITY);
            compact_mask[static_cast<size_t>(token)*nk + ik] = value;
            for (int head = 0; head < gqa; ++head) {
                expanded_mask[static_cast<size_t>(token*gqa + head)*nk + ik] = value;
            }
        }
    }

    std::vector<float> expanded(static_cast<size_t>(nq)*d);
    std::vector<float> repeated(static_cast<size_t>(nq)*d);
    std::vector<float> expanded_m(nq), expanded_s(nq), repeated_m(nq), repeated_s(nq);
    const int mask_stride = nk*sizeof(ggml_fp16_t);
    const bool expanded_handled = iqk_fa_256_256(
        GGML_TYPE_BF16, GGML_TYPE_BF16, nq, nk,
        d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
        mask_stride, d, q.data(), k.data(), v.data(), expanded_mask.data(),
        1.0f/std::sqrt(static_cast<float>(d)), 0.0f, expanded.data(), nullptr, 0,
        expanded_m.data(), expanded_s.data());
    const bool repeated_handled = iqk_fa_256_256(
        GGML_TYPE_BF16, GGML_TYPE_BF16, nq, nk,
        d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
        -mask_stride, d, q.data(), k.data(), v.data(), compact_mask.data(),
        1.0f/std::sqrt(static_cast<float>(d)), 0.0f, repeated.data(), nullptr, 0,
        repeated_m.data(), repeated_s.data());
    if (!expanded_handled || !repeated_handled) {
        std::fprintf(stderr, "flash attention rejected repeated-mask case tokens=%d, nk=%d\n", n_tokens, nk);
        return false;
    }

    float max_error = 0.0f;
    for (size_t i = 0; i < expanded.size(); ++i) {
        max_error = std::max(max_error, std::abs(expanded[i] - repeated[i]));
    }
    for (int i = 0; i < nq; ++i) {
        max_error = std::max(max_error, std::abs(expanded_m[i] - repeated_m[i]));
        max_error = std::max(max_error, std::abs(expanded_s[i] - repeated_s[i]));
    }
    if (max_error > 1.0e-6f) {
        std::fprintf(stderr,
            "repeated-mask flash attention mismatch for tokens=%d, nk=%d: max_error=%g\n",
            n_tokens, nk, max_error);
        return false;
    }

    std::printf("PASS AMX FA repeated masks tokens=%d, nq=%d, nk=%d, max_error=%g\n",
        n_tokens, nq, nk, max_error);
    return true;
}

int run_benchmark() {
    constexpr matrix_case tc = { 512, 128, 4096, 1 };
    std::mt19937 rng(0x9470cu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ggml_bf16_t> a(static_cast<size_t>(tc.m)*tc.k);
    std::vector<ggml_bf16_t> b(static_cast<size_t>(tc.n)*tc.k);
    std::vector<float> result(static_cast<size_t>(tc.m)*tc.n);
    for (auto & value : a) value = ggml_fp32_to_bf16(dist(rng));
    for (auto & value : b) value = ggml_fp32_to_bf16(dist(rng));

    auto invoke = [&] {
        return iqk_mul_mat(tc.m, tc.n, tc.k,
            GGML_TYPE_BF16, a.data(), tc.k*sizeof(ggml_bf16_t),
            GGML_TYPE_BF16, b.data(), tc.k*sizeof(ggml_bf16_t),
            result.data(), tc.m, 0, 1);
    };
    if (!invoke()) return 1;

    constexpr int iterations = 12;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!invoke()) return 1;
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double operations = 2.0*tc.m*tc.n*tc.k*iterations;
    std::printf("BF16 benchmark %dx%dx%d: %.3f ms/iteration, %.2f GFLOP/s%s\n",
        tc.m, tc.n, tc.k, 1e3*seconds/iterations, operations/seconds/1e9,
        std::getenv("GGML_AMX_DISABLE") ? " (AMX disabled)" : " (AMX requested)");
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        return run_benchmark();
    }

    const matrix_case cases[] = {
        { 64, 16, 1024, 1 }, // full AMX tiles
        { 37, 13,  512, 3 }, // partial tiles and per-thread XTILEDATA permission
        { 15,  8,  256, 1 }, // small-M AVX512 fallback
        { 33,  9,  128, 2 }, // small-K AVX512 fallback
    };
    for (const matrix_case & tc : cases) {
        if (!run_case(tc)) return 1;
    }
    if (!run_fa_case( 6, 64)) return 1; // q_step=8,  k_step=64
    if (!run_fa_case(12, 64)) return 1; // q_step=16, k_step=64
    if (!run_fa_case(30, 96)) return 1; // q_step=32, k_step=32
    if (!run_fa_repeated_mask_case(5, 96)) return 1;
    return 0;
}
