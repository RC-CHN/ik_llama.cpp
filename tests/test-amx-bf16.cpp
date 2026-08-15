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
    return 0;
}
