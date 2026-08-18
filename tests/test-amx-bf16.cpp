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

bool run_fa_sparse_page_case(int n_tokens, int nk) {
    constexpr int d = 256;
    constexpr int gqa = 6;
    // BF16 D=256 selects a 64-token K tile when nk is divisible by 64.
    // Alternate whole tiles so the sparse call exercises the page-pruning
    // branch rather than merely masking lanes inside a computed tile.
    constexpr int page = 64;
    const int nq = n_tokens*gqa;
    if (nk % (2*page) != 0) {
        return false;
    }
    const int packed_nk = nk/2;

    std::vector<float> q(static_cast<size_t>(nq)*d);
    std::vector<ggml_bf16_t> k(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> v(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> packed_k(static_cast<size_t>(packed_nk)*d);
    std::vector<ggml_bf16_t> packed_v(static_cast<size_t>(packed_nk)*d);
    std::vector<ggml_fp16_t> sparse_mask(static_cast<size_t>(n_tokens)*nk);
    std::vector<ggml_fp16_t> packed_mask(static_cast<size_t>(n_tokens)*packed_nk,
        ggml_fp32_to_fp16(0.0f));

    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.25f*std::sin(0.019f*static_cast<float>(i));
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = ggml_fp32_to_bf16(0.25f*std::cos(0.003f*static_cast<float>(i)));
        v[i] = ggml_fp32_to_bf16(std::sin(0.007f*static_cast<float>(i)));
    }

    int packed_row = 0;
    for (int block = 0; block < nk/page; ++block) {
        const bool visible = block % 2 == 0;
        for (int offset = 0; offset < page; ++offset) {
            const int row = block*page + offset;
            for (int token = 0; token < n_tokens; ++token) {
                sparse_mask[static_cast<size_t>(token)*nk + row] =
                    ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
            if (visible) {
                std::memcpy(packed_k.data() + static_cast<size_t>(packed_row)*d,
                    k.data() + static_cast<size_t>(row)*d, d*sizeof(ggml_bf16_t));
                std::memcpy(packed_v.data() + static_cast<size_t>(packed_row)*d,
                    v.data() + static_cast<size_t>(row)*d, d*sizeof(ggml_bf16_t));
                ++packed_row;
            }
        }
    }
    if (packed_row != packed_nk) {
        return false;
    }

    std::vector<float> sparse(static_cast<size_t>(nq)*d);
    std::vector<float> packed(static_cast<size_t>(nq)*d);
    std::vector<float> sparse_m(nq), sparse_s(nq), packed_m(nq), packed_s(nq);
    const int sparse_stride = nk*sizeof(ggml_fp16_t);
    const int packed_stride = packed_nk*sizeof(ggml_fp16_t);
    const bool sparse_handled = iqk_fa_256_256(
        GGML_TYPE_BF16, GGML_TYPE_BF16, nq, nk,
        d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
        -sparse_stride, d, q.data(), k.data(), v.data(), sparse_mask.data(),
        1.0f/std::sqrt(static_cast<float>(d)), 0.0f, sparse.data(), nullptr, 0,
        sparse_m.data(), sparse_s.data());
    const bool packed_handled = iqk_fa_256_256(
        GGML_TYPE_BF16, GGML_TYPE_BF16, nq, packed_nk,
        d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
        -packed_stride, d, q.data(), packed_k.data(), packed_v.data(), packed_mask.data(),
        1.0f/std::sqrt(static_cast<float>(d)), 0.0f, packed.data(), nullptr, 0,
        packed_m.data(), packed_s.data());
    if (!sparse_handled || !packed_handled) {
        std::fprintf(stderr, "flash attention rejected sparse-page case tokens=%d, nk=%d\n",
            n_tokens, nk);
        return false;
    }

    double squared_error = 0.0;
    double squared_reference = 0.0;
    float max_output_error = 0.0f;
    for (size_t i = 0; i < sparse.size(); ++i) {
        const float error = sparse[i] - packed[i];
        max_output_error = std::max(max_output_error, std::abs(error));
        squared_error += static_cast<double>(error)*error;
        squared_reference += static_cast<double>(packed[i])*packed[i];
    }
    const double relative_rms = std::sqrt(
        squared_error/std::max(1.0e-30, squared_reference));

    float max_stats_error = 0.0f;
    for (int i = 0; i < nq; ++i) {
        max_stats_error = std::max(max_stats_error, std::abs(sparse_m[i] - packed_m[i]));
        max_stats_error = std::max(max_stats_error, std::abs(sparse_s[i] - packed_s[i]));
    }
    if (max_output_error > 0.08f || relative_rms > 0.04 || max_stats_error > 1.0e-3f) {
        std::fprintf(stderr,
            "sparse-page flash attention mismatch for tokens=%d, nk=%d: "
            "max_output=%g, rel_rms=%g, stats=%g\n",
            n_tokens, nk, max_output_error, relative_rms, max_stats_error);
        return false;
    }

    std::printf("PASS AMX FA sparse pages tokens=%d, nq=%d, nk=%d, "
        "max_output=%g, rel_rms=%g, stats=%g\n",
        n_tokens, nq, nk, max_output_error, relative_rms, max_stats_error);
    return true;
}

bool run_fa_work_buffer_case() {
    ggml_tensor q   = {};
    ggml_tensor k   = {};
    ggml_tensor v   = {};
    ggml_tensor dst = {};

    q.type = GGML_TYPE_F32;
    k.type = GGML_TYPE_BF16;
    v.type = GGML_TYPE_BF16;
    q.ne[0] = 256;
    q.ne[1] = 10;
    q.ne[2] = 24;
    q.ne[3] = 1;
    k.ne[0] = v.ne[0] = 256;
    k.ne[1] = v.ne[1] = 128;
    k.ne[2] = v.ne[2] = 4;
    k.ne[3] = v.ne[3] = 1;
    dst.src[0] = &q;
    dst.src[1] = &k;
    dst.src[2] = &v;
    dst.op_params[5] = 1;   // return online-softmax statistics
    dst.op_params[6] = 4;   // NUMA shards
    dst.op_params[7] = 128; // cold-cache capacity
    dst.op_params[8] = 1;   // return one partial per NUMA shard

    constexpr size_t group_plan_bytes = 128;
    constexpr size_t partial_ptr_bytes = 52*sizeof(float *);
    constexpr size_t row_bytes = 24*256*sizeof(float);
    constexpr size_t narrow_bytes = group_plan_bytes + 5*row_bytes + partial_ptr_bytes;
    constexpr size_t wide_bytes   = group_plan_bytes + 10*row_bytes + partial_ptr_bytes;
    const size_t actual = iqk_fa_work_buffer_size(&dst, 52);
    if (actual != narrow_bytes && actual != wide_bytes) {
        std::fprintf(stderr,
            "NUMA FA work-buffer planner mismatch: expected %zu or %zu bytes, got %zu\n",
            narrow_bytes, wide_bytes, actual);
        return false;
    }
    std::printf("PASS NUMA FA work-buffer planner: %zu bytes (%s query groups)\n",
        actual, actual == wide_bytes ? "wide" : "narrow");
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

int run_fa_page_benchmark() {
    constexpr int d = 256;
    constexpr int n_tokens = 3;
    constexpr int gqa = 6;
    constexpr int nq = n_tokens*gqa;
    constexpr int nk = 32768;
    constexpr int page = 64;
    constexpr int packed_nk = nk/2;
    constexpr int iterations = 5;

    std::vector<float> q(static_cast<size_t>(nq)*d);
    std::vector<ggml_bf16_t> k(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> v(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> packed_k(static_cast<size_t>(packed_nk)*d);
    std::vector<ggml_bf16_t> packed_v(static_cast<size_t>(packed_nk)*d);
    std::vector<ggml_fp16_t> dense_mask(static_cast<size_t>(n_tokens)*nk,
        ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> sparse_mask(static_cast<size_t>(n_tokens)*nk);
    std::vector<ggml_fp16_t> packed_mask(static_cast<size_t>(n_tokens)*packed_nk,
        ggml_fp32_to_fp16(0.0f));
    std::vector<float> output(static_cast<size_t>(nq)*d);
    std::vector<float> max(nq), sum(nq);

    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.25f*std::sin(0.019f*static_cast<float>(i));
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = ggml_fp32_to_bf16(0.25f*std::cos(0.003f*static_cast<float>(i)));
        v[i] = ggml_fp32_to_bf16(std::sin(0.007f*static_cast<float>(i)));
    }

    int packed_row = 0;
    for (int block = 0; block < nk/page; ++block) {
        const bool visible = block % 2 == 0;
        for (int offset = 0; offset < page; ++offset) {
            const int row = block*page + offset;
            for (int token = 0; token < n_tokens; ++token) {
                sparse_mask[static_cast<size_t>(token)*nk + row] =
                    ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
            if (visible) {
                std::memcpy(packed_k.data() + static_cast<size_t>(packed_row)*d,
                    k.data() + static_cast<size_t>(row)*d, d*sizeof(ggml_bf16_t));
                std::memcpy(packed_v.data() + static_cast<size_t>(packed_row)*d,
                    v.data() + static_cast<size_t>(row)*d, d*sizeof(ggml_bf16_t));
                ++packed_row;
            }
        }
    }

    auto invoke = [&](const std::vector<ggml_bf16_t> & this_k,
                      const std::vector<ggml_bf16_t> & this_v,
                      const std::vector<ggml_fp16_t> & this_mask,
                      int this_nk) {
        return iqk_fa_256_256(
            GGML_TYPE_BF16, GGML_TYPE_BF16, nq, this_nk,
            d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
            -this_nk*(int) sizeof(ggml_fp16_t), d,
            q.data(), this_k.data(), this_v.data(), this_mask.data(),
            1.0f/std::sqrt(static_cast<float>(d)), 0.0f,
            output.data(), nullptr, 0, max.data(), sum.data());
    };
    if (!invoke(k, v, dense_mask, nk) ||
            !invoke(k, v, sparse_mask, nk) ||
            !invoke(packed_k, packed_v, packed_mask, packed_nk)) {
        return 1;
    }

    auto measure = [&](const std::vector<ggml_bf16_t> & this_k,
                       const std::vector<ggml_bf16_t> & this_v,
                       const std::vector<ggml_fp16_t> & this_mask,
                       int this_nk) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            if (!invoke(this_k, this_v, this_mask, this_nk)) {
                return -1.0;
            }
        }
        const auto end = std::chrono::steady_clock::now();
        return 1e3*std::chrono::duration<double>(end - begin).count()/iterations;
    };

    const double dense_ms = measure(k, v, dense_mask, nk);
    const double sparse_ms = measure(k, v, sparse_mask, nk);
    const double packed_ms = measure(packed_k, packed_v, packed_mask, packed_nk);
    if (dense_ms <= 0.0 || sparse_ms <= 0.0 || packed_ms <= 0.0) {
        return 1;
    }
    std::printf("BF16 FA page benchmark nq=%d nk=%d visible=50%%: "
        "dense=%.3f ms sparse=%.3f ms packed=%.3f ms "
        "dense/sparse=%.2fx sparse/packed=%.2fx%s\n",
        nq, nk, dense_ms, sparse_ms, packed_ms,
        dense_ms/sparse_ms, sparse_ms/packed_ms,
        std::getenv("GGML_AMX_DISABLE") ? " (AMX disabled)" : " (AMX requested)");
    return 0;
}

int run_fa_group_benchmark(int n_tokens, int nk, int iterations, bool disjoint_masks = false) {
    constexpr int d = 256;
    constexpr int gqa = 6;
    if (n_tokens < 2 || n_tokens > 10 || n_tokens % 2 != 0 ||
            nk < 32 || nk % 32 != 0 || iterations < 1) {
        std::fprintf(stderr,
            "query-group benchmark requires even tokens in [2,10], nk divisible by 32, and iterations > 0\n");
        return 2;
    }
    const int nq = n_tokens*gqa;

    std::vector<float> q(static_cast<size_t>(nq)*d);
    std::vector<ggml_bf16_t> k(static_cast<size_t>(nk)*d);
    std::vector<ggml_bf16_t> v(static_cast<size_t>(nk)*d);
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(n_tokens)*nk,
        ggml_fp32_to_fp16(0.0f));
    if (disjoint_masks) {
        // Model two independently growing slots sharing one physical KV
        // cache: each half of the query cohort can see a disjoint half of K.
        // This exercises per-page AMX query-tile pruning in the combined call.
        const int half_tokens = n_tokens/2;
        for (int token = 0; token < n_tokens; ++token) {
            for (int ik = 0; ik < nk; ++ik) {
                const bool visible = token < half_tokens ? ik < nk/2 : ik >= nk/2;
                mask[(size_t) token*nk + ik] =
                    ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
        }
    }
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.25f*std::sin(0.011f*static_cast<float>(i));
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = ggml_fp32_to_bf16(0.25f*std::cos(0.005f*static_cast<float>(i)));
        v[i] = ggml_fp32_to_bf16(std::sin(0.017f*static_cast<float>(i)));
    }

    std::vector<float> combined(static_cast<size_t>(nq)*d);
    std::vector<float> split(static_cast<size_t>(nq)*d);
    std::vector<float> combined_m(nq), combined_s(nq), split_m(nq), split_s(nq);
    const int mask_stride = nk*sizeof(ggml_fp16_t);

    auto invoke = [&](bool one_group) {
        if (one_group) {
            return iqk_fa_256_256(
                GGML_TYPE_BF16, GGML_TYPE_BF16, nq, nk,
                d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
                -mask_stride, d, q.data(), k.data(), v.data(), mask.data(),
                1.0f/std::sqrt(static_cast<float>(d)), 0.0f, combined.data(), nullptr, 0,
                combined_m.data(), combined_s.data());
        }

        const int half_tokens = n_tokens/2;
        const int half_nq = half_tokens*gqa;
        for (int half = 0; half < 2; ++half) {
            if (!iqk_fa_256_256(
                    GGML_TYPE_BF16, GGML_TYPE_BF16, half_nq, nk,
                    d*sizeof(float), d*sizeof(ggml_bf16_t), d*sizeof(ggml_bf16_t),
                    -mask_stride, d,
                    q.data() + (size_t) half*half_nq*d, k.data(), v.data(),
                    mask.data() + (size_t) half*half_tokens*nk,
                    1.0f/std::sqrt(static_cast<float>(d)), 0.0f,
                    split.data() + (size_t) half*half_nq*d, nullptr, 0,
                    split_m.data() + half*half_nq, split_s.data() + half*half_nq)) {
                return false;
            }
        }
        return true;
    };

    if (!invoke(true) || !invoke(false)) {
        return 1;
    }
    float max_output_error = 0.0f;
    float max_stats_relative_error = 0.0f;
    for (size_t i = 0; i < combined.size(); ++i) {
        max_output_error = std::max(max_output_error, std::abs(combined[i] - split[i]));
    }
    for (int i = 0; i < nq; ++i) {
        max_stats_relative_error = std::max(max_stats_relative_error,
            std::abs(combined_m[i] - split_m[i])/std::max(1.0f, std::abs(split_m[i])));
        max_stats_relative_error = std::max(max_stats_relative_error,
            std::abs(combined_s[i] - split_s[i])/std::max(1.0f, std::abs(split_s[i])));
    }
    const bool mismatch = max_output_error > 0.08f || max_stats_relative_error > 1.0e-3f;
    if (mismatch) {
        std::fprintf(stderr,
            "BF16 FA query-group mismatch: output=%g stats_relative=%g\n",
            max_output_error, max_stats_relative_error);
    }

    auto measure = [&](bool one_group) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            if (!invoke(one_group)) {
                return -1.0;
            }
        }
        const auto end = std::chrono::steady_clock::now();
        return 1e3*std::chrono::duration<double>(end - begin).count()/iterations;
    };
    const double combined_ms = measure(true);
    const double split_ms = measure(false);
    if (combined_ms <= 0.0 || split_ms <= 0.0) {
        return 1;
    }
    std::printf("BF16 FA query-group benchmark pattern=%s tokens=%d nq=%d nk=%d: "
        "combined=%.3f ms split=%.3f ms split/combined=%.2fx "
        "output_error=%g stats_relative=%g%s\n",
        disjoint_masks ? "disjoint" : "dense",
        n_tokens, nq, nk, combined_ms, split_ms, split_ms/combined_ms,
        max_output_error, max_stats_relative_error,
        std::getenv("GGML_AMX_DISABLE") ? " (AMX disabled)" : " (AMX requested)");
    return mismatch ? 1 : 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        return run_benchmark();
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench-fa-pages") == 0) {
        return run_fa_page_benchmark();
    }
    if (argc >= 2 && argc <= 5 &&
            (std::strcmp(argv[1], "--bench-fa-groups") == 0 ||
             std::strcmp(argv[1], "--bench-fa-disjoint") == 0)) {
        const int tokens = argc >= 3 ? std::atoi(argv[2]) : 8;
        const int nk = argc >= 4 ? std::atoi(argv[3]) : 4096;
        const int iterations = argc >= 5 ? std::atoi(argv[4]) : 100;
        return run_fa_group_benchmark(
            tokens, nk, iterations, std::strcmp(argv[1], "--bench-fa-disjoint") == 0);
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
    if (!run_fa_case(48, 64)) return 1; // q_step=64, k_step=64
    if (!run_fa_repeated_mask_case(5, 96)) return 1;
    if (!run_fa_repeated_mask_case(8, 128)) return 1;
    if (!run_fa_sparse_page_case(3, 256)) return 1;
    if (run_fa_group_benchmark(10, 256, 1, true) != 0) return 1;
    if (!run_fa_work_buffer_case()) return 1;
    return 0;
}
