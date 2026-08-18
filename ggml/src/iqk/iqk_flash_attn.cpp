//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_config.h"
#include "iqk_mul_mat.h"
#include "iqk_flash_impl.h"
#include "ggml.h"

#include <atomic>

namespace {
std::atomic<uint64_t> g_fa_pages_checked{0};
std::atomic<uint64_t> g_fa_pages_skipped{0};
std::atomic<uint64_t> g_fa_layout_plans{0};
std::atomic<uint64_t> g_fa_layout_runs{0};
std::atomic<uint64_t> g_fa_group_observations{0};
}

extern "C" IQK_API void iqk_fa_page_stats_add(
        uint64_t pages_checked, uint64_t pages_skipped) {
    g_fa_pages_checked.fetch_add(pages_checked, std::memory_order_relaxed);
    g_fa_pages_skipped.fetch_add(pages_skipped, std::memory_order_relaxed);
}

extern "C" IQK_API void iqk_fa_page_stats_get(
        uint64_t * pages_checked, uint64_t * pages_skipped) {
    if (pages_checked != nullptr) {
        *pages_checked = g_fa_pages_checked.load(std::memory_order_relaxed);
    }
    if (pages_skipped != nullptr) {
        *pages_skipped = g_fa_pages_skipped.load(std::memory_order_relaxed);
    }
}

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unordered_set>

#if defined(GGML_AMX_BF16) && defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__x86_64__)
#include <cpuid.h>
#endif

namespace {
inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}
inline void accumulate_qkv(int Dv, float& M, float& S, float Mj, float Sj, float * Racc, const float * R) {
    if (Mj == -INFINITY) return;
    if (Mj > M) {
        if (M == -INFINITY) {
            std::memcpy(Racc, R, Dv*sizeof(float));
            S = Sj;
        } else {
            float c = exp(M - Mj);
            S = c*S + Sj;
            for (int i = 0; i < Dv; ++i) Racc[i] = c*Racc[i] + R[i];
        }
        M = Mj;
    } else {
        float c = exp(Mj - M);
        S += c*Sj;
        for (int i = 0; i < Dv; ++i) Racc[i] += c*R[i];
    }
}
inline float pack_softmax_max(float M) {
    return std::isfinite(M) ? M : -32752.0f;
}

inline int32_t fa_param_i32(const ggml_tensor * tensor, int index) {
    return tensor->op_params[index];
}

inline bool numa_fa_eligible(const ggml_tensor * dst, int nth) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int n_shards = fa_param_i32(dst, 6);
    const int cold_capacity = fa_param_i32(dst, 7);
    if (fa_param_i32(dst, 5) == 0 || n_shards < 2 || nth % n_shards != 0 ||
            Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_BF16 || V->type != GGML_TYPE_BF16 ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
            Q->ne[0] != 256 || V->ne[0] != 256 || Q->ne[0] != K->ne[0] ||
            Q->ne[1] < 1 || Q->ne[1] > 16 || K->ne[1] != V->ne[1] ||
            K->ne[1] < 32 || K->ne[1] % 32 != 0 ||
            K->ne[2] < 1 || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0 ||
            cold_capacity < K->ne[1] || cold_capacity % (32*n_shards) != 0) {
        return false;
    }
    const int gqa = Q->ne[2]/K->ne[2];
    return gqa > 1 && gqa <= 256;
}

inline int numa_fa_chunks(const ggml_tensor * dst, int nth) {
    const int n_shards = fa_param_i32(dst, 6);
    const int n_pairs = dst->src[1]->ne[2]*dst->src[0]->ne[1];
    const int local_nth = nth/n_shards;
    return std::max(1, (local_nth + n_pairs - 1)/n_pairs);
}

inline bool numa_fa_batch_queries_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_NUMA_FA_BATCH_QUERIES");
        return value == nullptr || value[0] == '\0' ||
            (std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
             std::strcmp(value, "off") != 0);
    }();
    return enabled;
}

inline bool numa_fa_batch_queries_eligible(const ggml_tensor * dst, int nth) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int n_queries = dst->src[0]->ne[1];
    const int gqa = Q->ne[2]/K->ne[2];
    return numa_fa_eligible(dst, nth) && numa_fa_batch_queries_enabled() &&
        n_queries >= 1 && n_queries <= 16 && Q->ne[2] == 24 &&
        K->ne[2] == 4 && gqa == 6;
}

inline size_t numa_fa_align64(size_t size) {
    return (size + 63) & ~(size_t) 63;
}

// The AVX-512 kernel consumes two 16-row query tiles at once.  The AMX
// implementation can retain four such tiles across two register waves. Derive
// the token cohort from that live ISA capacity and the model's actual GQA
// ratio, instead of assigning a context- or concurrency-specific depth.
constexpr int numa_fa_query_tile_rows = 16;
constexpr int numa_fa_avx_query_tiles = 2;
constexpr int numa_fa_amx_query_tiles = 4;
constexpr int numa_fa_max_query_rows = numa_fa_query_tile_rows*numa_fa_amx_query_tiles;
constexpr int numa_fa_max_query_group_tokens = numa_fa_max_query_rows/6;

inline bool numa_fa_amx_available() {
#if defined(GGML_AMX_BF16) && defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__x86_64__)
    static const bool available = [] {
        const char * disabled = std::getenv("GGML_AMX_DISABLE");
        const char * fa_qk = std::getenv("GGML_AMX_FA_QK");
        if ((disabled && disabled[0] != '\0' && disabled[0] != '0') ||
                (fa_qk && (fa_qk[0] == '0' || std::strcmp(fa_qk, "off") == 0 ||
                           std::strcmp(fa_qk, "false") == 0))) {
            return false;
        }
        unsigned int eax, ebx, ecx, edx;
        constexpr unsigned int cpuid_amx_bf16 = 1u << 22;
        constexpr unsigned int cpuid_amx_tile = 1u << 24;
        return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) &&
            (edx & (cpuid_amx_bf16 | cpuid_amx_tile)) == (cpuid_amx_bf16 | cpuid_amx_tile);
    }();
    return available;
#else
    return false;
#endif
}

enum class numa_fa_group_policy {
    adaptive,
    narrow,
    wide,
};

inline numa_fa_group_policy numa_fa_query_group_policy() {
    static const numa_fa_group_policy policy = [] {
        const char * value = std::getenv("GGML_NUMA_FA_WIDE_QUERY_GROUPS");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0 ||
                std::strcmp(value, "adaptive") == 0) {
            return numa_fa_group_policy::adaptive;
        }
        if (value[0] == '0' || std::strcmp(value, "false") == 0 ||
                std::strcmp(value, "off") == 0) {
            return numa_fa_group_policy::narrow;
        }
        return numa_fa_group_policy::wide;
    }();
    return policy;
}

inline int numa_fa_narrow_query_group_tokens(int gqa) {
    return std::max(1, numa_fa_query_tile_rows*numa_fa_avx_query_tiles/std::max(1, gqa));
}

inline int numa_fa_wide_query_group_tokens(int gqa) {
    return std::max(1, numa_fa_query_tile_rows*numa_fa_amx_query_tiles/std::max(1, gqa));
}

inline int numa_fa_max_query_group_tokens_for_layout(int gqa) {
    const int query_rows = numa_fa_query_tile_rows*
        (numa_fa_amx_available() && numa_fa_query_group_policy() != numa_fa_group_policy::narrow
            ? numa_fa_amx_query_tiles : numa_fa_avx_query_tiles);
    return std::max(1, query_rows/std::max(1, gqa));
}

constexpr int numa_fa_feedback_token_buckets = 17;
constexpr int numa_fa_feedback_work_buckets = 32;

struct numa_fa_group_feedback {
    std::atomic<uint64_t> samples[2] = {};
    std::atomic<uint64_t> total_us[2] = {};
    std::atomic<uint64_t> decisions{0};
};

numa_fa_group_feedback g_numa_fa_group_feedback
    [numa_fa_feedback_token_buckets][numa_fa_feedback_work_buckets];

inline int numa_fa_work_bucket(int rows) {
    int bucket = 0;
    for (int value = std::max(1, rows); value > 1; value = (value + 1)/2) {
        ++bucket;
    }
    return std::min(bucket, numa_fa_feedback_work_buckets - 1);
}

inline int numa_fa_max_rows_per_worker(
        int n_kv, int cold_capacity, int n_shards, int local_nth) {
    int result = 0;
    for (int node = 0; node < n_shards; ++node) {
        const int first = std::min(n_kv, (int) ((int64_t) cold_capacity*node/n_shards));
        const int last = std::min(n_kv, (int) ((int64_t) cold_capacity*(node + 1)/n_shards));
        const int blocks = std::max(0, last - first)/32;
        result = std::max(result, 32*((blocks + local_nth - 1)/local_nth));
    }
    return result;
}

inline int numa_fa_choose_group_arm(numa_fa_group_feedback & feedback) {
    const uint64_t narrow_samples = feedback.samples[0].load(std::memory_order_relaxed);
    const uint64_t wide_samples = feedback.samples[1].load(std::memory_order_relaxed);
    const uint64_t decision = feedback.decisions.fetch_add(1, std::memory_order_relaxed) + 1;

    // Calibrate both kernels in the live topology before exploiting. Continue
    // to challenge the current winner sparsely so thermal or co-tenant changes
    // can move the crossover without a restart.
    if (narrow_samples < 8 || wide_samples < 8) {
        return narrow_samples <= wide_samples ? 0 : 1;
    }
    const uint64_t narrow_total = feedback.total_us[0].load(std::memory_order_relaxed);
    const uint64_t wide_total = feedback.total_us[1].load(std::memory_order_relaxed);
    const int winner = wide_total*narrow_samples < narrow_total*wide_samples ? 1 : 0;
    return decision % 64 == 0 ? 1 - winner : winner;
}

inline void numa_fa_observe_group_arm(
        numa_fa_group_feedback & feedback, int arm, uint64_t elapsed_us) {
    feedback.total_us[arm].fetch_add(elapsed_us, std::memory_order_relaxed);
    feedback.samples[arm].fetch_add(1, std::memory_order_relaxed);
}

inline size_t numa_fa_group_plan_size(int n_tokens) {
    return numa_fa_align64((1 + 2*(size_t) n_tokens)*sizeof(int32_t));
}
}

size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nth) {
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    const bool numa_eligible = numa_fa_eligible(dst, nth);
    const bool batch_eligible = numa_eligible && numa_fa_batch_queries_eligible(dst, nth);
    if (std::strncmp(dst->name, "hybrid_kv_cold_stats", 20) == 0) {
        const uint64_t index = g_fa_layout_plans.fetch_add(1, std::memory_order_relaxed) + 1;
        if (index <= 8) {
            const int n_shards = fa_param_i32(dst, 6);
            const int gqa = K->ne[2] > 0 ? Q->ne[2]/K->ne[2] : 0;
            std::fprintf(stderr,
                    "iqk_fa: layout plan=%llu nth=%d q=[%lld,%lld,%lld,%lld] "
                    "k=[%lld,%lld,%lld,%lld] stats=%d shards=%d cold=%d "
                    "numa=%d batch=%d amx=%d group=%d\n",
                    (unsigned long long) index, nth,
                    (long long) Q->ne[0], (long long) Q->ne[1],
                    (long long) Q->ne[2], (long long) Q->ne[3],
                    (long long) K->ne[0], (long long) K->ne[1],
                    (long long) K->ne[2], (long long) K->ne[3],
                    fa_param_i32(dst, 5), n_shards, fa_param_i32(dst, 7),
                    numa_eligible, batch_eligible, numa_fa_amx_available(),
                    gqa > 0 ? numa_fa_max_query_group_tokens_for_layout(gqa) : 0);
        }
    }
    if (numa_eligible) {
        const int n_shards = fa_param_i32(dst, 6);
        if (batch_eligible) {
            const int local_nth = nth/n_shards;
            const size_t group_plan_size = numa_fa_group_plan_size(Q->ne[1]);
            const int gqa = Q->ne[2]/K->ne[2];
            const int group_tokens = numa_fa_max_query_group_tokens_for_layout(gqa);
            const size_t packed_q_size = numa_fa_align64(
                    (size_t) std::min<int64_t>(Q->ne[1], group_tokens)*
                    Q->ne[2]*Q->ne[0]*sizeof(float));
            return group_plan_size + packed_q_size +
                (size_t) n_shards*local_nth*sizeof(float *);
        }
        const int n_pairs = K->ne[2]*Q->ne[1];
        const int n_chunks = numa_fa_chunks(dst, nth);
        const int gqa = Q->ne[2]/K->ne[2];
        const size_t result_size = (V->ne[0] + 16)*gqa*sizeof(float);
        return (size_t) n_shards*n_pairs*n_chunks*result_size;
    }
    auto indexer = dst->src[5];
    if (indexer && indexer->type == GGML_TYPE_I32 && indexer->ne[0] < K->ne[1] &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 && K->ne[2] == 1) {
        auto row_size_k = ggml_row_size(K->type, K->ne[0]);
        auto row_size_v = ggml_row_size(V->type, V->ne[0]);
        auto work_size  = (row_size_k + row_size_v + 64) * indexer->ne[0];
        size_t result = work_size * nth;
        if (Q->ne[1]== 1) result += 512*sizeof(float);
        return result;
        //return work_size * nth;
    }
    int rk2 = Q->ne[2]/K->ne[2];
    size_t size = 0;
    if (Q->ne[1] >= 8 && K->type == GGML_TYPE_Q8_0) {
        size = ggml_row_size(GGML_TYPE_Q8_0, K->ne[0]) * K->ne[1]*K->ne[2]*K->ne[3];
    }
    if (Q->ne[1] == 1 && Q->ne[3] == 1 && Q->ne[2]/K->ne[2] > 1 && nth >= 1 && K->ne[1]/32 > 1) {
        if (K->ne[2] > 1) {
            int gcd = simple_gcd(K->ne[2], nth);
            int nth_k  = nth/gcd;
            int nek2_k = K->ne[2]/gcd;
            int nchunk = nek2_k*K->ne[1]/32;
            int npt = (nchunk + nth_k - 1)/nth_k;
            int nk;
            if (npt*nth_k == nchunk) {
                nk = 32 * (K->ne[1]*K->ne[2]/(32*nth));
            } else {
                //int nm = std::max(1, npt/8);
                int nm = 1;
                while (true) {
                    if (nm*4 >= npt) break;
                    nm *= 2;
                }
                nk = 32*nm;
            }
            int nkk = (K->ne[1] + nk - 1)/nk;
            int nstep_k = K->ne[2]*nkk;
            size_t result_size = (V->ne[0] + 16)*Q->ne[2]/K->ne[2]*sizeof(float);
            size += nstep_k*result_size;
            return size;
        }
        int nstep_k = K->ne[1]/32;
        if (nstep_k >= 4*nth) {
            auto size_thread = (V->ne[0] + 16)*rk2*sizeof(float);
            size += size_thread*nth;
            return size;
        }
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                auto size_thread = (V->ne[0] + 16)*nq_per_thread*sizeof(float);
                size += size_thread*nth;
                return size;
            }
        }
        int rv2 = Q->ne[2] / V->ne[2];
        if (Q->ne[1] == 1 && Q->ne[3] == 1 && rk2 > 1 && rk2 == rv2 && K->ne[1]*K->ne[2] >= 32*nth) {
            auto result_size = (V->ne[0] + 16)*rk2*sizeof(float);
            size += result_size*nth;
        }
        return size;
    }
    return size;
}

static inline const std::unordered_set<ggml_type> & supported_kv_types() {
#ifdef GGML_IQK_FA_ALL_QUANTS
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL
    };
#else
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0,
    };
#endif
    return k_supported;
}

static inline bool are_kv_types_supported(ggml_type type_k, ggml_type type_v) {
    if (type_k == GGML_TYPE_BF16) {
        if (type_v != type_k) {
            return false;
        }
#ifdef __AVX512BF16__
        return true;
#else
        return false;
#endif
    }
    auto & supported = supported_kv_types();
    auto it_k = supported.find(type_k);
    auto it_v = supported.find(type_v);
    return it_k != supported.end() && it_v != supported.end();
}

// TODO: get the ggml_type enum here without polution
//
extern "C" IQK_API bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int int_type_k_in,      // type of k
                            int int_type_v,         // type of v
                            int Dk,                 // K head size
                            int Dv,                 // V head size
                            int neq1,               // number of columns in q
                            int nek1,               // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            const void  * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            const void  * sinks,    // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            bool          return_stats,
                            int           numa_shards,
                            int           numa_cold_capacity,
                            bool          numa_return_partials,
                            float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer_in, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            int ith, int nth, int n_swa, [[maybe_unused]] ggml_tensor * indexer) {

    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;
    if (return_stats && (indexer || sinks || n_swa > 0)) return false;

    if (indexer && indexer->type == GGML_TYPE_I32) {
        //if (indexer->ne[0] < nek1 && neq1 >= nth && neq3 == 1 && nek3 == 1 && nev3 == 1 && nek2 == 1) {
        if (indexer->ne[0] < nek1 && neq3 == 1 && nek3 == 1 && nev3 == 1 && nek2 == 1) {
            // Workbuffer: we need
            // * indexer->ne[0] * sizeof(ggml_half) to extract the mask for a row
            // * indexer->ne[0] * ggml_row_size(int_type_k_in, Dk) to extract the selected K cache entries
            // * indexer->ne[0] * ggml_row_size(int_type_v, Dv) to extract the selected V cache entries
            auto row_size_k = ggml_row_size(ggml_type(int_type_k_in), Dk);
            auto row_size_v = ggml_row_size(ggml_type(int_type_v   ), Dv);
            auto work_size  = (row_size_k + row_size_v + 64) * indexer->ne[0];
            ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
            int  nkv = indexer->ne[0];
            if (neq1 == 1) {
                GGML_ASSERT(neq2 <= 256);
                int npt = (neq2 + nth - 1)/nth;
                int ith_mid = nth;
                int neq2_this_thread = npt;
                int first = ith*npt;
                if (npt*nth > neq2) {
                    ith_mid = neq2 - nth*(npt - 1);
                    if (ith >= ith_mid) {
                        --neq2_this_thread;
                        //if (neq2_this_thread < 1) return true;
                        first = ith_mid*npt + (ith - ith_mid)*neq2_this_thread;
                    }
                }
                auto idx = (const int *)indexer->data;
                auto M = (const ggml_fp16_t *)mask;
                auto work_k = (char *)work_buffer_in;
                auto work_v = work_k + row_size_k*indexer->ne[0];
                auto work_m = (ggml_fp16_t *)(work_v + row_size_v*indexer->ne[0]) + indexer->ne[0]*ith;
                int last_found = -1;
                for (int j = 0; j < nkv; ++j) {
                    if (idx[j] >= 0) {
                        last_found = j;
                        work_m[j] = M[idx[j]];
                        if (j % nth == ith) {
                            std::memcpy(work_k + row_size_k*j, ((const char *)k + idx[j]*stride_k), row_size_k);
                            if (k != v) {
                                std::memcpy(work_v + row_size_v*j, ((const char *)v + idx[j]*stride_v), row_size_v);
                            }
                        }
                    } else {
                        work_m[j] = h_inf;
                        if (j % nth == ith) {
                            std::memset(work_k + row_size_k*j, 0, row_size_k);
                            if (k != v) {
                                std::memset(work_v + row_size_v*j, 0, row_size_v);
                            }
                        }
                    }
                }
                barrier(barrier_data);
                if (last_found < 0 || neq2_this_thread < 1) return true;
                ++last_found;
                int this_nkv = 32*((last_found + 31)/32);
                auto this_q = (const char *)q + first*nbq2;
                auto this_qkv = qkv + first*nb1/sizeof(float);
                if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                         Dk, Dv, neq2_this_thread, this_nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                         (const float *)this_q, work_k, k == v ? work_k : work_v, work_m, (const float *)sinks, 1,
                         scale, softcap,
                         this_qkv, nullptr, nullptr)) return false;
                return true;
            }
            int npt = (neq1 + nth - 1)/nth;
            int ith_mid = nth;
            int neq1_this_thread = npt;
            int first = ith*npt;
            if (npt*nth > neq1) {
                ith_mid = neq1 - nth*(npt - 1);
                if (ith >= ith_mid) {
                    --neq1_this_thread;
                    if (neq1_this_thread < 1) return true;
                    first = ith_mid*npt + (ith - ith_mid)*neq1_this_thread;
                }
            }
            auto work_k = (char *)work_buffer_in + ith*work_size;
            auto work_v = work_k + row_size_k*indexer->ne[0];
            auto work_m = (ggml_fp16_t *)(work_v + row_size_v*indexer->ne[0]);
            for (int iq = first; iq < first + neq1_this_thread; ++iq) {
                auto idx = (const int *)((const char *)indexer->data + iq*indexer->nb[1]);
                auto M = (const ggml_fp16_t *)((const char *)mask + iq*stride_m);
                int last_found = -1;
                for (int j = 0; j < nkv; ++j) {
                    if (idx[j] >= 0) {
                        std::memcpy(work_k + row_size_k*j, ((const char *)k + idx[j]*stride_k), row_size_k);
                        if (k != v) {
                            std::memcpy(work_v + row_size_v*j, ((const char *)v + idx[j]*stride_v), row_size_v);
                        }
                        work_m[j] = M[idx[j]];
                        last_found = j;
                    } else {
                        std::memset(work_k + row_size_k*j, 0, row_size_k);
                        if (k != v) {
                            std::memset(work_v + row_size_v*j, 0, row_size_v);
                        }
                        work_m[j] = h_inf;
                    }
                }
                if (last_found < 0) continue;
                ++last_found;
                int this_nkv = 32*((last_found + 31)/32);
                auto this_q = (const char *)q + iq*stride_q;
                auto this_qkv = qkv + iq*ne1*nb1/sizeof(float);
                if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                         Dk, Dv, neq2, this_nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                         (const float *)this_q, work_k, k == v ? work_k : work_v, work_m, (const float *)sinks, 1,
                         scale, softcap,
                         this_qkv, nullptr, nullptr)) return false;
            }
            return true;
        }
    }

    if (auto type_k = ggml_type(int_type_k_in), type_v = ggml_type(int_type_v); !are_kv_types_supported(type_k, type_v)) {
        if (ith == 0) {
            fprintf(stderr, "\n==================== K cache %s coupled with V cache %s is not a supported combination on the CPU backend.\n",
                    ggml_type_name(type_k), ggml_type_name(type_v));
            auto & supported = supported_kv_types();
            fprintf(stderr, "Supported types are:\n");
            for (auto type : supported) {
                fprintf(stderr, "    %s\n", ggml_type_name(type));
            }
            fprintf(stderr, "    Warning: ik_llama.cpp does not support Q5_0 or Q5_1 KV cache on the CPU.\n");
#ifdef __AVX512BF16__
            fprintf(stderr, "    %s, but only if K and V are both %s\n", ggml_type_name(GGML_TYPE_BF16), ggml_type_name(GGML_TYPE_BF16));
#endif
#ifndef GGML_IQK_FA_ALL_QUANTS
            fprintf(stderr, "    To enable q4_0, q4_1, and iq4_nl KV cache types, recompile with -DGGML_IQK_FA_ALL_QUANTS=ON\n");
#endif
        }
        barrier(barrier_data);
        GGML_ABORT("Fatal error");
    }

    if (n_swa > 0 && mask) {
        constexpr int kMinBatch = 256;
        int ntokens = std::max(kMinBatch, neq1);
        int nblock  = (ntokens + n_swa + kMinBatch - 1)/kMinBatch;
        int first   = nek1 - nblock*kMinBatch;
        if (first > 0) {
            k = (const char *)k + int64_t(first)*stride_k;
            v = (const char *)v + int64_t(first)*stride_v;
            mask = (const uint16_t *)mask + first;
            nek1 -= first;
        }
    }

    int rk2 = neq2/nek2;
    int rv2 = neq2/nev2;
    int rk3 = neq3/nek3;
    int rv3 = neq3/nev3;

    // The hybrid Qwen3.5 cache places each quarter of its maximum cold token
    // range on one HBM NUMA node. Exact worker pinning maps thread i to
    // node i % numa_shards. Compute only node-local K/V ranges, retaining the
    // raw online-softmax state for a stable cross-node merge.
    if (return_stats && numa_shards >= 2 && nth % numa_shards == 0 &&
            int_type_k_in == GGML_TYPE_BF16 && int_type_v == GGML_TYPE_BF16 &&
            neq3 == 1 && nek3 == 1 && nev3 == 1 && Dk == 256 && Dv == 256 &&
            neq1 >= 1 && neq1 <= 16 && nek1 >= 32 && nek1 % 32 == 0 && nek2 == nev2 &&
            neq2 % nek2 == 0 && rk2 == rv2 && rk2 > 1 && rk2 <= 256 &&
            numa_cold_capacity >= nek1 && numa_cold_capacity % (32*numa_shards) == 0 &&
            work_buffer_in != nullptr) {
        const int local_nth = nth/numa_shards;
        const int node = ith % numa_shards;
        const int local_ith = ith/numa_shards;

        auto node_bounds = [&](int inode, int & first, int & last) {
            first = (int) ((int64_t) numa_cold_capacity*inode/numa_shards);
            last  = (int) ((int64_t) numa_cold_capacity*(inode + 1)/numa_shards);
            first = std::min(first, nek1);
            last  = std::min(last,  nek1);
        };

        // Pack as many token rows as the active ISA's query-tile schedule can
        // consume in one K/V traversal. The BF16 kernel's negative-stride
        // convention repeats each input mask row for its GQA heads, so
        // independent slots retain their own masks.
        if (numa_fa_batch_queries_enabled() && neq1 >= 1 && neq1 <= 16 &&
                neq2 == 24 && nek2 == 4 && rk2 == 6) {
            const int n_chunks = local_nth;
            const int narrow_group_limit = numa_fa_narrow_query_group_tokens(rk2);
            const int max_group_limit = numa_fa_max_query_group_tokens_for_layout(rk2);
            const size_t group_plan_size = numa_fa_group_plan_size(neq1);
            const size_t packed_q_size = numa_fa_align64(
                    (size_t) std::min(neq1, max_group_limit)*neq2*Dk*sizeof(float));
            int32_t * group_plan = (int32_t *) work_buffer_in;
            float * packed_q = (float *) ((char *) work_buffer_in + group_plan_size);
            float ** partial_ptrs = (float **) ((char *) packed_q + packed_q_size);

            numa_fa_group_feedback * group_feedback = nullptr;
            int group_feedback_arm = -1;
            int group_feedback_bucket = -1;
            int group_feedback_rows = 0;
            uint64_t group_started_us = 0;
            if (ith == 0) {
                int group_limit = max_group_limit;
                if (max_group_limit > narrow_group_limit && neq1 > narrow_group_limit &&
                        numa_fa_query_group_policy() == numa_fa_group_policy::adaptive) {
                    group_feedback_rows = numa_fa_max_rows_per_worker(
                        nek1, numa_cold_capacity, numa_shards, local_nth);
                    group_feedback_bucket = numa_fa_work_bucket(group_feedback_rows);
                    group_feedback = &g_numa_fa_group_feedback
                        [std::min(neq1, numa_fa_feedback_token_buckets - 1)]
                        [group_feedback_bucket];
                    group_feedback_arm = numa_fa_choose_group_arm(*group_feedback);
                    group_limit = group_feedback_arm == 0 ? narrow_group_limit : max_group_limit;
                }

                const uint64_t index = g_fa_layout_runs.fetch_add(1, std::memory_order_relaxed) + 1;
                if (index <= 8) {
                    std::fprintf(stderr,
                            "iqk_fa: layout run=%llu nth=%d tokens=%d group=%d max_group=%d "
                            "plan_bytes=%zu packed_q_bytes=%zu work=%p partials=%p\n",
                            (unsigned long long) index, nth, neq1, group_limit,
                            max_group_limit, group_plan_size, packed_q_size,
                            work_buffer_in, (void *) partial_ptrs);
                }

                const int n_groups = (neq1 + group_limit - 1)/group_limit;
                const int group_tokens = neq1/n_groups;
                const int n_larger = neq1 % n_groups;
                int first = 0;
                for (int group = 0; group < n_groups; ++group) {
                    const int count = group_tokens + (group < n_larger ? 1 : 0);
                    group_plan[1 + 2*group + 0] = first;
                    group_plan[1 + 2*group + 1] = count;
                    first += count;
                }
                group_plan[0] = n_groups;
            }

            constexpr int max_queries = numa_fa_max_query_group_tokens*6;
            constexpr int result_stride = (256 + 16)*max_queries;
            // Qwen's four KV heads fit in one per-worker scratch cohort.  A
            // single compute/merge cycle avoids a second pair of 52-thread
            // NUMA barriers while retaining independent online-softmax state
            // for every head.
            alignas(64) float local_results[4][result_stride];

            partial_ptrs[ith] = local_results[0];
            barrier(barrier_data);
            if (ith == 0 && group_feedback != nullptr) {
                group_started_us = ggml_time_us();
            }

            auto chunk_bounds = [&](int inode, int ichunk, int & first, int & last) {
                int node_first, node_last;
                node_bounds(inode, node_first, node_last);
                const int n_blocks = (node_last - node_first)/32;
                first = node_first + 32*((int64_t) n_blocks*ichunk/n_chunks);
                last  = node_first + 32*((int64_t) n_blocks*(ichunk + 1)/n_chunks);
            };

            int first, last;
            chunk_bounds(node, local_ith, first, last);
            const int n_groups = group_plan[0];
            for (int group = 0; group < n_groups; ++group) {
                const int group_first = group_plan[1 + 2*group + 0];
                const int group_tokens = group_plan[1 + 2*group + 1];
                const int n_queries = group_tokens*rk2;

                for (int row = ith; row < group_tokens*neq2; row += nth) {
                    const int local_iq1 = row/neq2;
                    const int iq2 = row - local_iq1*neq2;
                    const int ik02 = iq2/rk2;
                    const int il = iq2 - ik02*rk2;
                    const int packed_row = ik02*n_queries + local_iq1*rk2 + il;
                    const float * src = (const float *) ((const char *) q +
                            (int64_t) (group_first + local_iq1)*stride_q + (int64_t) iq2*nbq2);
                    std::memcpy(packed_q + (size_t) packed_row*Dk, src, Dk*sizeof(float));
                }
                barrier(barrier_data);

                for (int head_first = 0; head_first < nek2; head_first += 4) {
                    const int n_heads = std::min(4, nek2 - head_first);
                    if (last > first) {
                        const char * this_m = mask ? (const char *) mask +
                                (int64_t) group_first*stride_m + (int64_t) first*sizeof(uint16_t) : nullptr;
                        for (int slot = 0; slot < n_heads; ++slot) {
                            const int ik02 = head_first + slot;
                            const float * this_q = packed_q + (size_t) ik02*n_queries*Dk;
                            const char * this_k = (const char *) k +
                                    (int64_t) ik02*nbk2 + (int64_t) first*stride_k;
                            const char * this_v = (const char *) v +
                                    (int64_t) ik02*nbv2 + (int64_t) first*stride_v;
                            float * out = local_results[slot];
                            if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                                    Dk, Dv, n_queries, last - first, Dk*sizeof(float),
                                    stride_k, stride_v, mask ? -stride_m : 0, Dv,
                                    this_q, this_k, this_v, this_m, nullptr, 0,
                                    scale, softcap, out, out + Dv*n_queries,
                                    out + (Dv + 1)*n_queries)) {
                                return false;
                            }
                        }
                    }
                    barrier(barrier_data);

                    if (numa_return_partials) {
                        const size_t shard_stride = (size_t) ne1*ne2*nb1;
                        for (int output = local_ith; output < n_heads*n_queries; output += local_nth) {
                            const int slot = output/n_queries;
                            const int packed_row = output - slot*n_queries;
                            const int local_iq1 = packed_row/rk2;
                            const int il = packed_row - local_iq1*rk2;
                            const int iq1 = group_first + local_iq1;
                            const int ik02 = head_first + slot;
                            const int iq2 = ik02*rk2 + il;
                            float * Racc = (float *) ((char *) qkv + (size_t) node*shard_stride +
                                    (iq2 + (int64_t) iq1*ne1)*nb1);
                            float M = -INFINITY;
                            float S = 0.0f;
                            std::memset(Racc, 0, Dv*sizeof(float));
                            for (int ichunk = 0; ichunk < n_chunks; ++ichunk) {
                                int chunk_first, chunk_last;
                                chunk_bounds(node, ichunk, chunk_first, chunk_last);
                                if (chunk_last <= chunk_first) {
                                    continue;
                                }
                                const int thread = ichunk*numa_shards + node;
                                const float * part = partial_ptrs[thread] + slot*result_stride;
                                const float * R = part + packed_row*Dv;
                                const float * Mj = part + Dv*n_queries;
                                const float * Sj = Mj + n_queries;
                                accumulate_qkv(Dv, M, S, Mj[packed_row], Sj[packed_row], Racc, R);
                            }
                            Racc[Dv + 0] = pack_softmax_max(M);
                            Racc[Dv + 1] = S;
                        }
                    } else {
                        for (int output = ith; output < n_heads*n_queries; output += nth) {
                            const int slot = output/n_queries;
                            const int packed_row = output - slot*n_queries;
                            const int local_iq1 = packed_row/rk2;
                            const int il = packed_row - local_iq1*rk2;
                            const int iq1 = group_first + local_iq1;
                            const int ik02 = head_first + slot;
                            const int iq2 = ik02*rk2 + il;
                            float * Racc = (float *) ((char *) qkv + (iq2 + (int64_t) iq1*ne1)*nb1);
                            float M = -INFINITY;
                            float S = 0.0f;
                            for (int inode = 0; inode < numa_shards; ++inode) {
                                for (int ichunk = 0; ichunk < n_chunks; ++ichunk) {
                                    int chunk_first, chunk_last;
                                    chunk_bounds(inode, ichunk, chunk_first, chunk_last);
                                    if (chunk_last <= chunk_first) {
                                        continue;
                                    }
                                    const int thread = ichunk*numa_shards + inode;
                                    const float * part = partial_ptrs[thread] + slot*result_stride;
                                    const float * R = part + packed_row*Dv;
                                    const float * Mj = part + Dv*n_queries;
                                    const float * Sj = Mj + n_queries;
                                    accumulate_qkv(Dv, M, S, Mj[packed_row], Sj[packed_row], Racc, R);
                                }
                            }
                            Racc[Dv + 0] = pack_softmax_max(M);
                            Racc[Dv + 1] = S;
                        }
                    }
                    barrier(barrier_data);
                }
            }
            if (ith == 0 && group_feedback != nullptr) {
                const uint64_t elapsed_us = ggml_time_us() - group_started_us;
                numa_fa_observe_group_arm(*group_feedback, group_feedback_arm, elapsed_us);
                const uint64_t index = g_fa_group_observations.fetch_add(1, std::memory_order_relaxed) + 1;
                if (index <= 8 || index % 256 == 0) {
                    const uint64_t narrow_samples =
                        group_feedback->samples[0].load(std::memory_order_relaxed);
                    const uint64_t wide_samples =
                        group_feedback->samples[1].load(std::memory_order_relaxed);
                    const double narrow_us = narrow_samples > 0 ?
                        group_feedback->total_us[0].load(std::memory_order_relaxed)/
                            (double) narrow_samples : 0.0;
                    const double wide_us = wide_samples > 0 ?
                        group_feedback->total_us[1].load(std::memory_order_relaxed)/
                            (double) wide_samples : 0.0;
                    std::fprintf(stderr,
                            "iqk_fa: adaptive group observation=%llu tokens=%d rows_per_worker=%d "
                            "bucket=%d arm=%s elapsed_us=%llu narrow_samples=%llu narrow_mean_us=%.1f "
                            "wide_samples=%llu wide_mean_us=%.1f\n",
                            (unsigned long long) index, neq1, group_feedback_rows,
                            group_feedback_bucket, group_feedback_arm == 0 ? "narrow" : "wide",
                            (unsigned long long) elapsed_us,
                            (unsigned long long) narrow_samples, narrow_us,
                            (unsigned long long) wide_samples, wide_us);
                }
            }
            return true;
        }

        const int n_pairs = nek2*neq1;
        const int n_chunks = std::max(1, (local_nth + n_pairs - 1)/n_pairs);
        const size_t result_size = (Dv + 16)*rk2*sizeof(float);
        char * result_buffer = (char *) work_buffer_in;

        auto chunk_bounds = [&](int inode, int ichunk, int & first, int & last) {
            int node_first, node_last;
            node_bounds(inode, node_first, node_last);
            const int n_blocks = (node_last - node_first)/32;
            first = node_first + 32*((int64_t) n_blocks*ichunk/n_chunks);
            last  = node_first + 32*((int64_t) n_blocks*(ichunk + 1)/n_chunks);
        };
        auto partial = [&](int inode, int pair, int ichunk) -> float * {
            const size_t index = ((size_t) inode*n_pairs + pair)*n_chunks + ichunk;
            return (float *) (result_buffer + index*result_size);
        };

        const int n_local_tasks = n_pairs*n_chunks;
        for (int local_task = local_ith; local_task < n_local_tasks; local_task += local_nth) {
            const int pair = local_task/n_chunks;
            const int ichunk = local_task - pair*n_chunks;
            const int ik02 = pair/neq1;
            const int iq1 = pair - ik02*neq1;
            int first, last;
            chunk_bounds(node, ichunk, first, last);
            if (last <= first) {
                continue;
            }

            const float * this_q = (const float *) ((const char *) q +
                    (int64_t) iq1*stride_q + (int64_t) ik02*rk2*nbq2);
            const char * this_k = (const char *) k +
                    (int64_t) ik02*nbk2 + (int64_t) first*stride_k;
            const char * this_v = (const char *) v +
                    (int64_t) ik02*nbv2 + (int64_t) first*stride_v;
            const char * this_m = mask ? (const char *) mask +
                    (int64_t) iq1*stride_m + (int64_t) first*sizeof(uint16_t) : nullptr;
            float * out = partial(node, pair, ichunk);
            if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                    Dk, Dv, rk2, last - first, nbq2, stride_k, stride_v, 0, Dv,
                    this_q, this_k, this_v, this_m, nullptr, 0,
                    scale, softcap, out, out + Dv*rk2, out + (Dv + 1)*rk2)) {
                return false;
            }
        }

        barrier(barrier_data);

        if (numa_return_partials) {
            const size_t shard_stride = (size_t) ne1*ne2*nb1;
            for (int output = local_ith; output < neq1*neq2; output += local_nth) {
                const int iq1 = output/neq2;
                const int iq2 = output - iq1*neq2;
                const int ik02 = iq2/rk2;
                const int il = iq2 - ik02*rk2;
                const int pair = ik02*neq1 + iq1;
                float * Racc = (float *) ((char *) qkv + (size_t) node*shard_stride +
                        (iq2 + (int64_t) iq1*ne1)*nb1);
                float M = -INFINITY;
                float S = 0.0f;
                std::memset(Racc, 0, Dv*sizeof(float));
                for (int ichunk = 0; ichunk < n_chunks; ++ichunk) {
                    int first, last;
                    chunk_bounds(node, ichunk, first, last);
                    if (last <= first) {
                        continue;
                    }
                    const float * part = partial(node, pair, ichunk);
                    const float * R = part + il*Dv;
                    const float * Mj = part + Dv*rk2;
                    const float * Sj = Mj + rk2;
                    accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
                }
                Racc[Dv + 0] = pack_softmax_max(M);
                Racc[Dv + 1] = S;
            }
        } else {
            for (int output = ith; output < neq1*neq2; output += nth) {
                const int iq1 = output/neq2;
                const int iq2 = output - iq1*neq2;
                const int ik02 = iq2/rk2;
                const int il = iq2 - ik02*rk2;
                const int pair = ik02*neq1 + iq1;
                float * Racc = (float *) ((char *) qkv + (iq2 + (int64_t) iq1*ne1)*nb1);
                float M = -INFINITY;
                float S = 0.0f;
                for (int inode = 0; inode < numa_shards; ++inode) {
                    for (int ichunk = 0; ichunk < n_chunks; ++ichunk) {
                        int first, last;
                        chunk_bounds(inode, ichunk, first, last);
                        if (last <= first) {
                            continue;
                        }
                        const float * part = partial(inode, pair, ichunk);
                        const float * R = part + il*Dv;
                        const float * Mj = part + Dv*rk2;
                        const float * Sj = Mj + rk2;
                        accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
                    }
                }
                Racc[Dv + 0] = pack_softmax_max(M);
                Racc[Dv + 1] = S;
            }
        }
        return true;
    }

    int first_k = 0, last_k = nek1;
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nek1 > 256 && mask) {
        // This is a quick hack for SWA models.
        // Given that the mask is the same for all layers, ideally we should determine the
        // cache bounds once, and reuse for the whole graph. But even with this simple hack
        // we get non-negligible performance gains for SWA models and long context.
        auto umask = (const uint16_t *)mask;
        for (; first_k < last_k; ++first_k) {
            if (umask[first_k] == 0) break;
        }
        if (first_k == last_k) {
            fprintf(stderr, "============================== %s: found empty attention mask: nek1 = %d, first_k = %d\n", __func__, nek1, first_k);
            GGML_ABORT("Fatal error");
        }
        for (; last_k > first_k; --last_k) {
            if (umask[last_k-1] == 0) break;
        }
        int non = 32*((last_k - first_k + 31)/32);
        first_k = std::max(0, last_k - non);
        last_k = std::min(first_k + non, nek1);
        //printf("nek1 = %d, first = %d, last = %d\n", nek1, first, last);
        if (last_k - first_k <= 3*nek1/4 && (last_k - first_k)%32 == 0) {
            //printf("Reducing from %d to %d\n", nek1, last_k - first_k);
            k = (const void *)((const char *)k + first_k*stride_k);
            v = (const void *)((const char *)v + first_k*stride_v);
            mask = (const void *)((const uint16_t *)mask + first_k);
            nek1 = last_k - first_k;
        }
    }

    int int_type_k = int_type_k_in;
    auto work_buffer = work_buffer_in;
    if (neq1 >= 8) {
        uint64_t row_size = 0;
        work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
        if (int_type_k != int_type_k_in) {
            stride_k = row_size;
            nbk2 = stride_k*nek1;
            nbk3 = nbk2*nek2;
            k = work_buffer_in;
            barrier(barrier_data);
        }
    }
    //uint64_t row_size = 0;
    //auto work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
    //if (int_type_k != int_type_k_in) {
    //    stride_k = row_size;
    //    nbk2 = stride_k*nek1;
    //    nbk3 = nbk2*nek2;
    //    k = work_buffer_in;
    //    barrier(barrier_data);
    //}

    // Getting confused all the time about where to load data from and store the results to
    // (especially when combining the results from the threads).
    // So, for now, making it work just for MLA (nek2 = 1).
    // I think it would also speed up things for GQA, but I'm leaving this for another day.
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nth >= 1 && nek1/32 > 1 && nek2 == 1) {
        int nstep_k = nek1/32;
        if (nstep_k >= 4*nth) {
            int nstep_k_per_thread = (nstep_k + nth - 1)/nth;
            int ith_mid = nth;
            int nstep_k_this_thread = nstep_k_per_thread;
            if (nstep_k_per_thread*nth > nstep_k) {
                ith_mid = nstep_k - nth*(nstep_k_per_thread - 1);
                if (ith >= ith_mid) --nstep_k_this_thread;
            }
            //if (ith == 0) fprintf(stderr, "nstep_k = %d, nstep_k_per_thread = %d, ith_mid = %d\n", nstep_k, nstep_k_per_thread, ith_mid);
            nstep_k_per_thread *= 32;
            nstep_k_this_thread *= 32;

            auto kv_offset = ith <= ith_mid ? ith*nstep_k_per_thread
                                           : ith_mid*nstep_k_per_thread + (ith - ith_mid)*nstep_k_this_thread;
            auto kth = (const char *)k + kv_offset*stride_k;
            auto vth = (const char *)v + kv_offset*stride_v;
            auto qth = (const char *)q;
            auto mth = mask ? (const char *)mask + kv_offset*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

            auto work = (char *)work_buffer;
            auto size_thread = (Dv + 16)*rk2*sizeof(float);
            auto result_buffer = work;
            auto work_this_thread = (float *)(result_buffer + ith*size_thread);
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, nstep_k_this_thread, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                     (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                     scale, softcap,
                     work_this_thread, work_this_thread + (Dv+0)*rk2, work_this_thread + (Dv+1)*rk2)) return false;

            barrier(barrier_data);

            for (int j = ith; j < rk2; j += nth) {
                auto Racc = qkv + j*nb1/sizeof(float);
                float M = -INFINITY, S = 0;
                for (int jth = 0; jth < nth; ++jth) {
                    auto R = (const float *)(result_buffer + jth*size_thread);
                    auto Mj = R + Dv*rk2;
                    auto Sj = Mj + rk2;
                    R += j*Dv;
                    accumulate_qkv(Dv, M, S, Mj[j], Sj[j], Racc, R);
                }
                if (return_stats) {
                    Racc[Dv + 0] = pack_softmax_max(M);
                    Racc[Dv + 1] = S;
                } else {
                    float norm = S > 0 ? 1/S : 1;
                    for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
                }
            }
            return true;
        }
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int ith_k = ith%gcd_k;
            int ith_q = ith/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                int ith_mid = nth_k;
                int nq_this_thread = nq_per_thread;
                if (nq_per_thread*nth_k > rk2) {
                    ith_mid = rk2 - nth_k*(nq_per_thread - 1);
                    if (ith_q >= ith_mid) --nq_this_thread;
                }
                int j_mid = ith_mid*nq_per_thread;
                auto work = (char *)work_buffer;
                auto size_thread = (Dv + 16)*nq_per_thread*sizeof(float);
                auto result_buffer = work;

                auto kth = (const char *)k + ith_k*(nek1/gcd_k)*stride_k;
                auto vth = (const char *)v + ith_k*(nek1/gcd_k)*stride_v;
                auto q_offset = ith_q < ith_mid ? ith_q*nq_per_thread*nbq2 : (ith_mid*nq_per_thread + (ith_q - ith_mid)*nq_this_thread)*nbq2;
                auto qth = (const char *)q + q_offset;
                auto mth = mask ? (const char *)mask + ith_k*(nek1/gcd_k)*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

                // Each thread will produce a result of size Dv*nq_this_thread*sizeof(float)
                // In addition, we need M, S for the nq_this_thread rows the thread is processing
                // => (Dv + 2)*nq_per_thread*sizeof(float). We use (Dv + 16) instead to make sure threads are not
                // writing onto the same cache line.
                auto work_this_thread = (float *)(result_buffer + ith*size_thread);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, nq_this_thread, nek1/gcd_k, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                            (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                            scale, softcap,
                            work_this_thread, work_this_thread + (Dv+0)*nq_this_thread, work_this_thread + (Dv+1)*nq_this_thread)) return false;

                barrier(barrier_data);

                // There are nek1/gcd_k contributions for each j that we need to sum up
                // Thread i computed k/v (i%gcd_k)*(nek1/gcd_k) for j (i/gcd_k)*(rk2/nth_k)...((i/gcd_k)+1)*(rk2/nth_k) and results at offset i*size_thread

                // TODO: simdify this
                // TODO: if nth > rk2, have threads process portions of the rows instead of entire rows as it is now
                for (int j = ith; j < rk2; j += nth) {
                    auto Racc = qkv + j*nb1/sizeof(float);
                    float M = -INFINITY, S = 0;
                    int jth_first, jj, nq_this_j;
                    if (j < j_mid) {
                        jth_first = j/nq_per_thread;
                        jj = j%nq_per_thread;
                        nq_this_j = nq_per_thread;
                    } else {
                        jth_first = ith_mid + (j - j_mid)/(nq_per_thread-1);
                        jj = (j - j_mid)%(nq_per_thread-1);
                        nq_this_j = nq_per_thread - 1;
                    }
                    jth_first *= gcd_k;
                    for (int jth = jth_first; jth < jth_first + gcd_k; ++jth) {
                        auto R = (const float *)(result_buffer + jth*size_thread);
                        auto Mj = R + Dv*nq_this_j;
                        auto Sj = Mj + nq_this_j;
                        R += jj*Dv;
                        accumulate_qkv(Dv, M, S, Mj[jj], Sj[jj], Racc, R);
                    }
                    if (return_stats) {
                        Racc[Dv + 0] = pack_softmax_max(M);
                        Racc[Dv + 1] = S;
                    } else {
                        float norm = S > 0 ? 1/S : 1;
                        for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
                    }
                }
                return true;
            }
        }
    }

    if (neq3 == 1 && rk2 > 1 && rk2 == rv2 && neq1 == 1 && nth >= 1 && nek2*nek1 >= 32*nth) {
        auto result_size = (Dv + 16)*rk2*sizeof(float);
        int gcd = simple_gcd(nek2, nth);
        int nth_k  = nth/gcd;
        int nek2_k = nek2/gcd;
        int nchunk = nek2_k*nek1/32;
        int npt = (nchunk + nth_k - 1)/nth_k;
        int nk;
        if (npt*nth_k == nchunk) {
            nk = 32 * (nek2*nek1/(32*nth));
        } else {
            //int nm = std::max(1, npt/8);
            int nm = 1;
            while (true) {
                if (nm*4 >= npt) break;
                nm *= 2;
            }
            nk = 32*nm;
        }
        //int nk = 32 * (nek2*nek1/(32*nth));
        int nkk = (nek1 + nk - 1)/nk;
        int nstep_k = nek2*nkk;
        //if (ith == 0) printf("rk2 = %d, nek1 = %d, nek2 = %d, nk = %d, nkk = %d, nstep_k = %d\n", (int)rk2, (int)nek1, (int)nek2, nk, nkk, nstep_k);
        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
            int ik01 = nk*(istep_k - ik02*nkk);
            int this_nk = ik01 + nk <= nek1 ? nk : nek1 - ik01;
            if (this_nk <= 0) break;
            auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
            auto this_q = (const float *)((const char *)q + ik02*rk2*nbq2);
            auto this_k = (const char *)k + ik01*stride_k + ik02*nbk2;
            auto this_v = (const char *)v + ik01*stride_v + ik02*nbv2;
            auto this_m = mask ? (const char *)mask + ik01*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, stride_k, stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m, nullptr, 0,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }

        barrier(barrier_data);

        // We have nkk results for each head
        for (int iq2 = ith; iq2 < neq2; iq2 += nth) {
            // ik02*rk2 + il = iq2 (il = 0...rk2-1) => ik02 = iq2/rk2, il = iq2%rk2;
            int ik02 = iq2/rk2;
            int il = iq2 - ik02*rk2;
            auto Racc = qkv + iq2*nb1/sizeof(float);
            //std::memset(Racc, 0, Dv*sizeof(float));
            float M = -INFINITY, S = 0;
            for (int ikk = 0; ikk < nkk; ++ikk) {
                int istep_k = ik02*nkk + ikk;
                auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
                const float * R  = this_result + il*Dv;
                const float * Mj = this_result + Dv*rk2;
                const float * Sj = Mj + rk2;
                accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
            }
            if (sinks) {
                float s = ((const float *)sinks)[iq2];
                if (s > M) {
                    float m = expf(M - s);
                    for (int i = 0; i < Dv; ++i) Racc[i] *= m;
                    S = S*m + 1;
                } else {
                    S += expf(s - M);
                }
            }
            if (return_stats) {
                Racc[Dv + 0] = pack_softmax_max(M);
                Racc[Dv + 1] = S;
            } else {
                float norm = S > 0 ? 1/S : 1;
                for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
            }
        }
        return true;
    }

    // Hybrid CPU/GPU attention asks for the unnormalized online-softmax state
    // for a small target-verification batch.  The generic scheduling below
    // distributes query heads independently.  For GQA that makes all query
    // heads reread the same K/V head.  Process the GQA group together instead:
    // all query heads for one (KV head, token) share a single K/V traversal.
    //
    // Keep the single-token path above: it also splits the long K dimension
    // across workers and is faster when there are only a few KV heads.
    if (return_stats && neq3 == 1 && neq1 > 1 && neq1 <= 16 &&
            rk2 > 1 && rk2 == rv2 && rk2 <= 256) {
        const int n_tasks = nek2*neq1;
        for (int task = ith; task < n_tasks; task += nth) {
            const int ik02 = task/neq1;
            const int iq1  = task - ik02*neq1;
            const int iq2  = ik02*rk2;

            const float * this_q = (const float *)((const char *)q + iq1*stride_q + iq2*nbq2);
            const char  * this_k = (const char *)k + ik02*nbk2;
            const char  * this_v = (const char *)v + ik02*nbv2;
            const char  * this_m = mask ? (const char *)mask + iq1*stride_m : nullptr;
            float * out = (float *)((char *)qkv + (iq2 + iq1*ne1)*nb1);

            float Mbuf[256];
            float Sbuf[256];
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                    Dk, Dv, rk2, nek1, nbq2, stride_k, stride_v, 0, nb1/sizeof(float),
                    this_q, this_k, this_v, this_m, nullptr, 0,
                    scale, softcap, out, Mbuf, Sbuf)) {
                return false;
            }
            const int stride_out = nb1/sizeof(float);
            for (int j = 0; j < rk2; ++j) {
                out[j*stride_out + Dv + 0] = pack_softmax_max(Mbuf[j]);
                out[j*stride_out + Dv + 1] = Sbuf[j];
            }
        }
        return true;
    }

    // I keep changing my mind what is the best strategy to split the threads when processing
    // multiple heads. This is my current thinking, the commented out code below was the previous.
    int ntg = nth/simple_gcd(neq2*neq3, nth);
    int neq1g = (neq1 + ntg - 1)/ntg;
    //int64_t work_per_slice = D*nek1*neq1;
    //int ntg = 1;
    //
    // When neq1 is large, it is better to have more than one thread process one (iq2,iq3) matrix
    // But we also want each thread to process the same amount of rows, so neq1 must be a multiple of
    // the number of threads processing the (iq2, iq3) matrix.
    //
    //if (neq1 >= 8*nth) {
    //    if      (nth%8 == 0 && neq1%8 == 0 && work_per_slice >= (1 << 23)) ntg = 8;
    //    else if (nth%4 == 0 && neq1%4 == 0 && work_per_slice >= (1 << 21)) ntg = 4;
    //    else if (nth%2 == 0 && neq1%2 == 0 && work_per_slice >= (1 << 19)) ntg = 2;
    //}
    int counter = 0;
    for (int64_t iq3 = 0; iq3 < neq3; iq3++) {
        for (int64_t iq2 = 0; iq2 < neq2; iq2++) {
            auto sinksf = sinks ? (const float *)sinks + iq2 : nullptr;
            if (counter++ % (nth/ntg) == ith/ntg) {
                int iq1 = (ith%ntg)*neq1g;
                int this_neq1 = std::min(neq1g, neq1-iq1);
                if (this_neq1 > 0) {
                if (return_stats && this_neq1 > 1024) return false;
                float Mbuf[1024];
                float Sbuf[1024];
                float * out = (float *)((char *)qkv + (iq3*ne2*ne1 + iq2 + iq1*ne1)*nb1);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                        Dk, Dv, this_neq1, nek1, stride_q, stride_k, stride_v, stride_m, ne1*nb1/sizeof(float),
                        (const float *)((const char *)q + iq2*nbq2 + iq3*nbq3 + iq1*stride_q),
                        (const void  *)((const char *)k + iq2/rk2*nbk2 + iq3/rk3*nbk3),
                        (const void  *)((const char *)v + iq2/rv2*nbv2 + iq3/rv3*nbv3),
                        mask ? (const void  *)((const char *)mask + iq1*stride_m) : nullptr, sinksf, 0,
                        scale, softcap,
                        out, return_stats ? Mbuf : nullptr, return_stats ? Sbuf : nullptr)) return false;
                if (return_stats) {
                    const int stride_out = ne1*nb1/sizeof(float);
                    for (int j = 0; j < this_neq1; ++j) {
                        out[j*stride_out + Dv + 0] = pack_softmax_max(Mbuf[j]);
                        out[j*stride_out + Dv + 1] = Sbuf[j];
                    }
                }
                }
            }
        }
    }

    return true;
}

#else

bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
                            [[maybe_unused]] int neq3, [[maybe_unused]] int neq2, [[maybe_unused]] long nbq3, [[maybe_unused]] long nbq2,
                            [[maybe_unused]] int nek3, [[maybe_unused]] int nek2, [[maybe_unused]] long nbk3, [[maybe_unused]] long nbk2,
                            [[maybe_unused]] int nev3, [[maybe_unused]] int nev2, [[maybe_unused]] long nbv3, [[maybe_unused]] long nbv2,
                            [[maybe_unused]] int ne2,  [[maybe_unused]] int ne1,  [[maybe_unused]] long nb1,
                            [[maybe_unused]] int type_k,             // type of k
                            [[maybe_unused]] int type_v,             // type of v
                            [[maybe_unused]] int Dk,                 // K head size
                            [[maybe_unused]] int Dv,                 // V head size
                            [[maybe_unused]] int nq,                 // number of columns in q
                            [[maybe_unused]] int nk,                 // number of rows in k
                            [[maybe_unused]] int stride_q,           // distance between q columns in bytes
                            [[maybe_unused]] int stride_k,           // distance between k rows in bytes
                            [[maybe_unused]] int stride_v,           // distance between v rows in bytes
                            [[maybe_unused]] int stride_m,           // distance between mask rows (in bytes
                            [[maybe_unused]] const void  * q,        // q matrix.
                            [[maybe_unused]] const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            [[maybe_unused]] float         scale,    // scale applied before softmax
                            [[maybe_unused]] float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            [[maybe_unused]] bool          return_stats,
                            [[maybe_unused]] int           numa_shards,
                            [[maybe_unused]] int           numa_cold_capacity,
                            [[maybe_unused]] bool          numa_return_partials,
                            [[maybe_unused]] float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            [[maybe_unused]] int ith, [[maybe_unused]] int nth, [[maybe_unused]] int n_swa, [[maybe_unused]] ggml_tensor * indexer) {
    return false;
}

#endif
