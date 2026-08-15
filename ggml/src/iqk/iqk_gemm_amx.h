#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

bool iqk_amx_int8_runtime_available();

// The generic matmul path may reuse a thread-local conversion buffer for
// different tensors.  Such storage must never enter the persistent AMX
// prepack cache.
void iqk_amx_set_transient_weights(bool transient);

// Describes the immutable matrix owning the row slice passed to a kernel.
// IQ3 uses this to share one full-matrix prepack across worker threads even
// when the scheduler changes row partitions between evaluations.
void iqk_amx_set_weight_origin(const void * base, int row_offset, int total_rows);

// Returns false before issuing an AMX instruction when XTILEDATA is not
// available to the calling thread. Callers can then use their existing VNNI
// implementation as a safe fallback.
bool iqk_amx_mul_mat_q4_k_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y);

bool iqk_amx_mul_mat_q5_k_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y);

bool iqk_amx_mul_mat_q6_k_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y);

bool iqk_amx_mul_mat_iq3_s_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y);

bool iqk_amx_mul_mat_q8_0_r8(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y);

bool iqk_amx_mul_mat_iq4_nl_r4(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x, int nrc_y);

#endif
