# Xeon CPU Max AMX/HBM benchmark record

This file records the experimental AMX operator acceptance point and the
subsequent NUMA/HBM token-generation work. It is intended to keep later layout
and heterogeneous-inference changes comparable with the same operator build.

## Snapshot

- Date: 2026-08-15 through 2026-08-16
- CPU: Intel Xeon CPU Max 9470C, 52 physical cores / 104 threads
- Memory: 64 GiB-class on-package HBM exposed as four SNC NUMA nodes
- AMX implementation revision: `1b3c6067`
- Source state: AMX implementation committed; this benchmark record is kept
  local via `.git/info/exclude`
- Model: `weights/Qwen3.8-27B-Q4_K_M.gguf`, 26.896B parameters, 16.806 GB file
- Build directory: `build-amx`, Release, `GGML_NATIVE=ON`,
  `GGML_AMX_BF16=ON`, `GGML_AMX_INT8=ON`, tests enabled

The model is approximately 74% Q4_K, 20.2% Q6_K, and 5.5% Q5_K. `-rtr 1` is
required to expose its Q4_K/Q5_K/Q6_K tensors to the R4 AMX kernels.

## PP512 acceptance result

The accepted cache policy is a compact Q4_K cache plus a 6144 MiB budget for
fully expanded Q4_K entries. The process was capped at 46 GiB with swap
disabled, pages were interleaved over all HBM NUMA nodes, and one hardware
thread per physical core was used:

```bash
GGML_AMX_Q4_K_EXPANDED_BUDGET_MB=6144 GGML_AMX_STATS=1 \
systemd-run --user --quiet --scope -p MemoryMax=46G -p MemorySwapMax=0 \
numactl --interleave=all taskset -c 0-51 /usr/bin/time -v \
./build-amx/bin/llama-bench \
    -m weights/Qwen3.8-27B-Q4_K_M.gguf \
    -p 512 -n 0 -b 512 -ub 512 -t 52 -r 10 -rtr 1 -o json
```

Results:

- First/cold sample: 51.45 tokens/s (includes persistent prepack construction)
- Nine warm samples: 216.56, 227.82, 221.40, 226.00, 222.32, 226.91,
  218.60, 219.65, and 225.52 tokens/s
- Warm arithmetic mean: **222.75 tokens/s**
- Peak resident set: **43,501,432 KiB = 41.49 GiB**
- AMX persistent prepack bytes: 26,881,443,072 bytes = 25.04 GiB
- Q4_K_R4 calls/tiles: 693,700 / 14,767,104,000
- Q5_K_R4 calls: 47,520
- Q6_K_R4 calls: 122,090
- Wall time: 37.69 seconds; no cgroup OOM

This satisfies the operator-stage acceptance target of 40--42 GiB RSS and at
least 210 tokens/s. Further work should treat this result as the PP512 operator
baseline and focus on NUMA/HBM placement rather than additional kernel tuning.

## Cache modes

- Default: compact Q4_K VNNI data and scales; expand only the current L1-sized
  tile before AMX execution.
- `GGML_AMX_Q4_K_EXPANDED_BUDGET_MB=<MiB>`: compact cache plus a bounded amount
  of fully expanded Q4_K data. The 6144 MiB setting produced the accepted
  memory/performance point above.
- `GGML_AMX_Q4_K_EXPANDED=1`: fully expanded Q4_K cache. This is useful for
  kernel experiments but is not the accepted inference configuration because
  of its larger resident set.

## Focused validation and kernel measurements

`build-amx/bin/test-amx-int8` passes all focused correctness cases. The largest
observed absolute errors for the Q4_K/Q5_K/Q6_K cases were approximately
3e-5--6e-5.

Representative single-kernel measurements from the same machine:

| Kernel / shape | AMX | AVX-512 | AMX / AVX-512 |
| --- | ---: | ---: | ---: |
| Q4_K_R4, N=512, compact cache | 259.68 GMAC/s | -- | -- |
| Q4_K_R4, N=512, expanded cache | 261.62 GMAC/s | -- | -- |
| Q5_K_R4, N=512 | 265.86 GMAC/s | 78.93 GMAC/s | 3.37x |
| Q8_0_R8, N=64 | 193.0 GMAC/s | 114.9 GMAC/s | 1.68x |
| IQ3_S_R4, N=32 | about 108 GMAC/s | about 46 GMAC/s | about 2.35x |

These microbenchmarks are diagnostic data rather than end-to-end acceptance
criteria. In particular, compact Q4_K decoding is amortized by dispatching up
to N=512 in one call; smaller dispatch chunks repeat decode work and should not
be used for the recorded PP512 comparison.

## Dense TG128 HBM baseline

The first token-generation baseline used CPU-only inference, runtime repacking,
one hardware thread per physical core, and interleaved memory placement:

```bash
systemd-run --user --quiet --scope -p MemoryMax=30G -p MemorySwapMax=0 \
numactl --interleave=all taskset -c 0-51 /usr/bin/time -v \
./build-amx/bin/llama-bench \
    -m weights/Qwen3.8-27B-Q4_K_M.gguf -ngl 0 \
    -p 0 -n 128 -t 52 -r 5 -rtr 1 -o json
```

- Five-run mean: **17.1909 tokens/s**, standard deviation 0.0542 tokens/s
- Samples: 17.2811, 17.1706, 17.1759, 17.1906, 17.1362 tokens/s
- Peak resident set: 16,683,820 KiB = 15.91 GiB
- No major faults or swaps

A three-run repeat at revision `1b3c6067` produced 17.0735 +/- 0.0192
tokens/s. During its steady region, direct uncore HBM counters initially
measured about 280 GB/s of reads. The counter mapping was subsequently
cross-checked with a controlled STREAM-like reader: `uncore_hbm_0..7` map to
node 0, `8..15` to node 1, `16..23` to node 2, and `24..31` to node 3; the
PMU/program comparison agreed at about 74/75 GB/s for one node.

Validated steady measurements were:

- Interleaved dense baseline: about **282.2 GB/s read**, **3.1 GB/s write**
- Row-sharded dense path: about **306.6 GB/s read**, **3.3 GB/s write**
- The row-sharded model pages were physically balanced at about 4.08 GB per
  NUMA node

The old 303 GB/s STREAM figure must not be treated as a hard practical ceiling:
the accepted row-local kernel exceeded it slightly, and bandwidth depends on
access pattern and concurrent controller use. The important finding is that
interleaving balanced aggregate traffic but still caused remote row access;
matching output-row scheduling to physical page placement improved both useful
bandwidth and token rate.

Thread and policy boundary checks:

- 39 threads with `--numa distribute`: 14.8058 +/- 0.0779 tokens/s
- 52 threads with `--numa distribute`: 17.2995 +/- 0.0739 tokens/s
- 104 SMT threads: 6.9285 +/- 0.3417 tokens/s and 258,505 involuntary
  context switches; SMT must not be used for this TG path
- Temporarily disabling automatic NUMA balancing: 17.2147 +/- 0.0791
  tokens/s, no improvement; the host setting was restored to `1`

`--numa distribute` only sets worker affinity (thread `i` goes to NUMA node
`i % 4`); it does not place memory. The recommended pairing on this host is
therefore `numactl --interleave=all` for pages plus `--numa distribute` for
workers, with `-t 52` and no SMT siblings.

## Accepted dense row-sharded result

The experimental path is enabled with `GGML_NUMA_ROW_SHARD=1`, requires
`--numa distribute`, and implies exact physical-core pinning unless explicitly
disabled with `GGML_NUMA_EXACT_PIN=0`. For model-page binding it also requires
`-mmp 0`; `-thp 1` was used in the accepted run.

```bash
systemd-run --user --quiet --scope -p MemoryMax=30G -p MemorySwapMax=0 \
env GGML_NUMA_ROW_SHARD=1 taskset -c 0-51 \
./build-amx/bin/llama-bench \
    -m weights/Qwen3.8-27B-Q4_K_M.gguf -ngl 0 \
    -p 0 -n 128 -t 52 -r 10 -rtr 1 -mmp 0 -thp 1 \
    --numa distribute -o json
```

- Strict same-build baseline: **17.233 tokens/s**
- Row-sharded result: **18.5708 +/- 0.0427 tokens/s**
- Improvement: **7.76%**
- Peak RSS: **15.91 GiB**

The page layout splits every output-row plane into four node-local ranges, and
the worker index is reordered so each node consumes its matching range. This
is a locality improvement rather than a numerical kernel change.

## Numerical comparison

An AVX-versus-AMX logits comparison on the dense 27B model produced:

- Mean KL divergence: **0.002018 +/- 0.000143**
- Perplexity change: **+0.0822%**
- Logit correlation: **99.95%**
- Top-1 agreement: **98.627%**
- Same-path self-control KL: zero

Together with the dialogue smoke test and focused operator tests, this is
consistent with the expected quantized-kernel rounding difference rather than
a functional accuracy regression.

## Functional dialogue smoke test

The optimized 27B dense path was also exercised through `llama-cli` with the
model's chat template, runtime repacking, a 4096-token context, CPU-only
execution, compact AMX cache, and a Chinese instruction asking it to role-play
a cat-girl, say good morning, calculate `37 + 58`, and give encouragement.

The first answer was coherent, stayed in character, and returned the correct
result (`95`):

> 早上好呀～☀️ 小满伸了个懒腰，尾巴尖轻轻晃了晃，凑过来蹭蹭你的手心，喵～
>
> 37 加 58 嘛，等于 **95** 哦。
>
> 今天也要像晒太阳的猫一样，慢悠悠地、暖暖的，把每一件事都过舒服了呀。你很棒，喵～🐾

A 30 GiB cgroup was too small once prompt processing constructed the compact
AMX prepack and killed only the scoped process. Repeating with `MemoryMax=40G`
succeeded; peak RSS was 35,880,968 KiB (34.22 GiB), with no swap. Piping one
line into interactive `llama-cli` leaves EOF as repeated empty turns, so the
test process was terminated after validating the first response. That is CLI
input-driving behavior, not a model-output or quantization error.

## 35B-A3B MoE TG128 result

Model: `weights/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`, 34.661B total parameters,
256 routed experts, 8 active experts, 40 layers, 2048 hidden size, and 512
expert intermediate size. The file is 18.030 GB. The accepted command is:

```bash
systemd-run --user --quiet --scope -p MemoryMax=30G -p MemorySwapMax=0 \
env GGML_NUMA_ROW_SHARD=1 taskset -c 0-51 \
./build-amx/bin/llama-bench \
    -m weights/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf -ngl 0 \
    -p 0 -n 128 -t 48 -r 10 -rtr 1 -mmp 0 -thp 1 \
    --numa distribute -o json
```

Accepted progression:

| Stage | TG128 |
| --- | ---: |
| Strict same-build interleaved baseline, 52 threads | 50.27 tok/s |
| Initial row-sharded path | 59.63 tok/s |
| 48-thread expert-balanced layout | about 61.1 tok/s |
| Compact active-expert list and balanced row chunks | 63.51 tok/s |
| IQ3_S vector-gather lookup | **66.20 tok/s** |

The final strict IQ3_S A/B was 63.176 versus 66.204 tokens/s over ten samples,
a conservative **4.79%** improvement from one lookup substitution. Discarding
the first two cold samples, the accepted path averaged **67.459 +/- 0.563
tokens/s**. Relative to the 50.27 tokens/s same-build baseline, the full MoE
work improved average TG128 by about **31.7%** (about 34.2% on the steady
samples).

The important implementation changes were:

- row-shard each expert plane over the four HBM NUMA nodes and schedule only
  its node-local output chunks;
- split chunk boundaries with balanced integer endpoints, avoiding short or
  empty tail chunks;
- choose 12 chunks for the 512-row, 8-active-expert up/gate projection so each
  of the 12 workers per node receives exactly two tasks;
- build one compact active-expert list per operation instead of rescanning all
  256 expert slots from every task;
- replace 32 scalar IQ3_S table loads per inner group with AVX2 gathers. At
  `M=512, N=1, K=2048`, this changed the microkernel from about 9.03 to 17.51
  GMAC/s with an identical checksum.

HBM sampling before the final IQ3_S change measured **158.3 GB/s read** and
**4.4 GB/s write**, distributed almost perfectly over the four nodes. A later
cycle profile moved IQ3_S from roughly 16% to 7.5% of samples; Q8_0 `N=1`,
OpenMP synchronization, Q6_K, and IQ4_NL became the remaining large costs.

Rejected experiments are recorded to avoid repeating them:

- forcing the Q8_0 `N=1` kernel from 256-bit to 512-bit was 0.39% slower;
- lowering the Q8_0 AMX threshold to `N=1` gave 31.39 versus 31.26 GMAC/s,
  effectively no gain, while `N=8` regressed;
- merged up/gate expert storage improved only about 0.45%;
- a dynamic 52-thread dense / 48-thread expert scheduler produced 64.71 versus
  64.82 tokens/s and was reverted.

This is the CPU-only MoE operator/layout stopping point. Further work should
require a credible double-digit end-to-end gain and target heterogeneous GPU
execution, speculative block verification, or hot/cold KV placement rather
than adding scheduler or kernel complexity for sub-5% results.

## Heterogeneous serving follow-up (2026-08-18)

The production long-context work now deliberately uses spare HBM bandwidth for
the output projection while keeping all 65 repeating transformer layers on the
RTX 3090 (`-ngl 65`). This is a capacity trade, not a claim that the CPU-only MoE
baseline above became the end-to-end path. Pinning a host-resident
`result_output` tensor to the CPU backend reduced each embedded MTP CUDA compute
buffer from 1,247 MiB to 139 MiB; with two physical slots, peak VRAM fell from
23,396 to 21,164 MiB. The same 20K/40K two-request A/B improved aggregate
prompt-window PP from 556.22/336.54 to 612.89/356.15 tok/s and decode from
37.03/23.35 to 38.45/23.72 tok/s.

The distinction between bandwidth and capacity is important on this host. The
historical 256K generation sample used about 169.5 GB/s HBM reads versus roughly
410 GB/s raw copy, and PCIe RX/TX was only about 822/154 MB/s. In the newer 40K
decode sample, PCIe RX was about 0.49 GiB/s and the process averaged about 13.2
CPU cores. Moving the output projection to HBM therefore spends underused
resources to recover scarce VRAM. By contrast, moving every private MTP KV
attention to HBM saved another 1.60 GiB but reduced decode throughput because of
per-context synchronization latency; it was reverted.

Absolute HBM counters are not available to the ordinary benchmark user on the
current boot (`kernel.perf_event_paranoid=4`). The production observability JSON
marks them unavailable instead of substituting NVIDIA memory-utilization. Run
the existing LIKWID/uncore recipe as root in a separate measurement window, and
do not multiplex all HBM PMUs on CPU 0 while CPU 0 is also the OpenMP coordinator;
that setup previously depressed the measured generation rate and is monitoring
interference, not a model regression. See
`xeon-max-3090-hybrid-long-context-plan.md` for the multi-request scheduler,
MTP-depth sweep, disk-cache workload, and context-growth results.

### Adaptive 128K/200K serving evidence

The production scheduler no longer fixes MTP-4, a two-request cohort, or a
512-token mixed-prefill quantum from the earlier 40K sweep. With
`--mtp-adaptive-scheduling`, candidate depths come from the configured model
limit, candidate widths come from the runnable slots, and prompt quanta come
from the live batch/ubatch and KV-contiguity envelope. Runtime aggregate
tokens/s selects among them. Transferred context/concurrency history is only a
dimensionless prior: every inherited arm must receive a local observation.
Decode width is updated only after that width's depth controller has locally
covered its arms and is using its current best depth, so cold depth exploration
is not misattributed to the cohort width.

A complete three-session, two-physical-slot run grew every session from 20K to
200K (`tmp/bench-adaptive-fs-aware-growth200k-v13-20260818`). It processed
603,658 prompt and 9,600 generated tokens in 4,842.70 s. Aggregate prompt-union
throughput fell from 516.33 tok/s at 20K to 91.63 tok/s at 200K; the final 200K
decode union was 10.15 tok/s. All 30 prefix-reuse guards passed. There was no
CUDA OOM, disk ENOSPC, KV allocation retry, or fragmented-restore failure,
despite a peak KV fragmentation ratio of 0.994236. Scatter restore handled seven
large states, and a live largest-hole cap prevented impossible prompt batches.
Atomic-write headroom caused three deliberate disk-cache evictions; the 46,000
MiB logical cache had a 50.165 GiB transient peak and only 0.872 GiB minimum
filesystem headroom.

The follow-up 128K run
(`tmp/bench-adaptive-joint-128k-long-v14-20260818`) used two 128,026-token
prompts and generated 2,048 tokens per request. The controller locally explored
MTP depths 0--8 and widths 1--2, then preferred width 2; at its 256th comparable
width update, normalized recent rewards were 1.102 for width 2 and 0.981 for
width 1. Aggregate decode was 17.65 tok/s (8.91/8.82 per request), with a
232.10 s decode union.

This does not invalidate the historical 34.45 tok/s at 128K. That result was a
single `llama-cli` context with a 24,576-token hot ring, `-ngl 99`, one MTP
context, and the output projection on CUDA. The capacity run used two target
slots plus two private MTP contexts, a 4,096-token hot ring, `-ngl 65`, and a
host output projection. Supporting two MTP-8 sequences alone reserved 2,405.25
MiB of CUDA per-step recurrent checkpoints, and peak VRAM was already 23,750
MiB. During its decode window, GPU utilization averaged only 20.00%, PCIe RX/TX
averaged 1.075/0.120 GiB/s, and the process averaged 21.34 CPU cores. Together
with the earlier 169.5/410 GB/s HBM measurement, this points to private-MTP,
recurrent-checkpoint, small-graph, and CPU/GPU synchronization latency rather
than exhausted HBM or PCIe bandwidth.

The next credible optimization is therefore to reduce or reuse the recurrent
checkpoint workspace and/or batch private MTP contexts together, then spend the
recovered VRAM on returning the output projection to CUDA. PagedAttention is
still useful for capacity, prefix sharing, and long-lived fragmentation, but a
block table alone cannot remove these recurrent and heterogeneous execution
barriers.

### Accepted 128K throughput recovery and 200K rerun

The `17.65 tok/s` result above is now superseded as the best two-request 128K
result. It was a capacity layout, not evidence that two-way scheduling was
intrinsically slower than a single queue. Three changes removed the dominant
bubbles:

- recurrent speculative checkpoints now allocate one aggregate draft-row
  budget instead of a sequence-by-depth Cartesian product;
- exact all-masked cold-KV pages are skipped by the CPU BF16 flash-attention
  path (zero numerical difference in the focused test, about 2.33x AMX and
  2.01x AVX speedup in the sparse microcase);
- a variance-aware online controller learns depth, runnable cohort width, and
  mixed-prefill quantum from measured token/time samples. Candidate sets are
  derived from model and live topology; there is no 128K/Xeon/3090 lookup table.

The accepted performance-tier run is
`tmp/bench-goal-dual128k-variance-ucb-v29-20260818`: two approximately
128,026-token prompts, 2,048 generated tokens each, MTP-8, a 24,576-row hot
ring, all model layers including output on CUDA, BF16 target KV, and Q8_0 draft
KV. Aggregate decode reached **37.934 tok/s** (19.037/20.070 per request), with
a 107.98 s decode union and 5.94 s finish skew. This is 9.95% above the
historical 34.5 tok/s single-request target and 55.65% above the pre-pruning
two-request v20 result of 24.37 tok/s. Exact page pruning skipped 48.17% of
356.343 million checked page/query combinations. Peak VRAM was 24,026 MiB;
there was no OOM, full prompt reprocess, KV retry, or restore failure.

The production capacity tier remains separate: output stays in CPU HBM,
`-ngl 65`, hot ring 4,096, and `ctx=409600`. The corrected full-scale run
`tmp/bench-goal-production-growth200k-hbm-output-v39-20260818` used three
logical sessions over two physical slots and completed all 20K--200K stages.
At 200K it generated 6,144 tokens with a **15.865 tok/s** decode union and
100.06 tok/s prompt union; all three reuse guards passed. Relative to the old
v13 capacity run, final decode improved from 10.154 tok/s (+56.23%) and stage
wall time fell from 1,284.44 to 986.10 s.

The run exercised seven fragmented full-state restores and four local partial
checkpoint restores, including a successful 181K restore in 41.32 ms. There
were zero CUDA OOMs, ENOSPC events, KV retries, or restore failures. Peak VRAM
was 23,916 MiB. The 46,000 MiB disk tier reached a 49.89 GiB atomic-write peak
with 1.15 GiB minimum filesystem headroom; four evictions, three specifically
for atomic-write headroom, kept every snapshot consistent. Exact attention
page pruning skipped 47.67% of 1.087 billion checked page/query combinations.

This reinforces the bandwidth conclusion rather than reversing it. In the
full capacity run, PCIe RX averaged about 9.15 GiB/s and reached 26.14 GiB/s at
P95 during prefill bursts, while decode periods were much lighter. The CPU
process averaged about 5.99 cores over the complete run. Ordinary-user HBM PMU
access remains unavailable, so no synthetic HBM number is reported. The
accepted architecture spends spare HBM/PCIe capacity on the output projection
only when the GPU capacity envelope requires it; the 128K performance tier
returns that projection to CUDA.

PagedAttention remains a credible future capacity project, especially for
copy-on-write prefix sharing and long-lived fragmentation. It was not the
shortest path to this throughput gain: this model has canonical KV in HBM, a
GPU hot ring, recurrent state, and MTP checkpoints, so a target-only block
table would not remove the private-MTP and small-graph synchronization costs.
Mask-driven page pruning supplied exact fine-grained compute scheduling without
requiring those four state domains to be rewritten at once.

The final build and the 21 independently runnable CTest cases passed 21/21.
The remaining four of 25 were reproduced as pre-existing environment/fixture
failures: missing Python `jinja2`, an unavailable model download in the
libcurl-free build, a BGE tokenizer fixture mismatch, and a ChatGLM4 trailing
newline expectation mismatch. None executes the scheduler, KV-restore, or CPU
flash-attention path changed here.
