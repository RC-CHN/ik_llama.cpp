#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${1:-} == "--server-child" ]]; then
    pid_file=${2:?missing pid file}
    shift 2
    printf '%s\n' "$$" > "$pid_file"
    exec "$@"
fi

result_dir=${1:?usage: run-hybrid-growth-benchmark.sh RESULT_DIR}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

workload=${BENCH_WORKLOAD:-growth}
model=${BENCH_MODEL:-weights/Qwen3.8-27B-Q4_K_M.gguf}
corpus=${BENCH_CORPUS:-/mnt/optane-tmp-stage/ik-hybrid-256k.txt}
port=${BENCH_PORT:-18088}
ctx_size=${BENCH_CTX_SIZE:-409600}
parallel=${BENCH_PARALLEL:-2}
sessions=${BENCH_SESSIONS:-2}
draft_ctx=${BENCH_DRAFT_CTX:-204800}
batch_size=${BENCH_BATCH_SIZE:-512}
ubatch_size=${BENCH_UBATCH_SIZE:-256}
hot_tokens_requested=${BENCH_HOT_TOKENS:-auto}
hot_tokens=$hot_tokens_requested
hybrid_block=${BENCH_HYBRID_BLOCK:-256}
capacity_profile=${BENCH_CAPACITY_PROFILE:-}
gpu_headroom_fraction=${BENCH_GPU_HEADROOM_FRACTION:-0.0625}
throughput_target=${BENCH_THROUGHPUT_TARGET:-}
mtp_max=${BENCH_MTP_MAX:-4}
request_mtp_max=${BENCH_REQUEST_MTP_MAX:-}
mtp_adaptive=${BENCH_MTP_ADAPTIVE:-1}
mtp_prompt_chunk=${BENCH_MTP_PROMPT_CHUNK:-0}
gpu_layers=${BENCH_GPU_LAYERS:-99}
milestones=${BENCH_MILESTONES:-20000,40000,60000,80000,100000,120000,140000,160000,180000,200000}
decode_tokens=${BENCH_DECODE_TOKENS:-128}
final_decode_tokens=${BENCH_FINAL_DECODE_TOKENS:-2048}
sticky_slots=${BENCH_STICKY_SLOTS:-1}
cache_ram=${BENCH_CACHE_RAM:-0}
cache_disk=${BENCH_CACHE_DISK:-0}
cache_dir=${BENCH_CACHE_DIR:-}
slot_state_dir=${BENCH_SLOT_STATE_DIR:-}
slot_state_action=${BENCH_SLOT_STATE_ACTION:-none}
cache_type_k=${BENCH_CACHE_TYPE_K:-q8_0}
cache_type_v=${BENCH_CACHE_TYPE_V:-q8_0}
cache_type_k_draft=${BENCH_CACHE_TYPE_K_DRAFT:-q8_0}
cache_type_v_draft=${BENCH_CACHE_TYPE_V_DRAFT:-q8_0}
memory_max=${BENCH_MEMORY_MAX:-46G}
threads=${BENCH_THREADS:-}
cpu_bind=${BENCH_CPU_BIND:-}
model_sha256=${BENCH_MODEL_SHA256:-7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169}
decode_sweep_context=${BENCH_SWEEP_CONTEXT:-40000}
decode_sweep_depths=${BENCH_SWEEP_DEPTHS:-0,2,3,4,5,6,8}
decode_sweep_modes=${BENCH_SWEEP_MODES:-concurrent,serial}
decode_sweep_warmup=${BENCH_SWEEP_WARMUP:-16}
decode_sweep_concurrency_levels=${BENCH_SWEEP_CONCURRENCY_LEVELS:-}
decode_sweep_prompt_template=${BENCH_SWEEP_PROMPT_TEMPLATE:-resident-sweep}
round_robin_context=${BENCH_ROUND_CONTEXT:-200000}
round_robin_decode_tokens=${BENCH_ROUND_DECODE_TOKENS:-2048}
round_robin_throughput_target=${BENCH_ROUND_THROUGHPUT_TARGET:-20}
numa_fa_wide_query_groups=${BENCH_NUMA_FA_WIDE_QUERY_GROUPS:-auto}
capacity_plan_json=
capacity_plan_sha256=

if [[ -z "$throughput_target" && "$workload" == round-robin-decode ]]; then
    throughput_target=$round_robin_throughput_target
fi

if [[ ! "$hybrid_block" =~ ^[0-9]+$ || ! "$ubatch_size" =~ ^[0-9]+$ ]] ||
        (( hybrid_block <= 0 || ubatch_size <= 0 )); then
    printf 'hybrid block and ubatch must be positive integers: block=%s ubatch=%s\n' \
        "$hybrid_block" "$ubatch_size" >&2
    exit 2
fi

if [[ "$hot_tokens" == auto ]]; then
    hot_tokens=$(( (ubatch_size + hybrid_block - 1) / hybrid_block * hybrid_block ))
    if [[ -n "$capacity_profile" ]]; then
        capacity_args=("$capacity_profile" --ubatch "$ubatch_size" \
            --block "$hybrid_block" --headroom-fraction "$gpu_headroom_fraction" \
            --max-tokens "$ctx_size")
        if [[ -n "$throughput_target" ]]; then
            capacity_args+=(--throughput-target "$throughput_target")
        fi
        capacity_plan_json=$(python3 scripts/derive-hybrid-capacity.py \
            "${capacity_args[@]}" --json)
        hot_tokens=$(python3 scripts/derive-hybrid-capacity.py "${capacity_args[@]}")
    fi
fi

if [[ -z "$threads" ]]; then
    mapfile -t physical_cpus < <(
        lscpu -p=CPU,CORE,SOCKET | awk -F, '
            $1 !~ /^#/ {
                key = $3 ":" $2
                if (!(key in seen)) {
                    seen[key] = 1
                    print $1
                }
            }'
    )
    threads=${#physical_cpus[@]}
    cpu_bind=$(IFS=,; printf '%s' "${physical_cpus[*]}")
elif [[ -z "$cpu_bind" ]]; then
    mapfile -t online_cpus < <(lscpu -p=CPU | awk -F, '$1 !~ /^#/ { print $1 }')
    if (( threads > ${#online_cpus[@]} )); then
        printf 'BENCH_THREADS=%s exceeds %s online CPUs\n' "$threads" "${#online_cpus[@]}" >&2
        exit 2
    fi
    selected_cpus=("${online_cpus[@]:0:threads}")
    cpu_bind=$(IFS=,; printf '%s' "${selected_cpus[*]}")
fi
if (( threads <= 0 )); then
    printf 'benchmark worker count must be positive, got: %s\n' "$threads" >&2
    exit 2
fi
if [[ ! "$hot_tokens" =~ ^[0-9]+$ ]]; then
    printf 'resolved hot ring must be a positive integer: hot=%s\n' "$hot_tokens" >&2
    exit 2
fi
if (( hybrid_block <= 0 || hot_tokens <= 0 || hot_tokens % hybrid_block != 0 ||
        hot_tokens < ubatch_size )); then
    printf 'resolved hot ring must be block-aligned and at least ubatch: hot=%s block=%s ubatch=%s\n' \
        "$hot_tokens" "$hybrid_block" "$ubatch_size" >&2
    exit 2
fi

if [[ ! -f "$model" ]]; then
    printf 'model not found: %s\n' "$model" >&2
    exit 2
fi
if [[ ! -f "$corpus" ]]; then
    printf 'corpus not found: %s\n' "$corpus" >&2
    exit 2
fi
if [[ "$sticky_slots" != 0 && "$sticky_slots" != 1 ]]; then
    printf 'BENCH_STICKY_SLOTS must be 0 or 1, got: %s\n' "$sticky_slots" >&2
    exit 2
fi
if [[ "$mtp_adaptive" != 0 && "$mtp_adaptive" != 1 ]]; then
    printf 'BENCH_MTP_ADAPTIVE must be 0 or 1, got: %s\n' "$mtp_adaptive" >&2
    exit 2
fi
if [[ "$numa_fa_wide_query_groups" != auto &&
        "$numa_fa_wide_query_groups" != 0 && "$numa_fa_wide_query_groups" != 1 ]]; then
    printf 'BENCH_NUMA_FA_WIDE_QUERY_GROUPS must be auto, 0, or 1, got: %s\n' \
        "$numa_fa_wide_query_groups" >&2
    exit 2
fi
if [[ "$slot_state_action" != none && "$slot_state_action" != save &&
        "$slot_state_action" != restore && "$slot_state_action" != restore-save &&
        "$slot_state_action" != bootstrap ]]; then
    printf 'BENCH_SLOT_STATE_ACTION must be none, save, restore, restore-save, or bootstrap, got: %s\n' \
        "$slot_state_action" >&2
    exit 2
fi
if [[ "$slot_state_action" != none && -z "$slot_state_dir" ]]; then
    printf 'BENCH_SLOT_STATE_DIR is required when BENCH_SLOT_STATE_ACTION=%s\n' \
        "$slot_state_action" >&2
    exit 2
fi
if [[ "$sticky_slots" == 1 && "$sessions" -gt "$parallel" ]]; then
    printf 'sticky sessions (%s) cannot exceed physical slots (%s)\n' "$sessions" "$parallel" >&2
    exit 2
fi
if [[ "$slot_state_action" != none && "$slot_state_action" != bootstrap &&
        "$sticky_slots" != 1 ]]; then
    printf 'slot state save/restore requires BENCH_STICKY_SLOTS=1\n' >&2
    exit 2
fi
if [[ "$slot_state_action" == bootstrap && "$sticky_slots" != 0 ]]; then
    printf 'slot state bootstrap requires BENCH_STICKY_SLOTS=0\n' >&2
    exit 2
fi

mkdir -p "$result_dir"
if [[ -n "$(find "$result_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'result directory is not empty: %s\n' "$result_dir" >&2
    exit 2
fi
if [[ -n "$capacity_plan_json" ]]; then
    printf '%s\n' "$capacity_plan_json" > "$result_dir/capacity-plan.json"
    capacity_plan_sha256=$(sha256sum "$result_dir/capacity-plan.json" | cut -d' ' -f1)
fi

# The benchmark frequently evaluates an uncommitted optimization.  Capture the
# exact source delta and executable so runs sharing the same Git revision do
# not get mistaken for the same build later.
git status --short > "$result_dir/git-status.txt"
git diff --binary > "$result_dir/source.diff"
source_diff_sha256=$(sha256sum "$result_dir/source.diff" | cut -d' ' -f1)
server_binary_sha256=$(sha256sum build-cuda/bin/llama-server | cut -d' ' -f1)
runner_sha256=$(sha256sum scripts/run-hybrid-growth-benchmark.sh | cut -d' ' -f1)
growth_driver_sha256=$(sha256sum scripts/bench-hybrid-growth.py | cut -d' ' -f1)
decode_sweep_driver_sha256=$(sha256sum scripts/bench-hybrid-decode-sweep.py | cut -d' ' -f1)
round_robin_driver_sha256=$(sha256sum scripts/bench-hybrid-round-robin-decode.py | cut -d' ' -f1)
capacity_driver_sha256=$(sha256sum scripts/derive-hybrid-capacity.py | cut -d' ' -f1)
summarizer_sha256=$(sha256sum scripts/summarize-hybrid-growth.py | cut -d' ' -f1)

pid_file=$result_dir/server.pid
server_log=$result_dir/server.log
script_path=$(readlink -f "${BASH_SOURCE[0]}")
server_pid=
scope_pid=
monitor_pids=()

cleanup() {
    for monitor_pid in "${monitor_pids[@]}"; do
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill -TERM "$monitor_pid" 2>/dev/null || true
        fi
        wait "$monitor_pid" 2>/dev/null || true
    done
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    if [[ -n "$scope_pid" ]]; then
        wait "$scope_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

server_env=(GGML_NUMA_ROW_SHARD=1 GGML_NUMA_FA_BATCH_QUERIES=1)
if [[ "$numa_fa_wide_query_groups" != auto ]]; then
    server_env+=("GGML_NUMA_FA_WIDE_QUERY_GROUPS=$numa_fa_wide_query_groups")
fi

server_cmd=(
    numactl --physcpubind="$cpu_bind" --interleave=0-3
    env "${server_env[@]}"
    build-cuda/bin/llama-server
    --model "$model" --host 127.0.0.1 --port "$port"
    --ctx-size "$ctx_size" --batch-size "$batch_size" --ubatch-size "$ubatch_size"
    --parallel "$parallel" --ctx-size-draft "$draft_ctx"
    --threads "$threads" --numa distribute --n-gpu-layers "$gpu_layers" --flash-attn on
    --hybrid-kv --hybrid-kv-hot "$hot_tokens" --hybrid-kv-block "$hybrid_block"
    --cache-type-k "$cache_type_k" --cache-type-v "$cache_type_v"
    --cache-type-k-draft "$cache_type_k_draft" --cache-type-v-draft "$cache_type_v_draft"
    --cache-ram "$cache_ram" --cache-disk "$cache_disk"
    --ctx-checkpoints 2 --ctx-checkpoints-tolerance 5 --ctx-checkpoints-interval 0
    --spec-ckpt-mode per-step --spec-type "mtp:n_max=$mtp_max,p_min=0.0"
    --prompt-fair-share --mtp-prompt-piggyback --mtp-prompt-chunk "$mtp_prompt_chunk"
    --temp 0 --seed 1 --ignore-eos
    --logit-bias 248044-inf --logit-bias 248046-inf --logit-bias 248063-inf
    --logit-bias 248064-inf --logit-bias 248065-inf --metrics
)
if [[ "$mtp_adaptive" == 1 ]]; then
    server_cmd+=(--mtp-adaptive-scheduling)
fi
if [[ -n "$cache_dir" ]]; then
    mkdir -p "$cache_dir"
    server_cmd+=(--cache-disk-path "$cache_dir")
fi
if [[ "$slot_state_action" != none ]]; then
    mkdir -p "$slot_state_dir"
    if [[ "$slot_state_action" == save ]] &&
            [[ -n "$(find "$slot_state_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'slot state save directory is not empty: %s\n' "$slot_state_dir" >&2
        exit 2
    fi
    if [[ "$slot_state_action" == restore || "$slot_state_action" == restore-save ||
            "$slot_state_action" == bootstrap ]]; then
        for id_slot in $(seq 0 $((parallel - 1))); do
            if [[ ! -f "$slot_state_dir/slot-$id_slot.bin" ]]; then
                printf 'slot state file not found: %s\n' "$slot_state_dir/slot-$id_slot.bin" >&2
                exit 2
            fi
        done
    fi
    server_cmd+=(--slot-save-path "$slot_state_dir")
fi

systemd-run --user --quiet --scope -p "MemoryMax=$memory_max" -p MemorySwapMax=0 -- \
    bash "$script_path" --server-child "$pid_file" "${server_cmd[@]}" \
    > "$server_log" 2>&1 &
scope_pid=$!

for _ in $(seq 1 180); do
    if [[ -s "$pid_file" ]]; then
        server_pid=$(<"$pid_file")
        if kill -0 "$server_pid" 2>/dev/null &&
                curl --silent --fail --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
            break
        fi
    fi
    if ! kill -0 "$scope_pid" 2>/dev/null; then
        printf 'server exited during startup; see %s\n' "$server_log" >&2
        exit 1
    fi
    sleep 1
done
if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
    printf 'server did not become ready; see %s\n' "$server_log" >&2
    exit 1
fi

if [[ "$slot_state_action" == restore || "$slot_state_action" == restore-save ||
        "$slot_state_action" == bootstrap ]]; then
    for id_slot in $(seq 0 $((parallel - 1))); do
        printf '{"time":"%s","action":"restore","id_slot":%s,"response":' \
            "$(date --iso-8601=seconds)" "$id_slot" >> "$result_dir/slot-state.jsonl"
        curl --silent --show-error --fail-with-body --max-time 600 \
            -H 'Content-Type: application/json' \
            -d "{\"filename\":\"slot-$id_slot.bin\"}" \
            "http://127.0.0.1:$port/slots/$id_slot?action=restore" \
            >> "$result_dir/slot-state.jsonl"
        printf '}\n' >> "$result_dir/slot-state.jsonl"
    done
fi

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'revision=%s\n' "$(git rev-parse HEAD)"
    printf 'source_diff_sha256=%s\n' "$source_diff_sha256"
    printf 'server_binary_sha256=%s\n' "$server_binary_sha256"
    printf 'runner_sha256=%s\ngrowth_driver_sha256=%s\ndecode_sweep_driver_sha256=%s\nround_robin_driver_sha256=%s\ncapacity_driver_sha256=%s\nsummarizer_sha256=%s\n' \
        "$runner_sha256" "$growth_driver_sha256" "$decode_sweep_driver_sha256" \
        "$round_robin_driver_sha256" "$capacity_driver_sha256" "$summarizer_sha256"
    printf 'server_pid=%s\n' "$server_pid"
    printf 'workload=%s\n' "$workload"
    printf 'milestones=%s\n' "$milestones"
    printf 'parallel=%s\nsessions=%s\nsticky_slots=%s\n' "$parallel" "$sessions" "$sticky_slots"
    printf 'ctx_size=%s\ndraft_ctx=%s\nbatch_size=%s\nubatch_size=%s\n' \
        "$ctx_size" "$draft_ctx" "$batch_size" "$ubatch_size"
    printf 'hot_tokens_requested=%s\nhot_tokens=%s\nhybrid_block=%s\n' \
        "$hot_tokens_requested" "$hot_tokens" "$hybrid_block"
    printf 'capacity_profile=%s\ncapacity_plan_sha256=%s\ngpu_headroom_fraction=%s\nthroughput_target=%s\nmemory_max=%s\n' \
        "$capacity_profile" "$capacity_plan_sha256" "$gpu_headroom_fraction" \
        "$throughput_target" "$memory_max"
    printf 'cache_ram_mib=%s\ncache_disk_mib=%s\ncache_dir=%s\n' "$cache_ram" "$cache_disk" "$cache_dir"
    printf 'slot_state_dir=%s\nslot_state_action=%s\n' "$slot_state_dir" "$slot_state_action"
    if [[ -n "$cache_dir" ]]; then
        df -B1 --output=size,avail,target "$cache_dir" | sed -n '2p' | \
            awk '{ printf "cache_filesystem_size_bytes=%s\ncache_filesystem_available_bytes=%s\ncache_filesystem_mount=%s\n", $1, $2, $3 }'
    fi
    printf 'cache_type_k=%s\ncache_type_v=%s\n' "$cache_type_k" "$cache_type_v"
    printf 'cache_type_k_draft=%s\ncache_type_v_draft=%s\n' "$cache_type_k_draft" "$cache_type_v_draft"
    printf 'mtp_max=%s\nrequest_mtp_max=%s\nmtp_adaptive=%s\nmtp_prompt_chunk=%s\n' \
        "$mtp_max" "$request_mtp_max" "$mtp_adaptive" "$mtp_prompt_chunk"
    printf 'gpu_layers=%s\n' "$gpu_layers"
    printf 'decode_tokens=%s\nfinal_decode_tokens=%s\n' "$decode_tokens" "$final_decode_tokens"
    printf 'model_sha256=%s\n' "$model_sha256"
    printf 'decode_sweep_context=%s\ndecode_sweep_depths=%s\ndecode_sweep_modes=%s\ndecode_sweep_warmup=%s\n' \
        "$decode_sweep_context" "$decode_sweep_depths" "$decode_sweep_modes" "$decode_sweep_warmup"
    printf 'decode_sweep_concurrency_levels=%s\n' "$decode_sweep_concurrency_levels"
    printf 'decode_sweep_prompt_template=%s\n' "$decode_sweep_prompt_template"
    printf 'round_robin_context=%s\nround_robin_decode_tokens=%s\nround_robin_throughput_target=%s\n' \
        "$round_robin_context" "$round_robin_decode_tokens" "$round_robin_throughput_target"
    printf 'corpus_sha256='; sha256sum "$corpus" | cut -d' ' -f1
    printf 'threads=%s\nnuma_physcpubind=%s\nnuma_interleave=0-3\n' "$threads" "$cpu_bind"
    printf 'GGML_NUMA_ROW_SHARD=1\nGGML_NUMA_FA_BATCH_QUERIES=1\n'
    printf 'GGML_NUMA_FA_WIDE_QUERY_GROUPS=%s\n' "$numa_fa_wide_query_groups"
    printf 'server_command='; printf '%q ' "${server_cmd[@]}"; printf '\n'
} > "$result_dir/manifest.txt"

(
    printf 'timestamp,memory_used_mib,memory_free_mib,gpu_util_pct,memory_util_pct,power_w,sm_clock_mhz,memory_clock_mhz\n'
    while kill -0 "$server_pid" 2>/dev/null; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits
        sleep 1
    done
) > "$result_dir/gpu.csv" & monitor_pids+=("$!")

nvidia-smi dmon -s t -d 1 -o DT > "$result_dir/pcie-dmon.txt" 2>&1 & monitor_pids+=("$!")

(
    printf 'timestamp,pid,rss_kib,vmsize_kib,threads,read_bytes,write_bytes,utime_ticks,stime_ticks,processor\n'
    while kill -0 "$server_pid" 2>/dev/null; do
        status_file=/proc/$server_pid/status
        io_file=/proc/$server_pid/io
        if [[ -r "$status_file" && -r "$io_file" ]]; then
            rss=$(awk '/^VmRSS:/ { print $2 }' "$status_file")
            vmsize=$(awk '/^VmSize:/ { print $2 }' "$status_file")
            threads=$(awk '/^Threads:/ { print $2 }' "$status_file")
            read_bytes=$(awk '/^read_bytes:/ { print $2 }' "$io_file")
            write_bytes=$(awk '/^write_bytes:/ { print $2 }' "$io_file")
            read -r utime stime processor < <(awk '{ print $14, $15, $39 }' "/proc/$server_pid/stat")
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
                "$server_pid" "${rss:-0}" "${vmsize:-0}" "${threads:-0}" \
                "${read_bytes:-0}" "${write_bytes:-0}" "${utime:-0}" "${stime:-0}" "${processor:-0}"
        fi
        sleep 1
    done
) > "$result_dir/process-io.csv" & monitor_pids+=("$!")

if [[ -n "$cache_dir" ]]; then
    (
        printf 'timestamp,bytes,filesystem_available_bytes\n'
        while kill -0 "$server_pid" 2>/dev/null; do
            cache_bytes=$(du -sb "$cache_dir" 2>/dev/null | awk '{ print $1 }')
            filesystem_available=$(df -B1 --output=avail "$cache_dir" 2>/dev/null | awk 'NR == 2 { print $1 }')
            printf '%s,%s,%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
                "${cache_bytes:-0}" "${filesystem_available:-0}"
            sleep 1
        done
    ) > "$result_dir/cache-disk.csv" & monitor_pids+=("$!")
fi

pidstat -h -u -r -d -w -p "$server_pid" 1 > "$result_dir/pidstat.txt" 2>&1 & monitor_pids+=("$!")

(
    while kill -0 "$server_pid" 2>/dev/null; do
        printf '# timestamp %s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
        curl --silent --max-time 2 "http://127.0.0.1:$port/metrics" || true
        printf '\n'
        sleep 2
    done
) > "$result_dir/metrics.prom" & monitor_pids+=("$!")

(
    while kill -0 "$server_pid" 2>/dev/null; do
        printf '===== %s =====\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
        numastat -p "$server_pid" || true
        sleep 30
    done
) > "$result_dir/numastat.txt" 2>&1 & monitor_pids+=("$!")

set +e
case "$workload" in
    growth)
        driver_cmd=(python3 -u scripts/bench-hybrid-growth.py \
            --url "http://127.0.0.1:$port" --corpus "$corpus" --result-dir "$result_dir" \
            --milestones "$milestones" --decode-tokens "$decode_tokens" \
            --final-decode-tokens "$final_decode_tokens" --sessions "$sessions")
        if [[ "$slot_state_action" == restore || "$slot_state_action" == restore-save ||
                "$slot_state_action" == bootstrap ]]; then
            required_initial_restores=$((sessions < parallel ? sessions : parallel))
            driver_cmd+=(--slot-state-dir "$slot_state_dir" \
                --required-initial-restores "$required_initial_restores")
        fi
        if [[ "$sticky_slots" == 1 ]]; then
            driver_cmd+=(--sticky-slots)
        fi
        if [[ -n "$request_mtp_max" ]]; then
            driver_cmd+=(--speculative-n-max "$request_mtp_max")
        fi
        ;;
    decode-sweep)
        if [[ "$sessions" -gt "$parallel" ]]; then
            printf 'decode-sweep requires one resident physical slot per session\n' >&2
            exit 2
        fi
        driver_cmd=(python3 -u scripts/bench-hybrid-decode-sweep.py \
            --url "http://127.0.0.1:$port" --corpus "$corpus" --result-dir "$result_dir" \
            --context-tokens "$decode_sweep_context" --decode-tokens "$decode_tokens" \
            --warmup-tokens "$decode_sweep_warmup" --depths "$decode_sweep_depths" \
            --modes "$decode_sweep_modes" --sessions "$sessions" \
            --prompt-template "$decode_sweep_prompt_template")
        if [[ -n "$decode_sweep_concurrency_levels" ]]; then
            driver_cmd+=(--concurrency-levels "$decode_sweep_concurrency_levels")
        fi
        ;;
    round-robin-decode)
        if [[ "$parallel" -ne 2 || "$sessions" -ne 3 ]]; then
            printf 'round-robin-decode requires BENCH_PARALLEL=2 and BENCH_SESSIONS=3\n' >&2
            exit 2
        fi
        if [[ "$slot_state_action" != bootstrap ]]; then
            printf 'round-robin-decode requires BENCH_SLOT_STATE_ACTION=bootstrap\n' >&2
            exit 2
        fi
        if [[ "$cache_disk" == 0 ]]; then
            printf 'round-robin-decode requires a non-zero BENCH_CACHE_DISK\n' >&2
            exit 2
        fi
        driver_cmd=(python3 -u scripts/bench-hybrid-round-robin-decode.py \
            --url "http://127.0.0.1:$port" --corpus "$corpus" --result-dir "$result_dir" \
            --slot-state-dir "$slot_state_dir" --context-tokens "$round_robin_context" \
            --decode-tokens "$round_robin_decode_tokens" --physical-slots "$parallel" \
            --sessions "$sessions" --throughput-target "$round_robin_throughput_target")
        if [[ -n "$request_mtp_max" ]]; then
            driver_cmd+=(--speculative-n-max "$request_mtp_max")
        fi
        ;;
    *)
        printf 'unknown BENCH_WORKLOAD: %s\n' "$workload" >&2
        exit 2
        ;;
esac
"${driver_cmd[@]}" > "$result_dir/driver.log" 2>&1
status=$?
set -e

if [[ ( "$slot_state_action" == save || "$slot_state_action" == restore-save ) &&
        "$status" == 0 ]]; then
    for id_slot in $(seq 0 $((parallel - 1))); do
        printf '{"time":"%s","action":"save","id_slot":%s,"response":' \
            "$(date --iso-8601=seconds)" "$id_slot" >> "$result_dir/slot-state.jsonl"
        if ! curl --silent --show-error --fail-with-body --max-time 600 \
                -H 'Content-Type: application/json' \
                -d "{\"filename\":\"slot-$id_slot.bin\"}" \
                "http://127.0.0.1:$port/slots/$id_slot?action=save" \
                >> "$result_dir/slot-state.jsonl"; then
            status=1
        fi
        printf '}\n' >> "$result_dir/slot-state.jsonl"
    done
fi

curl --silent --max-time 5 "http://127.0.0.1:$port/slots" > "$result_dir/final-slots.json" || true
python3 scripts/summarize-hybrid-growth.py "$result_dir" \
    > "$result_dir/summarizer.log" 2>&1 || true
printf 'finished_at=%s\nexit_status=%s\n' "$(date --iso-8601=seconds)" "$status" >> "$result_dir/manifest.txt"
exit "$status"
