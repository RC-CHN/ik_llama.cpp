# Xeon Max + RTX 3090 混合长上下文推理计划

## 目标与边界

目标模型是 `weights/Qwen3.8-27B-Q4_K_M.gguf`（15.92 GiB），目标机器为一张 24 GiB
RTX 3090 和一颗带四个 SNC/HBM NUMA 节点的 Xeon Max 9470C。首版只考虑
64K、128K 和 256K 上下文，重点是单会话代码库问答、代码补全和代码修改。

首版明确不做：

- 超过 256K 的上下文；
- Q8/Q4 冷 KV，冷 KV 统一使用 BF16；
- CPU FFN、复杂的层间流水或多用户连续批处理；
- 为不足 5% 的端到端收益长期维护复杂专用代码。

模型元数据实测为 24Q/4KV、K/V head dimension 256、65 个 block、每四层一个
全注意力层，以及 1 个 MTP layer。主干的 16 个全注意力层每 token 需要 64 KiB
BF16 KV，256K 时为 16 GiB；MTP context 另需 1 GiB。默认把最近 64K 的约
4 GiB 主干 KV 和 0.25 GiB MTP KV 留在 GPU，把 256K 时其余约 12.75 GiB
放进 Xeon Max HBM。

当前通用 allocator 还会在 target context 中为未由主干 graph 执行的 MTP layer
多分配一份 KV（256K 时约 1 GiB）；混合路径必须消除这份浪费。未经该修正时，
全量 target + MTP KV 的实际分配约为 18 GiB，而不是计划路径的 17 GiB。

## 总体思路

模型权重和主解码图尽量完整地留在 GPU。每个全注意力层把逻辑 KV 缓存
分成两个连续区间：

- GPU 热区：最近约 64K token，由 CUDA Flash Attention 处理；
- HBM 冷区：更早的 token，由 CPU BF16 批量 attention kernel 处理。

GPU 产生当前层的 query 后，GPU 热区 attention 和 CPU 冷区 attention 并行。
两边不直接返回已经归一化的最终结果，而是分别返回 softmax 的局部最大值、
指数和及加权 value。GPU 使用稳定的 log-sum-exp 公式合并两组统计量，再继续
执行该层的输出投影和 FFN。稳态解码不能在 PCIe 上来回搬运冷 KV；PCIe 只传
query、很小的归并结果，以及从热区一次性淘汰的 KV 块。

冷 KV 按块分配，并物理分散到四个 HBM NUMA 节点。每个工作线程只扫描本节点
的 KV 分片，最后归并四份局部结果，避免对同一批页面产生跨 NUMA 远端读取。
仍然使用物理核，不使用 SMT。

## MTP 路径

可用时始终启用模型自带的 MTP，普通单 token 解码只作为正确性 A/B 和故障
兜底。MTP 一轮产生若干候选后，主模型以 `m = k + 1` 的小批量验证候选。
CPU 冷区 kernel 在一次 KV 扫描中同时处理这些 query，从而同时摊薄 HBM 读取、
GPU 权重读取和 kernel/同步开销。首轮优先优化 `m = 2..6`，同时保留 `m = 1`
实现用于对照。

MTP 参数不能用普通聊天提示词调优。最终的 `n_max`、`p_min` 和 head 数量以
下面的真实代码库语料为准，并记录每轮草稿数、接受数、平均提交 token 数和
拒绝分布。

## 实现路径

### 第一版支持范围

先增加显式的实验开关和热窗口/块大小参数，默认不开启。开关启用时严格检查：

- 架构为当前 Qwen 3.5 dense hybrid 模型，KV 类型为 BF16/F16；
- CUDA Flash Attention 可用，只有一个 sequence/slot，按位置单调追加；
- 不允许 context shift、任意区间 `seq_cp` 或 defrag；首版只支持 clear、尾部删除
  和 MTP 尾部回滚；
- 实际可用显存不足安全线时直接报错，不能静默退回全 CPU 或申请完整 GPU KV。

这些限制先把复杂度收敛到当前真实用途。热/冷 KV 正确稳定后，再决定是否把接口
推广到其他 GQA dense 模型和多 slot server。

### KV 数据结构与放置

`llama_kv_cache::cells` 继续描述完整的逻辑 256K 位置，物理存储另加一层映射：

- `hot_begin/hot_end` 和 GPU ring write cursor；
- 每层 GPU K/V ring tensor，物理容量只按热窗口分配；
- 每层、每 NUMA 节点一组冷 K/V block descriptor；
- 每个 block 的逻辑起点、有效 token 数、所属节点、迁移状态和完成 event；
- 全局 `committed_pos` 与 `provisional_pos`，区分正式上下文和 MTP 候选尾部。

冷块初始按 256 或 512 token 对齐。块布局以全注意力层和 KV head 为外层、token tile
和 head dimension 为连续内层，使一个节点能顺序读取自己的 K/V，而不跨节点抓取
cache line。四个节点按完整 token block 轮转放置；query 很小，可以复制到每个节点，
各节点返回自己的局部 softmax 统计量。

HBM 使用预分配的大块匿名内存并通过 `mbind`/libnuma 固定到目标节点，推理热路径
不允许 `malloc`。启动后用 `/proc/<pid>/numa_maps` 或 `move_pages` 抽查实际页位置，
避免“线程绑核但页面仍在其他节点”的假本地化。

### Attention 算子接口

现有 Flash Attention 只返回已经归一化的输出，热区和冷区无法直接相加。源码中的
CUDA FA 和 CPU IQK FA 实际都已经在内部维护在线 softmax 的三元组，因此内部
stats 形式直接保留每个 query/head 的：

```text
M = max(score)
S = sum(exp(score - M))
R = sum(exp(score - M) * V)
```

热区得到 `(R_hot, M_hot, S_hot)`，冷区得到 `(R_cold, M_cold, S_cold)`，随后在
GPU 上计算：

```text
M  = max(M_hot, M_cold)
wh = exp(M_hot  - M)
wc = exp(M_cold - M)
S  = wh * S_hot + wc * S_cold
O  = (wh * R_hot + wc * R_cold) / S
```

这样只需从 CPU 回传每个 query/head 的 FP32 value 分子和两个标量，不传 attention
score；与“归一化输出 + LSE”相比只多一个 FP32 标量，却省掉两端的局部归一化和
随后反向加权。GGML 层面优先增加一个返回 `D_v + 2` 个 FP32 值的内部 Flash
Attention 变体和一个 merge op；CUDA 侧复用 `ggml/src/ggml-cuda/fattn-*` kernel
已有的 numerator/max/sum，CPU 侧复用 `ggml/src/iqk/iqk_flash_attn.cpp` 已经存在但
尚未向 GGML 暴露的 `M`/`S` 输出。

CPU 冷 kernel 必须融合 QK、mask、在线 softmax 和 PV，不能物化
`[context, query_head, m]` score 矩阵。先把 IQK 已有的 FP32 累加、AVX-512 BF16
路径改为保留三元组，再比较 AMX 小矩阵版本。关键形状是 24 个 query head、4 个 KV head以及
`m = 2..6`；同一 K/V tile 要在六个 GQA query head和全部 `m` 行之间复用。

### 每个全注意力层的执行时序

```text
GPU Q/K/V projection
  ├─ GPU: 写入 provisional hot KV，计算 hot FA stats
  └─ PCIe D2H: query -> pinned staging -> 四个 NUMA 冷 attention worker
                                      └─ 四节点局部 stats 归并
CPU stats --PCIe H2D--> GPU stats merge -> output projection -> FFN
```

首先让 GGML backend scheduler 表达这两个独立分支并检查是否真的并行；如果 profiler
显示 graph split 被串行执行，再增加专用协调器：CUDA stream/event 启动热分支，固定
HBM worker pool 同时运行冷分支，merge 节点只等待两个 completion event。每层分别
记录 query copy、hot FA、cold FA、merge 和等待时间，不能只看总 TG 猜测是否重叠。

64K 以内没有冷区，直接沿用原 CUDA FA 快路径。超过 64K 后，只在完整且已经提交的
block 离开热窗口时进行一次 D2H 淘汰；复用 ring slot 前必须等待该 block 的复制 event。
prefill 同样边生成边溢出，不能先创建完整 256K GPU KV 再统一搬走。

### MTP 提交协议

候选 token 的 KV 只写入 GPU ring 的 provisional 尾部，并预留至少 `n_max + 1`
个不会触发冷淘汰的 slot。验证完成后：

- 接受前 `a` 个候选时，把对应尾部标记为 committed，并把 write cursor 移到新尾部；
- 第一个拒绝位置及其后的 KV 直接作废，cold block 不需要复制或回滚；
- DeltaNet recurrent state 继续使用已有 per-step checkpoint，恢复到第一个拒绝位置；
- 只有完整 committed block 跨过热窗口边界后，才允许进入 HBM 迁移队列。

这使冷 KV 始终是只追加、不可变的数据，MTP 回滚不会在四个 NUMA 节点上产生随机写。

### 预计代码落点

- `src/llama-context.h`：热/冷 KV descriptor、迁移队列和提交位置；
- `src/llama.cpp`：分配、clear/尾删、prefill 淘汰、checkpoint 生命周期和统计；
- `src/llama-build-context.cpp`：在 `llm_build_kv()`/`llm_build_kqv()` 接入双分支；
- `src/graphs/build_qwen35.cpp`：只处理 Qwen 3.5 hybrid/MTP 的能力检查，通用逻辑
  不复制到模型 graph builder；
- `ggml/include/ggml.h`、`ggml/src/ggml.c`：内部 stats/merge op；
- `ggml/src/iqk/iqk_flash_attn.cpp` 与 CUDA `fattn-*`：CPU 冷 kernel 和 GPU 热 stats。

### 分阶段落地

1. **建立基线**：恢复可用的 CUDA 构建，测量纯 GPU 的短上下文和可容纳的最长
   上下文，记录无 MTP、MTP、VRAM、HBM、PCIe 和接受率基线。
2. **拆分 KV 存储**：实现 GPU 热环形缓存、HBM 冷块存储和块级异步淘汰；prefill
   时直接逐块溢出，不能先在 GPU 上申请完整 256K KV。
3. **实现 CPU 冷 attention**：先完成 BF16、24Q/4KV 对应形状的正确版本，再做
   四 NUMA 节点本地分片和 `m = 2..6` 批量 kernel。
4. **实现并行与归并**：并行启动 CUDA 热区和 CPU 冷区计算，在 GPU 上完成稳定
   softmax 归并，并接入 CUDA event、固定缓冲区和无逐 token 分配的调度。
5. **接入 MTP 状态管理**：候选验证期间使用临时 KV 尾部，接受后一次提交，拒绝
   时正确回滚；同时验证 DeltaNet/MTP 等模型状态不会被错误提前提交。
6. **测量后再调布局**：根据 HBM PMU、GPU profiler 和 PCIe 计数器决定块大小、
   线程数及热窗口是否需要在 32K--64K 内调整。只保留有明确端到端收益的改动。

## 代码库测试集

基准语料固定取自一个明确 git revision 的 `ik_llama.cpp` 源码树。后续生成一份
文件清单和内容哈希，过滤构建产物、权重、日志和重复生成文件。64K、128K、256K
使用嵌套前缀，长档是在短档基础上加入更多真实文件，不用重复文本填满长度。

每个长度至少包含三类任务：

1. **代码补全**：在真实函数中间截断并继续生成，代表 MTP 较容易命中的场景；
2. **跨文件问答/修改**：给出实现、头文件和调用点，要求解释或生成补丁；
3. **低重复代码生成**：要求设计一个仓库内尚不存在的小功能，防止只靠复制已有
   代码得到虚高的 MTP 接受率。

主性能测试使用 greedy decoding 以保证可复现，每题生成至少 256 token；另以
`temperature = 0.2` 的常用代码设置做非门槛复测。每档至少五次暖运行，报告中位数、
均值和离散度。TG、首次 prefill/TTFT、前缀缓存复用必须分开记录，不能用缓存后的
数字冒充首次加载体验。

## 验收标准

### 正确性与稳定性

- 64K、128K、256K 都能完成真实代码 prompt 的 prefill 和至少 256 token 生成；
- 在 `MemoryMax=46G`、禁用 swap 的保护下无 OOM、无非法访问，RTX 3090 峰值显存
  不超过 23 GiB；
- 分区 attention 的聚焦测试使用 FP32 参考，暂定 `atol <= 2e-3`、
  `rtol <= 2e-3`；
- 端到端 teacher-forced 对照的平均 KL divergence 不超过 `1e-3`，top-1 一致率
  不低于 99.5%；MTP 与同一混合 kernel 的 greedy 非投机路径应产生相同 token；
- 256K 档连续生成至少 4096 token，无 KV 错位、回滚错误或 RSS/VRAM 持续增长。

### 性能

以下均为单会话、代码库测试集、生成阶段 TG，不包含首次 prefill：

| 上下文 | 预期范围 | 首版通过线 |
| --- | ---: | ---: |
| 64K | 60--80 tok/s | >= 60 tok/s |
| 128K | 45--62 tok/s | >= 48 tok/s |
| 256K | 34--46 tok/s | >= 35 tok/s |

通过线按三类代码任务合并后的中位数判断，同时任一任务类别的中位数不得低于总通过线
的 80%，避免只挑高重复补全样本。MTP 的总体平均提交量目标为每次主模型验证至少
2 token，并且生产配置必须比同布局的无 MTP 路径快至少 50%。

无 MTP 不是最终性能目标，但应保留下列兜底量级作为诊断参考：64K 为 30--38
tok/s、128K 为 24--30 tok/s、256K 为 17--22 tok/s。若 MTP 主路径没有通过，
报告应分别指出是候选接受率不足、批量验证 kernel 不足，还是热/冷 attention 调度
未能重叠，不能只给一个总 TG 数字。

## 结果记录

每次正式验收记录代码 revision、完整命令、模型哈希、CUDA/驱动版本、MTP 参数、
prompt 清单哈希、TG/PP/TTFT、平均接受 token、VRAM/RSS、HBM 各 NUMA 节点流量和
PCIe 流量。最终结果追加到本文件或单独的同目录 benchmark 记录中，失败和回退实验
也保留简短结论，避免以后重复尝试。

## 2026-08-16 首版实现与实测

本节记录的是首个可运行版本，不回写上面的设计目标。结论先说：冷热 KV、raw
softmax stats、稳定归并、四路 HBM 定位和 MTP/GQA 查询融合都已工作，正确性门槛
通过；但 CPU/GPU attention 分支尚未实现真正的层内异步重叠，三档 TG 仍低于上面
最初设定的通过线。因此这是“功能路径和关键冷算子成立”，不是完整性能目标达成。

### 版本与机器

- 源码基线：`9fd1d734a21b86c0b37e8511fe3c85d96c07ef4b`；最终混合推理实现提交：
  `9ebe1917`（`feat: add Xeon Max RTX 3090 hybrid KV path`）。
- CPU：Intel Xeon CPU Max 9470C，1 socket，52 physical cores/104 threads，4 个
  HBM NUMA node；正式命令只绑定 CPU 0--51，不使用 SMT。
- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；驱动 595.84。
- 工具链：CUDA 12.4 (`nvcc 12.4.131`)，Release，`GGML_CUDA=ON`、
  `GGML_NATIVE=ON`。运行时确认 AVX-512 BF16、VNNI、AMX BF16/INT8 可用。
- 保护：所有正式推理都在 `MemoryMax=46G`、`MemorySwapMax=0` 的 user scope 中；
  外层使用 `numactl --physcpubind=0-51 --interleave=0-3`。
- 模型：`weights/Qwen3.8-27B-Q4_K_M.gguf`，17,106,775,008 bytes；SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`。

代码库 prompt 是嵌套前缀，哈希如下：

| 档位 | 文件 | CLI prompt-eval token | SHA-256 |
| --- | --- | ---: | --- |
| 64K | `/tmp/ik-hybrid-64k.txt` | 64,294 | `133fd02972f4b3a3dfff31146c6b755f51f21fb1ee4d4df89cddae7243e97404` |
| 128K | `/tmp/ik-hybrid-128k.txt` | 130,277 | `2a1c7b6f2063e88e9150b095344dddd997703a5ea97e76ff3c66c0b72fab5a01` |
| 256K | `/tmp/ik-hybrid-256k.txt` | 251,244 | `d870ca1e387374936a3d96a11042ceae4eb576200b0a1c907965d8b28aa82a51` |

### 实际落地形态

首版与最初“只把淘汰块异步搬到 HBM”的设计略有不同：

- 主 context 的 16 个 full-attention layer 预分配完整 BF16 host-pinned K/V；最近
  `hot` 个 token 另写入 GPU F16 ring。DeltaNet recurrent state 和独立 MTP context
  继续完全驻留 GPU；target context 不再为仅由 MTP context 执行的 tail layer 重复
  分配 KV。
- `n_tokens > 16` 的首次 prefill 直接使用完整 host BF16 cache 的标准 FA 路径；
  `n_tokens <= 16` 的 decode/MTP 才拆成 CPU cold raw stats 与 CUDA hot raw stats，
  最后在 CUDA 上按 log-sum-exp 合并。这避免了首版先实现复杂异步淘汰状态机。
- 新增内部 FA stats 形式 `[R(Dv), M, S]`。CUDA vector FA 和 IQK CPU FA 都能返回
  未归一化 online-softmax 状态；普通 FA API 和默认行为保持不变。
- 启用 `GGML_NUMA_ROW_SHARD=1` 且 CLI 使用 `--numa distribute` 时，CUDA host
  buffer 会先解除 pin，按固定 cold-capacity 的四个连续区间 `mbind` 到 node 0--3，
  再重新注册。52 个物理线程按 `ith % 4` 映射节点，每节点 13 个 worker 只读本地页。
- 对 Qwen 24Q/4KV/GQA6 的 2--5 token MTP batch，把每个 KV head 的 12--30 条
  query 打包；52 个 worker 分片扫描 K/V 一次，每两个 KV head 做一次 barrier/归并。
  每线程使用约 64 KiB 栈暂存，热路径无 heap allocation。
- 查询融合在严格目标 shape 下默认启用；`GGML_NUMA_FA_BATCH_QUERIES=0` 可在同一
  二进制关闭，供 A/B 和精度对照。整个混合路径仍由 `--hybrid-kv` 显式开启，默认关闭。
- 为避免把实验路径误用于尚未实现的状态迁移，初始化时要求 CUDA 构建、恰好一张可见
  CUDA 设备和完整模型 offload，并拒绝自动 defrag。主 hybrid cache 禁用 context shift、
  prompt/state save/restore、任意区间 KV 删除/复制及 position add/div；MTP 独立 context
  没有 hybrid hot ring，因此它需要的临时尾部回滚仍保持可用。

当前尚未完成的原计划项：块级 D2H 淘汰队列、CPU/GPU 冷热 attention 真正并行、
多 slot/context shift/任意 KV 编辑，以及自动显存 fit。首版只支持单序列、Qwen3.5
dense、F16/BF16 cache 和当前追加/尾部回滚用途。

### 正式命令

三个档位共用以下命令；表中的 `FILE/CTX/BATCH/HOT/N` 分别替换：

```bash
systemd-run --user --quiet --scope -p MemoryMax=46G -p MemorySwapMax=0 \
numactl --physcpubind=0-51 --interleave=0-3 \
env GGML_NUMA_ROW_SHARD=1 GGML_NUMA_FA_BATCH_QUERIES=1 \
build-cuda/bin/llama-cli \
  -m weights/Qwen3.8-27B-Q4_K_M.gguf -f FILE \
  -c CTX -b BATCH -ub BATCH -n N -t 52 --numa distribute \
  -ngl 99 -fa on --hybrid-kv --hybrid-kv-hot HOT --hybrid-kv-block 256 \
  --spec-type mtp:n_max=4,p_min=0.0 --temp 0 --seed 1 \
  --ignore-eos \
  --logit-bias 248044-inf --logit-bias 248046-inf --logit-bias 248063-inf \
  --logit-bias 248064-inf --logit-bias 248065-inf --no-display-prompt
```

| 档位 | `CTX` | `BATCH` | `HOT` | `N` |
| --- | ---: | ---: | ---: | ---: |
| 64K cold-path stress | 65,536 | 128 | 512 | 1,024 |
| 64K production candidate | 65,536 | 128 | 49,152 | 1,024 |
| 128K production candidate | 131,072 | 64 | 49,152 | 256 |
| 256K rejected performance edge | 262,144 | 64 | 32,768 | 256 |
| 256K compliant/recommended | 262,144 | 64 | 24,576 | 256/4,096 |

64K 的 `HOT=512` 是为了让冷 kernel 承担几乎全部上下文；48K hot 是可留出 16K
冷区的生产候选。实测后者没有更快，因此不再继续做不足 5% 的热窗扫描。256K 的
32K 热窗可运行，但显存越过硬线，因此推荐值固定为 24K。

### 端到端结果

下面都是首次 prefill，不使用 prompt cache。`前进/轮` 是输出 token 数除以主模型
验证调用数；“接受 token”取 CLI 的 MTP `#acc tokens/#gen tokens`。

| 档位 | PP tok/s | TG tok/s | MTP calls | 前进/轮 | 接受 token | host KV | 最高轮询 VRAM | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64K, query fusion off | 518.38 | 20.61 | 257 | 3.98 | 766/1,028 | 4.00 GiB | 未记录 | 同二进制 A/B 基线 |
| 64K, query fusion on | 520.34 | 30.75 | 260 | 3.94 | 762/1,040 | 4.00 GiB | 未记录 | `+49.2%` TG |
| 64K, hot 48K production | 519.09 | 29.31 | 248 | 4.13 | 775/992 | 4.00 GiB | 22,621 MiB (22.09 GiB) | 跑满 1,024 token；不优于 cold stress |
| 128K, fusion 前 | 217.51 | 10.20 | 未记录 | 未记录 | 83 accepted | 8.00 GiB | 约 21,256 MiB | 历史同布局基线 |
| 128K, fusion后 | 218.02 | 16.14 | 77 | 3.32 | 177/308 | 8.00 GiB | 23,126 MiB | `+58.2%` TG，正常代码输出 |
| 256K, hot 32K | 129.19 | 10.92 | 67 | 3.82 | 187/268 | 16.00 GiB | 至少 23,802 MiB | 可运行，但超过 23 GiB 线 |
| 256K, hot 24K | 129.41 | 11.29 | 67 | 3.82 | 187/268 | 16.00 GiB | 23,270 MiB (22.72 GiB) | 推荐；严格通过显存线 |
| 256K, hot 24K, 4096 token | 129.33 | 12.16 | 973 | 4.21 | 3,121/3,892 | 16.00 GiB | 23,316 MiB (22.77 GiB) | 连续稳定性通过 |

关键对照：

- 64K、512-hot、128-token 早期 NUMA A/B：未精确绑页时 PP/TG 为
  `520.39/11.51`，四路精确 HBM 后为 `515.71/18.16`，TG `+57.8%`。
- 查询融合的短一致性 A/B 使用 3,129-token `common/ngram-cache.cpp`、生成 64 token：
  off 为 `64.34 tok/s`，on 为 `67.01 tok/s`，可见输出和 MTP 统计完全相同；短冷区
  只有 `+4.2%`，但 64K 正式冷区达到 `+49.2%`，所以保留该实现。
- 短上下文纯 GPU 诊断为：无 MTP `43.23 tok/s`，MTP `71.91 tok/s`。这只能说明
  模型自带 MTP 在代码场景有效，不能冒充长上下文混合结果。
- 64K/48K-hot 生产候选完整跑满 1,024 token，只用 248 次主模型验证，MTP 接受
  `775/992`；但 TG 为 `29.31 tok/s`，略低于 512-hot 压力配置的 `30.75 tok/s`。
  扩大 CUDA hot 分支没有抵消双分支和同步成本，按 5% 停止线不再继续扫 32K/40K。
- 256K/32K 与 256K/24K 两次输出前缀和接受 token 总数一致；24K PP 不降、TG
  差异落在运行波动内，却固定释放约 512 MiB，因此停止继续扫热窗。
- 256K/24K 连续生成 4,096 token 用时 336.74 秒。生成期一分钟间隔观测的 RSS
  `17,999,640 -> 17,999,772 KiB`，VRAM 始终为 23,316 MiB；没有随 token 增长的
  常驻内存、KV 错位或回滚错误。代码延续越长，MTP 接受率升到 3,121/3,892，
  所以 TG 高于 256-token 短样本。

### 数值正确性

使用 `llama-perplexity` 在同一二进制中先以
`GGML_NUMA_FA_BATCH_QUERIES=0` 保存 teacher-forced 基线，再打开融合读取同一份
log-probability 文件。参数为 `ctx=1536, batch=ubatch=5, hot=512, chunks=1`，
语料为 `common/ngram-cache.cpp`，实际比较 767 个位置：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| Mean KLD | `0.000140 ± 0.000019` | `<= 0.001` |
| 99% KLD | `0.001358` | 记录项 |
| Maximum KLD | `0.008545` | 记录项 |
| RMS probability delta | `0.382 ± 0.027%` | 记录项 |
| Same top-1 | `99.870 ± 0.130%` | `>= 99.5%` |
| PPL ratio | `1.000977 ± 0.000708` | 记录项 |

门槛通过。长 greedy 输出在几百 token 后可能因归并顺序的浮点差异走向不同格式，
但 64-token A/B 完全一致，完整分布 KL/Top-1 也支持这是数值顺序漂移，不是 mask、
KV 索引或回滚错误。

最终源码分别在 CUDA 和纯 CPU/AMX 配置下于 `MemoryMax=24G`、无 swap 的 scope 中
成功构建。两套 `test-amx-bf16`/`test-amx-int8`、CUDA 的
`test-model-load-cancel`/`test-autorelease` 均通过；最终 CUDA 二进制也完成 27B 模型
加载、四节点 cold KV 绑定、双 context 初始化和 MTP 接受/回滚短回归。全量 CTest
中的 tokenizer/chat fixture、Python Jinja 与 eval-callback 失败分别来自缺少测试数据、
未安装 `jinja2`，以及本地没有 `stories260K.gguf` 且该构建未启用 libcurl；没有把这些
环境缺件误记为本次实现通过。

### 性能计数器与瓶颈判断

在查询融合前的 64K 生成阶段，四个 HBM PMU 的稳定合计约为 `82.6 GB/s` read、
`0.85 GB/s` write。52 个物理核约 `2.16 GHz/core`、IPC `0.516`，LLC miss 比例约
`20.4%`。同机 `likwid-bench copy_avx512` 报约 `573 GB/s`，对应 raw PMU 约
`410 GB/s`。因此当时不是物理 HBM 已饱和，而是同一 GQA K/V 被不同 query/MTP
行反复读取、指令与归并开销也很高。把 12--30 query 融合到一次扫描后，端到端
64K/128K TG 分别提高约 49%/58%，与该判断一致。

在 256K/24K 的 4,096-token 稳态生成窗口重新同步采样 30.03 秒，得到 HBM read
`169.5 GB/s`、write `0.63 GB/s`，约为 raw copy 上限的 41%。同期
`nvidia-smi dmon -s t` 的 29 个有效样本平均 PCIe RX/TX 约
`822/154 MB/s`，远低于 PCIe 4.0 x16 上限。融合确实提高了有效 HBM 利用率，但
当前仍不是物理 HBM 或 PCIe 饱和；低 GPU duty cycle、CPU 指令/归并成本和冷热
分支串行仍是主要限制。

256K/24K 的活动进程用 `numastat -p` 抽查，四节点总驻留分别为
`4396.79 / 4300.16 / 4354.03 / 4443.95 MiB`，最大差约 144 MiB（约 3.3%）；
总 private residency 为 17,251.46 MiB。这验证了 16 GiB host KV 的页位置确实近似
四等分落在四个 HBM node，而不只是线程 affinity 生效。

### 失败、回退与停止条件

- 128K 使用 64K hot ring 时，batch 128 和 batch 64 都触发 CUDA VMM OOM；改为
  48K hot、batch 64 后稳定运行。
- 查询融合最初把所有线程 partial stats 放进 GGML shared work buffer，需要约
  6.6 MiB，而调度器只规划约 4.5 MiB，触发 `malloc(): invalid size`。改为每线程
  约 64 KiB 栈 scratch，shared buffer 只保留 packed Q 与 52 个指针后通过长测。
- 256K/32K hot 最高轮询至少 23,802 MiB，超过 23 GiB 自定硬线；24K hot 在相同
  时间点稳定少约 510--524 MiB，最终 256-token 轮最高观测 23,270 MiB。
- 独立 AMX QK 包装和更复杂、不能一次复用完整 K/V 的 MTP/GQA 变体没有得到可靠
  `>=5%` 端到端收益，均已回退，不留维护负担。
- 当前生产候选 TG `29.31/16.14/11.29--12.16` 仍明显低于原计划 `60/48/35 tok/s` 通过线
  （64K cold-path stress 的最好值为 `30.75 tok/s`）。
  下一项只有在能实现 CPU cold 与 CUDA hot/主图真正重叠、或进一步消除整段 cold
  扫描时才值得继续；小于 5% 的布局微调按约定停止。

## CPU cold / CUDA hot 异步 executor 实验与回退（2026-08-16）

这一轮按独立 goal 实现并验证了层内异步 executor，保留条件为固定代码库 prompt 上
128K TG 至少提高 15%、256K TG 至少提高 10%；任何复杂实现达不到门槛都必须回退。

### 原型实现

- 把每个 full-attention layer 显式拆成 CPU cold raw online-softmax stats、CUDA hot
  stats、cold-ready H2D 和 CUDA log-sum-exp merge；四个 NUMA 节点分别返回 partial
  stats，由 GPU 归并，避免 CPU 节点间再读一遍 partial。
- scheduler 中加入一个持久 coordinator thread，以 condition variable 提交 cold
  split；主线程先启动 CUDA hot split，再以 CUDA event 和目标 backend stream 等待
  cold-ready copy，最后执行 merge。
- 加入逐层 query-copy、cold、hot、stats-copy、merge 计时。`ctx=4096/hot=512` 的
  代表性单层时间约为 query copy 0.011 ms、cold 0.366 ms、hot 约 0.95 ms（首轮
  1.367 ms）、stats copy 0.024 ms、merge 0.399 ms。
- 初版异步 H2D 使用 `cudaStreamPerThread`，它只保证 host-complete，不能与 graph
  allocator 所在主 stream 排序；结果是 hot 临时存储仍在使用时被覆盖，64K 会生成
  重复 `!`。原型改成目标 CUDA backend stream 后，serial/async 输出与 MTP 接受统计
  完全一致。该竞态修复随 executor 一并回退，因为原串行 scheduler 不需要这条特殊
  copy 路径。
- 另做过 52-thread 持久 CPU worker pool。睡眠/CV 版本在 4K 短测只有
  22.87 tok/s，50 ms busy-spin 版本只有 13.01 tok/s，而原 OpenMP 路径为
  60.45 tok/s。占满 52 个物理核会饿死 CUDA launch/host-side 调度，因此整个 pool
  原型已删除。

KTransformers `eb9b70c` 也被用于实现对照。其 `CPUInfer` 通过
`cudaLaunchHostFunc` 把 CPU task submit/sync callback 插入 CUDA stream，并由固定
`TaskQueue`/`WorkerPool` 执行；新版 MoE 还能把 deferred experts 跨层流水。但旧
`dynamic_attention` 的 decode attention 是 CPU task 提交后立刻在同一 stream sync，
没有可直接复用的 dense hot/cold 同层并行归并。dense attention 的下一层严格依赖本层
输出，也不能像 MoE deferred expert 那样安全跨层拖延，所以没有照搬该框架。

### 严格 F16 A/B

以下都用同一原型二进制、相同 prompt、相同 greedy/MTP 参数生成 256 token；serial
只设置 `GGML_HYBRID_ATTN_ASYNC=0`，其余完全相同。

| 档位 | hot | serial PP/TG | async PP/TG | TG 增益 | MTP（两侧相同） | RSS/VRAM |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 64K | 49,152 F16 | 520.91 / 27.74 | 522.09 / 30.33 | +9.34% | 55/65 drafts，189/260 tokens | RSS 15.55 GiB；async 采样 20,794 MiB |
| 128K | 49,152 F16 | 218.42 / 17.86 | 217.92 / 22.91 | +28.28% | 59/69 drafts，185/276 tokens | RSS 15.54 GiB；async 采样 21,196 MiB |
| 256K | 24,576 F16 | 129.40 / 10.92 | 129.70 / 11.64 | +6.59% | 59/69 drafts，185/276 tokens | RSS 17.13 GiB；历史生成峰值 23,270 MiB |

128K 超过保留线，但 256K 只提高 6.59%，低于 10%。24K hot 的可隐藏 CUDA 时间约
1.31 ms/layer，已经接近该布局的理论重叠上限；继续增加 scheduler 代码不能凭空扩大
hot 分支。按 goal 的硬停止条件，coordinator、partial stats、显式 split、GPU shard
merge、profile 和特殊 H2D copy 共约 640 行未提交源码全部回退。

数值方面，原型中 CPU shard merge 对 GPU shard merge 的 Mean KLD 为
`0.000207 ± 0.000069`、top-1 `99.870%`、PPL ratio `0.999260`；同一 GPU merge 下
serial 对 async 的 Mean KLD 近似零、最大约 `8.6e-5`、top-1 `100%`。因此回退原因
纯粹是性能门槛，不是精度失败。

### Q8 hot KV 救援实验

CUDA FA 已有 head-size 256 的 `Q8_0/Q8_0` 专用实例，CUDA copy 也支持
F32 -> Q8_0，所以曾用一个 8 行 cache-type 改动尝试在相同显存中把 256K hot 从
24K 扩到 48K。结果如下：

- 带复杂 executor 的 4K/hot512 短测为 60.66 tok/s；Mean KLD `0.000145`、top-1
  `99.739%`，精度通过。
- 按 256K 冷热比例构造的 64K/hot12288 A/B：serial 30.27、async 30.89 tok/s，
  只有 `+2.05%`。Q8 把 hot kernel 缩短后，反而更没有足够工作可隐藏。
- 回退 executor 后，精简 Q8 4K 短测为 69.33 tok/s；最终精简图相对 F16 logits 的
  Mean KLD `0.000196 ± 0.000054`、top-1 `99.739%`，仍通过。
- 64K/hot49152 Q8 为 PP 514.44、TG 29.29 tok/s，MTP 55/64 drafts、190/256
  tokens；进程显存采样 19,334 MiB。它和原精简 F16/hot49152 的历史 29.31 tok/s
  持平，只明确节省约 3 GiB 显存。
- 最终 256K/hot49152 Q8 为 PP 128.96、TG 10.85 tok/s，MTP 59/67 drafts、
  187/268 tokens；RSS 约 17.10 GiB，prefill 最高采样显存 22,188 MiB，输出是正常
  `common/ngram-map.h` 代码。它比现有 F16/hot24576 的 11.29 tok/s 低约 3.9%。

因此 Q8 hot 的确能省显存，但在本机 3090 + Xeon Max 的 256K 目标布局上没有带来
速度收益；按“不为低于 5% 的收益增加路径”的约定，其 cache-type 改动也回退。

### 最终保留项

最终源码不包含异步 executor、额外 worker pool 或 Q8 hybrid cache 接口，只保留
CUDA unary `ABS` 的 compute dispatch 与 supports-op 两个 case（共 4 行）。原有
hybrid softmax merge 图本来就使用 `ggml_abs`；补齐 CUDA backend 支持可以避免它
成为不必要的 CPU fallback，同时也是通用且低维护成本的算子完整性修复。

### 最终构建与回归

- CUDA 和 AMX 两套目标均在 `MemoryMax=24G`、`MemorySwapMax=0` 保护下完成构建。
- `test-amx-bf16` 与 `test-amx-int8` 的全部用例通过。
- 用实际 `Qwen3.8-27B-Q4_K_M.gguf` 运行 CUDA `test-model-load-cancel` 和
  `test-autorelease`，两者均以退出码 0 完成；后者完整装载 66/66 层后正常释放。
- 最终 F16 混合 KV 短回归使用 4K context、512-token hot ring、固定代码库输入和
  64-token greedy/MTP 输出：PP `929.87 tok/s`、TG `68.39 tok/s`，MTP 接受
  `47/64` 个生成 token，输出为正常的 `common_ngram_cache_print` C++ 代码片段。
- 最终 `git diff --check` 通过；源码搜索确认没有遗留 executor/profile/partial-stats
  开关。两份 Xeon Max 本机实验文档继续由 `.git/info/exclude` 排除，不进入仓库。

## 异步路径恢复与干净 perf 定位（2026-08-16）

上面的“回退”记录是当时按 256K 必须达到 10% 的原始 goal 做出的结论。随后重新按
实际使用权重和上下文权衡维护成本：64K/128K/256K 的严格 A/B 分别为
`+9.34% / +28.28% / +6.59%`，其中 128K 是主要目标且收益显著，因此最终恢复并
保留了 coordinator、四路 NUMA partial stats 和 GPU merge。默认只在 cold KV 至少
8,192 token 时启用，`GGML_HYBRID_ATTN_ASYNC=0` 可作为串行对照；4K 强制异步回归
与串行输出完全一致。

恢复后的 128K 实际代码 prompt 含 130,277 token，运行得到 PP `212.50 tok/s`、TG
`21.53 tok/s`，MTP 共提出 280 个 draft token、接受 185 个，输出为正常代码。该轮
使用 F16 hot KV、49,152-token hot ring、52 个物理核和四节点 cold KV 交错放置。

### 监测方法与被排除的污染数据

一次同时采集 192 个 HBM uncore 事件、CPU `perf record` 和 GPU 遥测的串行长测只有
`13.12 tok/s`。所有 HBM PMU 的 `cpumask` 都是 CPU 0，而精确绑核又把 OpenMP
主线程 `ith=0` 固定在 CPU 0；大量 multiplexed uncore 事件因此直接干扰关键调度
线程。51 个 worker 在该轮采样中表现出相同的 barrier 等待比例，也验证这不是正常
算子负载。该 `13.12 tok/s` 及其绝对时间分解标记为监测污染，不能用于性能 A/B。

干净定位改用同一代码库 prompt 的 28,647-token 前缀、32K context、8,192-token hot
ring、1,024-token MTP 生成，只做 15 秒 CPU 采样，不同时打开 HBM PMU。结果为 PP
`515.46 tok/s`、TG `48.79 tok/s`；896 个 draft token 中接受 799 个，输出正常。
采样主要分布如下：

- libgomp team idle 约 68%：串行 GPU hot 分支运行时 CPU team 没有工作，这是异步
  overlap 能获得收益的来源，不是需要优化的 CPU 指令；
- IQK 内部 barrier 约 7.6%，活动 flash-attention 约 21%；
- `accumulate_qkv` 归并算术 self 约 0.57%，跨 NUMA partial merge 不是热点。

把 hot ring 缩到 4,096、让 cold shard 更接近满载后，复测 PP/TG 为
`513.29/48.51 tok/s`。短样本的最后一个 NUMA shard 仍因 cold token 未装满而较早到达
barrier；128K 的约 81K cold KV 已接近 81,920-token 总 cold capacity，因此不能把
这个短测尾 shard 不平衡误当成 128K 的主要问题。

### BF16 flash-attention 单核 top-down

另写了一个只调用实际 `iqk_fa_256_256` 的微基准，形状为 BF16 K/V、`D=256`、
`nq=30`，依次测试 `nk=1536/1568/1600`，固定 CPU 0 和 HBM node 0。top-down 结果：

- IPC `1.905`，backend bound `57.0%`；
- core bound `55.4%`，memory bound 仅 `1.6%`；
- retiring `40.2%`、frontend bound `1.6%`、bad speculation `1.2%`。

因此活动 CPU attention kernel 本身已经是计算/执行端受限，并非 HBM 带宽受限。
采样符号中两个 `iqk_flash_helper_T` 变体合计约 `72.36%`，PV 的
`FlashQKV::accumulate_qkv` 约 `24.56%`。最热 helper 的反汇编落在连续
`vdpbf16ps` QK 点积和水平归约上；这才是 AMX 候选。PV 当前使用 FP32 概率权重与
AVX-512 FMA，若改成 AMX 需要先把概率降到 BF16，必须重新做 KLD/top-1 验收。

尝试把 cold chunk 边界优先对齐 64 token，以增加 64-step helper 命中率。单核微基准
归一化耗时只相差约 1--2% 且落在噪声内，端到端 TG 从 `48.51` 变为 `48.48 tok/s`，
没有收益，源码改动已回退。归并既只占不到 1%，其 log-sum-exp/标量缩放结构也不是
AMX tile 矩阵乘，继续改归并不符合 5% 停止条件。

下一项值得做的原型仅限 QK：在 hot KV 驱逐到 cold HBM 时原位形成 AMX B-tile 友好
布局，Q 临时打包成 BF16 A tile，避免额外维护约 4 GiB 的 K 镜像。先保持 PV 和
softmax 的 AVX-512 数值路径不变；只有固定 128K 代码 prompt 的端到端 TG 至少提高
5%，且 KLD/top-1 门槛继续通过，才保留该实现。相关原始采样暂存在
`/tmp/ik-clean-30k-tg-perf.data`、`/tmp/ik-clean-fullshard-30k-tg-perf.data` 和
`/tmp/iqk-fa-micro-perf.data`。

## AMX QK/PV 与 128K 验收结果（2026-08-16）

### 最终实现

实际 top-down 已证明 cold attention 是 core-bound 后，`D=256` 的 BF16 CPU flash
attention 增加了两段 AMX 路径：

- QK：Q 按 16/32 行临时打包为 BF16 A tile，K 直接按 canonical BF16 行读取，使用
  `tdpbf16ps` 生成 FP32 score；没有增加常驻 K 镜像。
- PV：softmax 概率从 FP32 转为 BF16 A tile，canonical BF16 V 以很小的 L1 scratch
  重排为 B tile，再用 `tdpbf16ps` 累加 FP32 输出。PV 开关在 query tile 外缓存，且
  每个 K block 只清理实际未使用的尾行。
- 单 token decode 和 2--5 token MTP verification 都使用六路 GQA query fusion，
  让每个 cold K/V shard 只扫描一次。

AMX 运行时通过 CPUID 和每线程 `arch_prctl` 请求 tile 权限；不支持 AMX 的机器仍走
原 AVX-512 路径。`GGML_AMX_FA_QK=0` 可关闭整条 AMX attention 路径，
`GGML_AMX_FA_PV=0` 可只关闭 PV，用于严格 A/B。

### 性能演进

固定 128K 代码 prompt（130,277 input token）、F16 hot KV 24,576、52 个物理核、
四 HBM NUMA 节点交错、greedy MTP 的结果如下：

| 版本 | PP tok/s | TG tok/s | 相对 QK-only |
| --- | ---: | ---: | ---: |
| AMX QK、AVX-512 PV | 207.95 | 23.01 | 基线 |
| AMX QK+PV，首轮 | 209.83 | 29.78 | +29.4% |
| AMX QK+PV，最终 512-token 验收 | 217.25 | **35.75** | **+55.4%** |

最终轮用时：prefill `599667.82 ms`，512-token generation `14321.84 ms`
（`27.97 ms/token`），总用时 `613989.66 ms`。MTP 共执行 117 次验证，提出 468 个
draft token、接受 393 个，平均每次主模型 forward 覆盖 `4.376` token。最终 TG 比
30 tok/s 目标高 19.2%；较长的 512-token 样本也让代码场景下的 MTP 稳态收益得到
充分体现。

最终验收命令：

```bash
systemd-run --user --quiet --scope -p MemoryMax=46G -p MemorySwapMax=0 \
  numactl --physcpubind=0-51 --interleave=0-3 \
  env GGML_NUMA_ROW_SHARD=1 GGML_NUMA_FA_BATCH_QUERIES=1 \
  build-cuda/bin/llama-cli \
  -m weights/Qwen3.8-27B-Q4_K_M.gguf -f /tmp/ik-hybrid-128k.txt \
  -c131072 -b64 -ub64 -n512 -t52 --numa distribute \
  -ngl99 -fa on --hybrid-kv --hybrid-kv-hot24576 --hybrid-kv-block256 \
  --spec-type mtp:n_max=4,p_min=0.0 --temp0 --seed1 --ignore-eos \
  --logit-bias248044-inf --logit-bias248046-inf --logit-bias248063-inf \
  --logit-bias248064-inf --logit-bias248065-inf --no-display-prompt
```

单核 kernel 微基准中，QK+PV 对比只启用 QK，在 `nk=1536/1568/1600` 上分别由约
`292/301/312 us` 降至最终约 `117/144/122 us`，约为额外 `2.0--2.6x`。最后的
tail-clear 小改动单独贡献约 5.4--7.5% kernel 加速，代码量很小，因此保留。

### 数值与稳定性

- 实际模型 logits A/B（QK-only 对 QK+PV）：Mean KLD
  `0.000110 +/- 0.000011`，RMS probability delta `0.355% +/- 0.030%`，same top-1
  `99.739% +/- 0.184%`，PPL ratio `1.000266 +/- 0.000591`；通过既定门槛。
- 直接 kernel 回归覆盖 `nq=12,nk=64` 和 `nq=30,nk=96` 以及 causal mask；相对
  AVX 路径的 relative RMS 分别为 `1.210%` 和 `1.184%`，统计量误差小于 `8e-6`。
- 一次 128K/768-token 探索暴露了 MTP 图复用工作区问题：2--5 token 使用约
  123 KiB 的融合图，但偶发单 token 回退到约 418 KiB 的通用六 GQA 路径，导致
  `M/S` 写越界。现在 1--5 token 全部进入同一融合路径；30K 无 MTP 单 token 生成、
  30K/1024-token MTP 和最终 128K/512-token MTP 均正常退出。
- row-shard + exact-pin 现在要求 generation/batch 线程数都是四个 NUMA shard 的
  整数倍；无效的 `-t 51` 会在初始化时明确报错，避免静默产生错误 shard 布局。

Q8_0 hot KV 接口最终作为可选的容量档保留，默认仍是 F16。30K 测试里把 hot ring
从 F16 8,192 扩至 Q8_0 16,384 后 TG 为 50.19 tok/s，对应 F16 为 52.91 tok/s；它
节省显存/扩大 hot 容量，但不作为速度优化宣传。

CUDA 和纯 AMX 两套完整构建均在 `MemoryMax=24G`、禁用 swap 的保护下完成；两套
`test-amx-bf16` 和 `test-amx-int8` 均通过，实际 27B 权重的
`test-model-load-cancel` 与 `test-autorelease` 也正常退出。系统
`kernel.numa_balancing` 的关闭 A/B 没有收益，已经恢复为 `1`。

## 原生双路 hybrid-MTP 并发（2026-08-16）

### 实现

这一轮不再用两套独立主模型 context 串行轮转，而是让 server 的多个 slot 进入同一个
target decode batch。放开范围刻意限制为 `--hybrid-kv` 加单一 MTP stage；复合投机链和
非 hybrid 模式继续沿用原来的单 slot 保护。

- server 将各 slot 的 `[root, draft...]` 对齐到共同 draft depth，并按连续 slot ID
  形成 sequence-major 矩形；不在最大连续组内的 slot 当轮降为 root-only。压缩 batch
  时会同步重映射所有 slot 的 target output index、draft index、cache token 和统计。
- hybrid decoder 能识别 `[seq0 x M][seq1 x M]...`，一次构建 `M x Nseq` recurrent
  graph。Qwen3.5 delta-net 的 conv/SSM 状态、Q/K/V stride、beta/gate 和 qnext state
  映射均扩展到连续多序列；任意非零起始 seq ID 也使用局部状态索引。
- per-step checkpoint 扩展为 `max_tokens x max_seqs`。SSM 保存布局是 step-major，conv
  保存布局是 sequence-major，restore 分别计算索引；每次恢复前重新选择共享 checkpoint
  storage，避免前一个 slot discard 后影响后一个 slot。
- server 在每轮 decode 开始显式注册需要 checkpoint 的 seq ID。这样新 prompt 的最后
  一个小 chunk 即使恰好有 2--5 个全输出行，也不会覆盖正在生成 slot 的 recurrent
  checkpoint。槽位释放后再复用、长短请求交错已经实际覆盖。
- CPU cold attention 现在可接受 1--16 个独立 query token，并拆成每组不超过 5 token。
  每个 token 的六个 GQA heads 复用自己的 mask row，多个 slot 仍只扫描一次 K/V shard；
  `nq=6/12/30` 和 5 个不同 mask row 都有直接 AMX kernel 回归。
- hybrid hot ring 的物理 cold/hot 边界在仍有活跃序列时保持高水位，旧 hole 写入只更新
  cold KV，不再错误覆盖 GPU ring 中仍然存活的 hot row；cache 全空时高水位重置。

矩形 per-step buffer 在本次 `-np 2`、`n_max=4` 配置下为 1208.25 MiB，覆盖 2 个 slot
和 root 加 4 个 draft step。它是共享缓冲，不是每个 slot 再复制一份。

### 最终吞吐验收

配置为 32K 总 context、`-np 2`（每 slot 16K）、F16 hot 8,192、52 个物理核、四 HBM
节点交错、RTX 3090 全层 offload、greedy MTP `n_max=4`。固定两条代码任务各输出 128
token，并发与串行都在同一个已热身 server 上运行：

| 模式 | 请求 A | 请求 B | 服务吞吐 |
| --- | ---: | ---: | ---: |
| 双路并发 | 69.28 tok/s | 69.75 tok/s | **138.56 tok/s** |
| 顺序执行 | 65.80 tok/s | 67.97 tok/s | **66.87 tok/s** |

并发服务吞吐按 `256 / max(T_A,T_B)` 计算，串行按 `256 / (T_A+T_B)` 计算；最终提升
为 **2.07x**。并发轮 A/B 的 generation 分别为 `1847.533/1835.061 ms`，MTP 分别
提出 `124/125` 个 draft、接受 `95/95` 个，合计接受率 76.3%。两路输出都是正常的
C++/Python 代码说明文本，没有 checkpoint、mask 或 recurrent-state 错误。

此前同一实现的独立 128-token 稳态轮为 121.80 tok/s，对应串行 70.70 tok/s（1.72x）；
最终热身后的固定轮进一步达到 138.56 tok/s。因代码生成的 MTP 接受率和图热身状态会
改变绝对值，保守容量规划仍应采用约 1.7x，2.07x 记录为最终固定工作负载实测。

### 冷路径与槽位复用验收

- 两条输入各 8,403 token，在每 slot hot 8,192 下都跨入 HBM cold path；两路均完成
  64-token 输出，server 明确记录 `hybrid CPU cold / GPU hot attention overlap enabled`。
  A/B generation 为 39.54/14.85 tok/s。差异来自 prompt scheduler 先让一个 slot 填满
  64-token prefill batch，先完成 prefill 的请求同时开始 generation；它是 TTFT/公平性
  取舍，不是计算空泡，因此没有为预计低于 5% 的公平调度改动增加代码。
- 槽位 churn：192-token 长请求与 16-token 短请求并行；短请求结束后立即在释放 slot
  启动新的 96-token 请求。三条请求均 HTTP 200、内容正常；generation 分别为
  54.36、33.05、57.36 tok/s，draft 接受分别为 `135/218`、`9/16`、`68/108`。
- continuous batching 下，即使关闭 MTP，相同 greedy 请求也可能因不同物理 KV row 和
  浮点归约路径而不与串行输出逐字相同；因此逐字差异不能归因于 MTP checkpoint。
  本轮以 HTTP 完成、状态/索引断言、正常语义输出和既有 KLD/top-1 门槛共同验收。

最终 `build-cuda` 全目标构建、`test-amx-bf16`、`test-amx-int8`、
`test-model-load-cancel`、`test-autorelease` 均在 `MemoryMax=24G`、禁用 swap 的保护下
通过，`git diff --check` 通过。实际 server 使用 `MemoryMax=46G`，没有再次触发 OOM。

## 四槽并发与过载排队验收（2026-08-16）

### 四个活跃槽位

把同一配置扩到 `-np 4` 后，每个 slot 获得 8,192 context；MTP per-step checkpoint
共享缓冲为 2,416.50 MiB（`max_tokens=5,max_seqs=4`）。四条 128-token 代码请求真正
同时活跃时均正常完成，各路 generation 为 30.20--39.36 tok/s，按最慢 generation
计算的聚合吞吐为 120.80 tok/s，端到端墙钟吞吐为 115.84 tok/s。它低于双槽的
138.56 tok/s，因此双槽仍是这台 3090 的吞吐甜点；四槽的价值是并发容量，不是提高
总吞吐。

### 槽位回收故障与修复

超过四路的突发请求最初暴露了两类只在 slot churn 下出现的 per-step checkpoint
布局问题：

- 图可为 `[seq0,seq1]` 构建后复用于 `[seq1,seq2]`。checkpoint payload 会按新输入
  写入，但 restore 的绝对 seq ID 仍来自建图时元数据。现在每次 `llama_set_inputs()`
  都从实际 batch 刷新 sequence-to-row 映射。
- 当一条请求只剩 root token，或释放槽位正在接入新 prompt 时，逻辑 batch 会被拆成
  多个 recurrent ubatch。后一个 verification ubatch 会覆盖前一个的共享 checkpoint
  描述，典型错误为 restore `seq1` 时只剩 `seq_ids=[2,3]`。解码器现在不会从后续 MTP
  rectangle 前面吞掉 root 行；server 还为纯 MTP hybrid 模式加入 prompt admission
  barrier：新 prompt 单独占一个调度轮，已有生成下一轮再与新槽组成 rectangle。

这里选择调度隔离而没有再分配一套 2.4 GiB scratch。短 prompt 只增加一个 admission
停顿；长 prompt/过载队列可以更慢，但不会让多个 ubatch 争用同一 checkpoint。restore
失败路径同时打印 seq ID、step 和容量，后续若再出现边界条件可直接定位。

### 过载压力测试

最终服务器仍固定四个活跃槽位，并由 HTTP 队列承接额外请求；主存保持
`MemoryMax=46G,MemorySwapMax=0`。先跑 8 路同时到达、每路 64 token，再连续三轮跑
16 路同时到达且把输出长度打散为 32/48/64/80 token：

| 突发 | 轮次 | 完成请求 | 生成 token | HTTP/restore 错误 | 墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 路 | 1 | 8/8 | 512/512 | 0 | 5.386 s |
| 16 路 | 1 | 16/16 | 896/896 | 0 | 9.989 s |
| 16 路 | 2 | 16/16 | 896/896 | 0 | 9.241 s |
| 16 路 | 3 | 16/16 | 896/896 | 0 | 9.572 s |
| 8 路（stale-layout 安全闩后） | 1 | 8/8 | 512/512 | 0 | 5.571 s |

合计 64/64 个请求、3,712/3,712 个生成 token、零 500、零 checkpoint restore 失败。
每轮结束 `/health` 都返回 `slots_idle=4,slots_processing=0`，服务继续存活；第三轮后
server 的 RTX 3090 显存约 20,806 MiB，没有随槽位回收增长或触发 OOM。结论是生产
配置保持 `-np 4`，额外并发排队；不使用 `-np 8` 去翻倍 checkpoint 显存。

## 首次长上下文 prefill 冲刺（2026-08-17）

### 目标与结论

这一轮固定使用同一份无 prompt cache 的代码库 prompt，目标是把 128K 首次 prefill
从 AMX QK+PV 验收时的 `217.25 tok/s` 提高到至少 650，并冲击约 700。最终当前 HEAD
`696e4cf5`（build 4846）得到 **697.30 tok/s**，相对基线为 **3.21x**；多轮相同
`ubatch=768` 结果为 `694.20 / 696.43 / 693.20 / 697.30 tok/s`，没有依赖一次偶然
高点。整个收益来自已有实现的批次生命周期和运行参数，不需要再增加 prefill 专用
CPU kernel、冷热分支或波前流水代码。

128K 最终命令如下：

```bash
systemd-run --user --quiet --scope -p MemoryMax=46G -p MemorySwapMax=0 \
  numactl --physcpubind=0-51 --interleave=0-3 \
  env GGML_NUMA_ROW_SHARD=1 GGML_NUMA_FA_BATCH_QUERIES=1 \
  build-cuda/bin/llama-cli \
  -m weights/Qwen3.8-27B-Q4_K_M.gguf -f /tmp/ik-hybrid-128k.txt \
  -c 131072 -b 768 -ub 768 -n 512 -t 52 --numa distribute \
  -ngl 99 -fa on --hybrid-kv --hybrid-kv-hot 24576 --hybrid-kv-block 256 \
  --spec-type mtp:n_max=8,p_min=0.0 --temp 0 --seed 1 --ignore-eos \
  --logit-bias 248044-inf --logit-bias 248046-inf --logit-bias 248063-inf \
  --logit-bias 248064-inf --logit-bias 248065-inf --no-display-prompt
```

### ubatch 筛选

64K 筛选只生成 1 token，prompt 为固定的 64,294-token 嵌套前缀，hot ring 为 24,576
token。默认 `attention-max-batch=256` 保持不变，因此全局 ubatch 扩大主要提高 GPU
dense/DeltaNet 图的工作粒度，attention 仍按不超过 256 token 的微块执行。

| ubatch | prompt time | PP tok/s | 相对前档 |
| ---: | ---: | ---: | ---: |
| 64 | 190,931.15 ms | 336.74 | - |
| 128 | 129,493.01 ms | 496.51 | +47.4% |
| 256 | 98,479.79 ms | 652.86 | +31.5% |
| 512 | 83,501.23 ms | 769.98 | +17.9% |
| 768 | 71,060.72 ms | **904.78** | +17.5% |
| 1,024 | 69,255.72 ms | 928.36 | +2.61% |

1,024 相对 768 已低于 5% 停止线，并进一步挤压 3090 显存，因此最终固定 768。
128K 的 `ubatch=512` 为 `223,909.21 ms / 581.83 tok/s`，768 的首次一-token 轮为
`187,664.55 ms / 694.20 tok/s`，提升 19.3%。当前 HEAD 的最终 512-token 轮为：

| 指标 | 结果 |
| --- | ---: |
| prompt tokens / time | 130,277 / 186,831.33 ms |
| PP | **697.30 tok/s** |
| generated tokens / time | 512 / 14,862.18 ms |
| TG | **34.45 tok/s** |
| MTP target calls | 88 |
| MTP drafted / accepted tokens | 704 / 422 |
| 总时间 | 201,693.51 ms |
| 最高总 FB memory | 23,678 MiB |
| 活动期 RSS 抽样 | 11,229,136 KiB |

最高 FB memory 由覆盖完整运行的 `nvidia-smi dmon -s m` 得到。运行时报告可用 VRAM
为 24,123 MiB，因此 MTP-8 仍有约 445 MiB 可用余量并正常退出，但它不是宽松的显存
配置；桌面额外占用明显增加时应退回 MTP-4 或缩短 hot ring。

### MTP 窗口与 TG

大 ubatch 改变 recurrent prefill 的浮点归约轨迹，最终生成的代码内容也会变化。最初
`MTP n_max=4` 的 128K 轮只有 `359/608` 个草稿 token 被接受，512 token 需要 152 次
target forward，TG 为 28.18 tok/s。每次 target forward 约 119.5 ms，反而略快于历史
35.75 tok/s 轮的约 122.4 ms；表面 TG 回退来自这条代码内容的接受率，而不是 decode
kernel 变慢。

先在严格相同的 64,294-token 原始 prompt 上筛选 MTP 窗口：

| 配置 | TG | target drafts | accepted tokens | 平均前进/轮 |
| --- | ---: | ---: | ---: | ---: |
| MTP-4 | 37.31 tok/s | 135 | 376/540 | 3.79 |
| MTP-8 | **42.35 tok/s** | 93 | 417/727 | 5.48 |

MTP-8 相对 MTP-4 提高 13.5%。其逐位置 accepted 数为
`[81,73,64,56,50,40,29,24]`，第八个草稿的条件接受率仍为 82.8%，所以扩窗不是在
盲目验证低价值尾部。把 `ngram-mod:n_max=8,n_min=2,ngram_size_n=8` 放在 MTP-4
前面的双阶段实验只有 35.09 tok/s，低于纯 MTP-4 的 37.31，已经淘汰。

最终 128K MTP-8 两轮分别为 `34.40/34.45 tok/s`，MTP 统计完全相同：88 次 target
forward、`422/704` accepted。相对同一大-ubatch 轨迹的 MTP-4 提高 22.3%，且只比
历史 35.75 tok/s 低 3.6%，落入正常内容/接受率波动范围。可见输出是连贯的
`stable_search.hpp` C++ 实现，没有 checkpoint、KV 或回滚错误。

### 硬件计数器与停止实现的理由

在 128K/ubatch-768 prefill 的 15.004 秒稳定窗口中，32 个 HBM PMU 聚合计数为：

- read CAS `2,596,735,774`，约 **11.08 GB/s**；
- write CAS `517,301,900`，约 **2.21 GB/s**；
- 合计约 **13.28 GB/s**，远低于本机 HBM raw copy 上限。

同期进程 task-clock 为 72.951 秒，即平均只占约 4.86 个 CPU core；IPC 为 1.203。
GPU 15 个样本平均 SM 利用率约 96.8%，多数样本为 98--100%，功耗约 338--351 W。
因此大 ubatch 已把首轮 prefill 从小批次/host 调度瓶颈推到 3090 dense 计算主导；HBM、
CPU cold attention 和 PCIe 都不是当前临界资源。继续实现 prefill CPU-cold/CUDA-hot
拆分、NUMA GQA kernel 或跨 microbatch 波前流水，不可能合理预期再带来超过 5% 的
端到端收益，按约定停止，不增加维护代码。

### 256K 泛化与显存边界

固定 251,244-token prompt 上，不能直接照搬 128K 配置：

- `ubatch=768, hot=24576, MTP-4` 的 main CUDA compute buffer 为 1,516.01 MiB，填充中
  `cuMemCreate` OOM，进程退出码 134；systemd scope 保护了系统，没有触发主机 OOM。
- `ubatch=512, hot=24576, MTP-4` 把 main/MTP compute buffer 降到
  `1,352.01/505.00 MiB`，但总显存仍增长到 23,780 MiB、只剩 344 MiB；按增长趋势会
  在完成前 OOM，因此主动终止，未拿它冒充成功结果。
- 最终 PP 泛化档使用 `ubatch=512, hot=8192, MTP-4`。大批 prefill 走完整 host BF16
  FA，hot ring 大小不参与该 attention 计算；缩短 hot 只为释放约 1 GiB VRAM，不能把
  这个配置的后续 TG 与既有 24K-hot TG 混为一谈。

最终 256K 安全档得到 `645,657.05 ms / 251,244 tokens = 389.13 tok/s`，相对原
24K-hot/ubatch-64 的 129.41 tok/s 为 **3.01x**。运行中最后一次遥测为总显存
23,036 MiB（余 1,088 MiB）和 RSS 19,202,680 KiB，随后正常退出。用于完整长上下文
生成时，24K hot 应搭配更小的 ubatch（已有 ubatch-64 稳定数据）；8K hot 仅是本轮
首次 prefill 的容量泛化档。

### 正确性、构建与工作树

本轮没有修改模型算术或源码，复用的 AMX/NUMA/CUDA kernel 已有 logits A/B：Mean KLD
`0.000110 +/- 0.000011`、same top-1 `99.739%`，通过既定门槛。MTP-8 仍由现有 1--16
token 路径按最多 5 token 分组处理；两次 128K 输出前缀和完整 MTP 统计一致。

当前 HEAD 重新完成全目标 CUDA 增量构建；`test-amx-bf16`、`test-amx-int8`、实际 27B
权重的 `test-model-load-cancel` 和 `test-autorelease` 均在内存保护 scope 中退出码 0。
本轮只形成运行参数和本地 benchmark 记录，没有值得提交的源码变化；按用户要求，本
文档继续由 `.git/info/exclude` 排除。工作树唯一 tracked 变化仍是用户自己的
`.gitignore` 中 `.rcoder/` 项，本轮没有触碰或暂存。

## `/tmp` 迁移到傲腾（2026-08-17）

迁移前 `/tmp` 是 `tmpfs`，容量 31.3 GiB，实占约 3.3 GiB RAM。傲腾设备
`/dev/nvme0n1p1`（型号 `ET000750KWJTF`）挂载于 `/workspace`，其上已有一个干净、
从未使用过的 64 GiB ext4 镜像：

`/workspace/.optane-tmp/tmp.ext4`，UUID `ffbc344d-6346-4ab2-9c1e-d5268659feed`。

镜像通过 `e2fsck -fn`，根目录设为 `root:root 1777`。由于 KDE、Edge、VS Code 等
进程正在持有原 `/tmp`，没有冒险热切换；先挂到 `/mnt/optane-tmp-stage`，并迁入本轮
927 个 `ik-*` 实验文件（3,259,758,481 bytes）。当前暂存文件系统可用约 59.5 GiB。

`/etc/fstab` 已将原 tmpfs 项替换为：

```fstab
/workspace/.optane-tmp/tmp.ext4 /tmp ext4 loop,nosuid,nodev,noatime,x-systemd.requires-mounts-for=/workspace 0 0
```

systemd 生成的 `tmp.mount` 正确包含 `RequiresMountsFor=/workspace` 和
`Before=local-fs.target`；宿主侧 `findmnt --verify` 为 0 parse errors / 0 errors，唯一
warning 是 loop 文件源属于普通文件，符合预期。配置将在下一次正常重启时生效；发行版
现有 `/usr/lib/tmpfiles.d/tmp.conf` 会按 10 天清理旧文件。原配置备份在
`/etc/fstab.pre-optane-tmp-20260817`，可用于回滚。

## HBM-only 分层 prompt/KV 缓存（2026-08-17）

### 目标与实现

本机当前处于 HBM-only 模式，四个 NUMA node 合计约 62 GiB；不存在容量型 DDR 慢层。
模型常驻页、4 GiB canonical BF16 KV、CPU scratch、RAM prompt cache 和 Linux 文件页
缓存都会竞争同一组 HBM。因此这里的“RAM 优先、空间不足后落盘”实现为显式两级预算，
而不是依赖 swap 或把 `/tmp` tmpfs 当作磁盘层：

- `--cache-ram N` 保存最新的、预算内可容纳的 displaced prompt state；
- `--cache-disk N --cache-disk-path PATH` 保存 RAM 放不下的旧状态，使用同目录
  `.partial` 后原子 rename；每个 server 使用权限 0700 的私有子目录，干净退出时删除；
- Linux 上大文件写完执行 `fdatasync`，读写完成后执行
  `POSIX_FADV_DONTNEED`，避免傲腾文件的 page cache 继续占用 HBM；
- hybrid KV 不在 prompt 结束后复制一份完整 HBM state。server 在 prompt 尾部前 5 token
  流式暂存一个自洽状态，只保留两个 checkpoint；被淘汰时若状态小于 RAM 预算则提升到
  RAM，否则直接收养傲腾文件；
- 傲腾状态加载后保留同一个不可变文件并把所有权挂回活动 slot，下一次 displacement
  直接重新收养，不再重复写出数 GiB；
- 磁盘预算按私有目录的实际文件大小计算，包含由活动 slot 租用、暂时不在 LRU 列表里的
  快照；多 slot 超出预算时先淘汰最老冷状态，无可淘汰项时拒绝新 staging，避免文件无界
  增长。slot 被另一状态覆盖时立即删除其旧 staged 文件；
- state blob 只保存 canonical HBM K/V。加载后同步重建 GPU hot ring，随后重置 scheduler；
  为保持逐 bit 一致，当前 hybrid hot-ring state I/O 要求 `-ctk bf16 -ctv bf16`。

混合缓存依赖 prompt-tail 磁盘 staging，因此只指定 RAM tier 会明确禁用 hybrid prompt
cache；指定磁盘 tier后，server 会把 checkpoint 数强制为 2、关闭周期 checkpoint，且在
用户把 tolerance 设为非正数时恢复为 5。对这台 HBM-only 主机，64K/27B 建议从
`--cache-ram 2048` 开始，避免默认 8 GiB RAM tier 吸收一份 4.1 GiB 状态。

恢复正确性调试还发现一个与缓存本身独立但会污染输出的 scheduler race：D=256 BF16
hot stats 原先不受 CUDA vector FA 支持，scheduler 把名为 `hybrid_kv_hot_stats` 的节点
留在 CPU cold split，却仍提前启动异步 cold worker，后续 GPU split 在 CPU 写完前读取
结果。修复包括：

1. CUDA vector FA 支持 D=256 BF16/BF16，先在 CUDA pool 中转为 FP16，再复用 FP16
   kernel；
2. 若 cold split 中仍发现 hot-stats 节点，则禁止异步 detach，安全退回串行；
3. cold stats 的 H2D async copy 后在 merge 前同步 backend，防止 gallocr 提前复用 host
   source；CPU cold 与已经提交的 CUDA hot 分支重叠不受影响。

### 64K 正式命令

重启前 `/tmp` 仍是 tmpfs，所以本轮明确使用已经挂载到傲腾镜像的
`/mnt/optane-tmp-stage`。server 命令为：

```bash
systemd-run --user --scope -p MemoryMax=18G -p MemorySwapMax=0 \
  numactl --physcpubind=0-51 --interleave=0-3 \
  env GGML_NUMA_ROW_SHARD=1 GGML_NUMA_FA_BATCH_QUERIES=1 \
  build-cuda/bin/llama-server \
  -m weights/Qwen3.8-27B-Q4_K_M.gguf -c 65536 -b 768 -ub 768 -t 52 \
  --numa distribute -ngl 99 -fa on \
  --hybrid-kv --hybrid-kv-hot 24576 --hybrid-kv-block 256 \
  -ctk bf16 -ctv bf16 --host 127.0.0.1 --port 18088 --parallel 1 \
  --cache-ram 2048 --cache-disk 12288 \
  --cache-disk-path /mnt/optane-tmp-stage/ik-prompt-cache-64k
```

测试脚本从当前 `examples/server/server-context.cpp` 重复语料中二分选择 64,351-token
代码前缀，执行 `A -> B -> A -> B -> A`；每次 greedy 生成 8 token。最终结果：

| 阶段 | prompt token / server prompt time | 速度 | disk load | 8-token generation |
| --- | ---: | ---: | ---: | ---: |
| A 首次 | 64,351 / 70,393.609 ms | 914.16 tok/s | - | 434.927 ms |
| A 恢复 1 | 5 / 90.666 ms | 55.15 tok/s | 1,507.05 ms | 382.310 ms |
| A 恢复 2 | 5 / 92.636 ms | 53.97 tok/s | 1,474.33 ms | 394.355 ms |

两次恢复内容都与 A 首次输出逐字相同。`prompt time` 不含 slot selection 阶段的 disk
load，因此用户可见恢复到首批输出约为 `1.5 s + 0.09 s + generation 首 token`。
初次尾部状态文件为 4,374,647,384 bytes（4,171.989 MiB）；两轮恢复后目录中仍只有
同一个文件。保留文件所有权前，恢复会额外把相同状态再写一次，导致尾部 5-token
prompt 阶段为 3,863.408 ms；零复制重新收养后降到约 91--93 ms。

底层驱动还把同一个 64K memory snapshot 连续恢复三次。每次状态 payload 为
4,374,389,988 bytes、snapshot token 为 64,346，GPU hot ring 重建 24,576 row /
1,536 MiB，用时 65.13--65.27 ms；抽查层的 K/V 在恢复后以及追加 5 个尾 token 后都为
0 mismatched token、hash 完全相同。

### HBM 与页缓存结果

最终 scope 的 `MemoryMax` 为 19,327,352,832 bytes（18 GiB），`MemoryPeak` 为
16,526,602,240 bytes（15.39 GiB），测试结束 `MemoryCurrent` 为 10,998,013,952 bytes
（10.24 GiB），swap current/peak 都为 0。结束时 `memory.stat` 为 anon 5.27 GiB、file
4.95 GiB，状态文件页已经不在 HBM。

作为对照，未调用 `POSIX_FADV_DONTNEED` 的双恢复轮把 scope 推到完整 18 GiB 上限，
其中 file cache 为 11.0 GiB；对 4.1 GiB 状态文件手工发出 DONTNEED 后，file cache
立刻减少约 4.4 GiB，`MemoryCurrent` 从约 17.6 GiB 降到 13.1 GiB。这验证了 HBM-only
模式下仅把路径放到傲腾并不够，必须主动处理 Linux page cache。

写后 `fdatasync` 使首次 staging 从约 2.0 s 增至 4.96 s，并让最终首次 PP 从约
951.95 降到 914.16 tok/s；这是一次性建缓存成本，仍高于既定 700 tok/s prefill 目标，
且换来无 HBM 假占用和完整落盘状态。傲腾回读约 1.47--1.51 s，连续恢复没有重复写。

### 构建与回归

- `build-cuda` 全目标增量构建在 `MemoryMax=18G, MemorySwapMax=0` 下通过；
- `test-amx-bf16`、`test-amx-int8`、`test-model-load-cancel`、`test-autorelease` 通过；
- 完整 CTest 为 21/25。四个失败与本改动无关：`test-jinja-py` 缺 Python `jinja2`，
  `test-eval-callback` 缺 `stories260K.gguf` 且构建无 libcurl，BERT tokenizer 本地 vocab
  与 golden 不匹配，`test-chat-template` 是已有 ChatGLM4 末尾换行差异；
- `git diff --check` 通过；真实 64K 双恢复、低层三恢复和 GPU hot-ring byte/hash
  检查全部通过。

提交前又以 Optane ccache 重建 `llama-server`，并单独重跑上述四项相关测试，全部通过。
这次只增加磁盘目录真实占用记账与 staged 文件生命周期清理，不触及已验证的 64K 数值或
算子路径。

下一次重启后 64 GiB 傲腾 ext4 镜像会直接挂为 `/tmp`，届时可以省略
`--cache-disk-path` 或显式指到 `/tmp/ik_llama_prompt_cache`。重启前不能使用默认路径，
否则仍会写进 HBM-backed tmpfs。

## 生产式持续增长、多路调度与容量回收（2026-08-18）

### 先统一 35 tok/s 的口径

此前的约 35 tok/s **不是短上下文结果**。可复核的正式记录是 130,277-token prompt、
512-token generation、24,576-token hot ring、`batch=ubatch=768`、MTP-8：PP
`697.30 tok/s`，TG `34.45 tok/s`。两次重复 TG 为 `34.40/34.45 tok/s`；MTP 共
88 次 target forward，提出 704 个 token、接受 422 个。当前代码环境重跑的同类控制轮
为 PP `686.24 tok/s`、512 token / 15.294 s = `33.48 tok/s`。另一个 128K 内容轨迹
得到 `16.14 tok/s`，其 MTP 接受率和生成内容不同，不能拿来证明 kernel 性能减半。

后面的生产测试区分三种速率，避免再次混淆：

- `request decode tok/s`：单请求自身 `predicted_ms` 的速率；
- `decode-union tok/s`：所有请求实际 decode 时间区间并集上的系统输出速率；
- `wall output tok/s`：把长 prefill、排队、磁盘恢复也算入整阶段墙钟，只表示混合业务产出，
  不能与上面的 128K 单请求 TG 直接相比。

### 当前调度实现

[`vLLM V1 scheduler`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
用统一 token budget 表示 prefill、decode 和 speculative token，
其[分块 prefill 策略](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md)
优先调度运行中的 decode，再用剩余 budget 分块接纳 prefill。这里采用同样的服务层原则，
但不能直接复制任意 token 排布：Qwen3.5 recurrent state 的 MTP 验证必须保持
`[seq0 root+draft...][seq1 root+draft...]` 等深矩形，否则共享 per-step checkpoint 会被
后一个 ubatch 覆盖。

本轮实现因此是受约束的 continuous batching：

- `--prompt-fair-share` 把一个 recurrent-safe physical ubatch 等分给所有待处理文本
  prompt，并在同一个 logical batch 内继续装入后续等深矩形；既消除一个长 prompt 独占，
  又保留大 `n_batch` 的 launch amortization。
- `--mtp-prompt-piggyback --mtp-prompt-chunk 512` 在新 prompt 进入时让已有生成只提交
  root row，target batch 完成后只抽取所属 sequence 更新其私有 MTP context；prompt
  结束后恢复普通等深 MTP-4。这样不破坏 checkpoint，又不再暂停 generation 十几秒。
- checkpoint 只在包含 prompt 最后一行的 physical slice 执行完后创建；若尾部 checkpoint
  尚未完成，logical batch 停止继续打包，避免提前 snapshot 尚未计算的 cache row。
- embedded MTP context 按物理 slot 使用自己的 context 容量，并真正遵守
  `--cache-type-k-draft/--cache-type-v-draft`；target-only 混合 batch 会先按 sequence 拆出
  私有 MTP 所属行。
- hybrid prompt cache 遇到 tokenizer join 改写或任意 divergent suffix 时先保存旧 slot；
  磁盘文件被 cache 收养后仍保留一个 data-only 本地 rewind checkpoint，避免活动 slot
  为几个边界 token 重新加载自己的数十 GiB 文件。

[`PagedAttention`](https://arxiv.org/abs/2309.06180) 的 block table、低碎片分配和前缀块共享仍然值得后续引入，但它主要解决
KV **容量/碎片**，不会自行消除这里的 recurrent checkpoint、私有 MTP context 和
CPU-cold/CUDA-hot 同步。当前先实现 token-budget/chunked-prefill 部分；只有需要在同一
target context 驻留更多不同前缀、且 block table 能同时覆盖 canonical HBM KV、GPU hot
ring 和 recurrent state 时，才值得做完整页式 KV 重构。

### 公平性与 prefill/decode 重叠 A/B

两条约 20K prompt 同时到达时，旧的 slot-first admission 与 fair-share 的结果为：

| 调度 | 阶段墙钟 | 聚合 prompt-window PP | 聚合 active-decode TG | prompt 完成偏差 |
| --- | ---: | ---: | ---: | ---: |
| slot-first | 59.159 s | 677.02 tok/s | 7.11 tok/s | 31.564 s |
| fair-share | 58.932 s | 679.63 tok/s | 42.65 tok/s | 0.671 s |

公平分块没有牺牲总 prefill，却把一条请求等待另一条完整 prefill 的 31.6 秒偏差消掉。

更贴近在线服务的 A/B 是 slot 0 连续生成 768 token，在第 32 个 stream event 后向 slot 1
注入 12,009-token prefill：

| 调度 | slot 0 TG | slot 1 PP | prefill 周围最大输出间隙 | slot 0 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| 原 prompt-only admission | 23.136 tok/s | 905.02 tok/s | 13.438 s | 33.337 s |
| root piggyback, chunk 512 | **23.348 tok/s** | 899.77 tok/s | **0.679 s** | **33.034 s** |

chunk 512 同时保持 PP/TG 和总墙钟，并把 generation 停顿从 13.4 秒降到 0.68 秒。更小的
256/128/64-token quantum 虽把最大间隙进一步压到 0.43/0.32/0.26 秒，却分别把 PP 降到
745.65/604.14/430.80 tok/s，TG 降到 21.74/20.27/17.36 tok/s；系统总吞吐目标下不选。

### 用 HBM 宽裕换取 GPU 容量

`-ngl 65` 让 65 个 repeating layers 留在 3090、约 1.19 GiB output layer 留在 CPU HBM。
此前 scheduler 仍会为每个 embedded MTP context 把约 1 GiB vocabulary matrix staging
进 CUDA compute buffer。本轮在 output weight 的 buffer type 为 host 时，把
`result_output` graph tensor 显式固定到 CPU backend：每个 MTP compute buffer 从
1,247 MiB 降到 139 MiB，两路合计释放 2,232 MiB VRAM。

| 两会话增长档 | 旧 staging prompt-window PP / decode TG | output 固定 HBM 后 | 峰值 VRAM |
| --- | ---: | ---: | ---: |
| 20K | 556.22 / 37.03 | **612.89 / 38.45** | 23,396 -> **21,164 MiB** |
| 40K | 336.54 / 23.35 | **356.15 / 23.72** | 23,396 -> **21,164 MiB** |

这不是为了节省 HBM/PCIe 带宽，而是主动使用它们的余量换 3090 容量。40K 的优化后 prefill
窗口 PCIe RX 约 9.18 GiB/s；decode 约 0.49 GiB/s，GPU memory-util 约 12.7%，进程平均
约 13.2 个 CPU core。历史 256K 稳态 HBM read/write 为 `169.5/0.63 GB/s`，而本机 raw
copy 约 `410 GB/s`；PCIe 同期仅约 `822/154 MB/s`。因此当前限制不是物理链路吃满。

本轮普通用户的 `perf_event_paranoid=4`，不能读取 32 个 `uncore_hbm_*` PMU；汇总文件
明确记录 `hbm_counters.available=false`，不会用 `nvidia-smi` 的 memory-util 伪装成 HBM
带宽。需要绝对 HBM 数值时应以 root 单独运行既有 LIKWID/uncore recipe，并避免同时把
全部 PMU multiplex 到负责 OpenMP `ith=0` 的 CPU 0，以免再次产生监测污染。

### MTP 深度与并发策略

同一 server 先把两条约 40K context 驻留，再以完全相同的 256-token 请求扫 MTP 深度；
下表为 512 个输出 token 除以实际 decode 时间并集：

| MTP depth | 两路并发 | 两路严格串行 |
| ---: | ---: | ---: |
| 0 | 14.63 | 5.44 |
| 1 | 18.98 | 13.67 |
| 2 | 16.00 | 14.74 |
| 3 | 17.91 | 16.70 |
| 4 | **21.20** | **17.94** |
| 5 | 16.53 | 13.34 |
| 6 | 18.21 | 13.93 |
| 8 | 17.85 | 13.99 |

结论不是“严格同时双路”，而是有多个 runnable decode 时组成 target MTP-4 矩形；没有
第二路时立即让单路继续，不为对齐而等待。严格串行更慢，因为 target verification 和
CPU-HBM output matmul 都失去跨 sequence batching。MTP-8 在历史单路 128K、长输出、
高接受率内容上能达到 34.45 tok/s，但在这个两路轨迹中额外验证使 PCIe/工作量上升且
尾部接受率下降；系统吞吐甜点是 MTP-4。

两个容量实验没有进入最佳吞吐配置：

- embedded MTP KV 改 Q4 每两路省约 400 MiB，但 40K decode 从 23.72 降到
  18.89 tok/s（-20.4%）；参数透传修复保留，生产仍选 Q8。
- 把 MTP cold KV/attention 移到 HBM 可再省约 1.60 GiB VRAM，但 20K/40K decode
  降到 31.69/22.32 tok/s；PCIe 仅约 0.35 GiB/s，损失来自每个私有 MTP context 的
  同步延迟而非带宽。该实验已回退；如果未来先把多个 slot 合成共享 MTP context，再考虑
  以它作为容量档。

所有 sweep 都保留逐请求开始/结束、prompt/decode interval、MTP 每深度接受数、GPU
util/memory/power、PCIe RX/TX 的平均/P50/P95/峰值、进程 CPU/RSS/I/O、NUMA residency、
磁盘目录大小、server 告警计数，以及 revision、source diff、二进制和 driver hash。

## 自适应生产调度与 200K 增长验收（2026-08-18）

### 不写死机器参数的调度器

前面的 MTP-4 和 prompt chunk 512 是某个 40K 轨迹的离线最优点，不应被当成
长期生产常量。新的 `--mtp-adaptive-scheduling` 直接以每次物理迭代实测的
服务吞吐作为反馈，候选集由当前运行时拓扑产生：

- decode 深度是模型/用户允许的 `[0, n_max]`，不存在“128K 就用 4”的表；
- decode cohort 宽度是 `[1, runnable_decode]`，每轮从实际可运行 slot 数重新确定；
- 混合 prefill 的 chunk 从 `n_batch` 开始递减，下界由 `n_ubatch`、并行数、
  当前 root 数和 KV 最大连续空洞共同约束；`--mtp-prompt-chunk 0` 表示运行时学习；
- workload key 包含活跃/常驻 decode 数、待处理 prompt 数和对数上下文层级。
  相邻 workload 只传递无量纲排名作为弱先验，每个继承 arm 仍必须本地实测。

深度和并发宽度是两层动作。宽度样本只在该宽度的深度控制器已经本地覆盖
所有 arm、且当前用的是其最佳深度时才记账。否则记为
`mtp_adaptive_decode_width_deferred_total`，避免把一个宽度的冷启动/深度探索成本
错算到宽度上。未完整执行选定 chunk 的 prompt 尾部或 KV 容量裁剪样本同样
不进入学习，分别记为 censored 和 capacity-limited。宽度小于居留请求数时使用
round-robin，所以优化聚合 token/s 不会饿饿某个 slot。

这里仍有 UCB 置信项和近期收益平滑这类算法常数，但没有绑定 Xeon Max、
3090、某个上下文长度或某个并发数的阈值。稳态选择由当前模型输出内容、
MTP 接受率和实际机器时延共同决定。

### 缓存和 KV 容量修复

长到 200K 后，比调度参数更先暴露的是状态恢复和物理空间边界。本轮同时做了
以下修复：

- hybrid prompt cache 在 hash 粗筛后对 token 做严格 LCP，只允许精确相同前缀
  复用；benchmark 按“客户端新增 token + 上轮实际生成 token”自动设置
  最大允许 reprocess，不用固定 tolerance 掩盖全量重算；
- 替换磁盘 snapshot 时，逻辑 cache budget 可扣除将被替换的旧文件，但原子
  temp-file + rename 所需的物理空间不能扣除。现在每次写入都以
  `filesystem::space().available` 检查原子峰值，必要时先驱逐最旧状态；
- sequence state 可以 scatter restore 到多个空闲 KV range，不再要求一个与
  100K--200K 序列等长的连续空洞；
- server 公开当前最大连续 KV 空洞，prompt logical batch 在每轮按实际空洞自动
  限幅。这是容量反馈，而不是“160K 后就改成 128”这样的上下文硬编码。

### 20K 到 200K 三会话正式结果

正式记录为
`tmp/bench-adaptive-fs-aware-growth200k-v13-20260818`。只配置两个物理 slot，三个逻辑
会话不绑 slot；每个会话从 20K 继续增长到 200K。主 KV 为 BF16，MTP KV 为
Q8_0，hot ring 4,096，`ctx-size=409600`，首层之后 65 个 repeating layer 驻留 3090，
output projection 驻留 HBM，MTP 上限 4，深度/宽度/prompt chunk 全部自适应。

| 逻辑上下文 | 阶段墙钟 | 三路新增 prompt | prompt union PP | decode union TG |
| ---: | ---: | ---: | ---: | ---: |
| 20K | 132.96 s | 60,078 | 516.33 | 15.10 |
| 40K | 206.11 s | 60,192 | 326.62 | 15.17 |
| 60K | 262.82 s | 60,455 | 255.54 | 5.21 |
| 80K | 350.94 s | 60,448 | 185.45 | 5.71 |
| 100K | 391.26 s | 60,328 | 164.42 | 8.13 |
| 120K | 483.91 s | 60,458 | 136.33 | 13.24 |
| 140K | 479.06 s | 60,325 | 133.35 | 7.79 |
| 160K | 623.53 s | 60,461 | 104.81 | 4.26 |
| 180K | 625.58 s | 60,454 | 105.39 | 11.33 |
| 200K | 1,284.44 s | 60,459 | 91.63 | 10.15 |

全程 4,842.70 s，处理 603,658 个 prompt token 和 9,600 个输出 token；按整个服务
墙钟计算为 prompt `124.65 tok/s`、output `1.98 tok/s`、有效总吞吐
`126.64 tok/s`。全部 30 个请求通过自动 prefix-reuse guard；只有首次冷启动的
全 prompt 计入 full-process 计数。过程中无 CUDA OOM、无 ENOSPC、无 KV batch
分配重试、无 fragmented-restore 失败。

终点的 KV fragmentation ratio 最高到 `0.994236`，最小空闲单元 2,805，最小最大连续
空洞只有 128；动态 cap 累计触发 138 次，但没有一次 allocator retry。共有 7 次
大序列使用 fragmented restore。磁盘 cache 逻辑上限 46,000 MiB，原子写入峰值
50.165 GiB，文件系统最低余量 0.872 GiB；三次驱逐全部是为了原子写入的
物理 headroom，没有损坏或部分 snapshot。

全程 GPU 平均 75.24%，P50/P95 为 99/100%，峰值 VRAM 23,920 MiB；PCIe RX
平均/P95/峰值为 9.16/26.11/27.64 GiB/s，TX 为 1.12/2.98/8.22 GiB/s。这些
高 PCIe 样本主要来自 prefill，不表示 decode 饱和。峰值 RSS 30.81 GiB，进程全程平均
约 7.47 个 CPU core。

### 128K 双路长输出与历史 35 tok/s

调度层最后的分层收敛修复用一个独立 128K 实验验收，记录为
`tmp/bench-adaptive-joint-128k-long-v14-20260818`。两路各有 128,026 个 prompt token，
同时请求 2,048 个输出 token；MTP 上限提到 8，其余保持 200K 容量布局。

| 指标 | 结果 |
| --- | ---: |
| 两路 prompt union PP | **266.23 tok/s** |
| 两路 decode union TG | **17.65 tok/s** |
| A / B 单请求 decode | 8.91 / 8.82 tok/s |
| decode 时间并集 | 232.10 s |
| 两路接受/提议 draft | 1,651/1,838；1,642/1,882 |
| 请求完成偏差 | 2.29 s |

控制器确实本地测量了深度 0--8 和宽度 1--2，累计 495 个深度观察、305 个
可比宽度观察，181 个宽度样本因深度仍在探索而延迟记账。在第 256 个可比
宽度样本处，宽度 2/1 的归一化近期收益为 `1.102/0.981`；同期双路深度控制器
的最佳 arm 为 4。这说明双路确实消掉了一部分空泡，但不是线性翻倍。
最后又将 deferred 样本从 width 归一化基准中也完全排除；5K/9K 三会话烟测
`tmp/bench-adaptive-refgate-smoke-v15-20260818` 正常完成，6/6 复用 guard 通过，
无 OOM、ENOSPC、KV retry 或 restore 失败。
最终二进制的 2K/4K 三会话烟测
`tmp/bench-adaptive-final-smoke-v16-20260818` 亦正常结束，深度 0--4、宽度 1--2 都被
观测，6/6 guard 通过，同样没有上述容量/恢复错误。

`17.65 tok/s` 不能与历史 `34.45 tok/s` 解读为同一布局的回归。历史数是
单路 `llama-cli`、`ctx=131072`、`batch=ubatch=768`、24,576 hot、`-ngl 99`，
output layer 和唯一 MTP context 均走 GPU。本次是双 target slot + 双私有 MTP context、
`ctx=409600`、`batch/ubatch=512/256`、4,096 hot、`-ngl 65`，output projection 在 HBM。
为允许两路 MTP-8，target per-step recurrent checkpoint 预留了 2,405.25 MiB VRAM；
峰值总 VRAM 已是 23,750 MiB，不能再直接塞回约 1.19 GiB 的 output layer。

更关键的证据是这一轮 decode 时 GPU 平均只有 20.00%（P50/P95 18/31%），PCIe
RX/TX 只有 1.075/0.120 GiB/s，进程平均约 21.34 个 CPU core。普通用户仍无权
读 HBM PMU，所以不伪造绝对 HBM 数字；历史同类长 decode 只有 169.5 GB/s read，
相对 410 GB/s raw copy 也有明显余量。当前瓶颈是双私有 MTP context、recurrent
per-step checkpoint、CPU output projection 之间的细粒度同步和小图，不是 PCIe/HBM
字节带宽已吃满。

因此下一个值得做的结构优化是减少 recurrent checkpoint 的 VRAM 积（例如只为
实际 cohort/depth 提供可复用 workspace），或把多 slot 的私有 MTP draft 合并成一个可矩形
batch 的共享 context；节省的 VRAM 应优先让 output projection 回 GPU。完整
PagedAttention 仍能提高 KV 容量、前缀块共享和稳态碎片控制，但它不会自动合并私有
MTP 图或消除 recurrent/GPU-CPU 同步；它不是这个 17.65 -> 35 tok/s 缺口的
单独解法。

### 复现命令与可观测性

下面的命令复现三会话 20K--200K 容量验收。磁盘路径必须是实际挂载的
非 HBM 文件系统，并预留足以同时容纳旧文件和 temp snapshot 的原子写入余量：

```bash
BENCH_WORKLOAD=growth \
BENCH_CTX_SIZE=409600 BENCH_DRAFT_CTX=204800 \
BENCH_PARALLEL=2 BENCH_SESSIONS=3 BENCH_STICKY_SLOTS=0 \
BENCH_BATCH_SIZE=512 BENCH_UBATCH_SIZE=256 BENCH_HOT_TOKENS=4096 \
BENCH_GPU_LAYERS=65 BENCH_MTP_MAX=4 BENCH_MTP_ADAPTIVE=1 \
BENCH_MTP_PROMPT_CHUNK=0 BENCH_CACHE_TYPE_K=bf16 BENCH_CACHE_TYPE_V=bf16 \
BENCH_CACHE_TYPE_K_DRAFT=q8_0 BENCH_CACHE_TYPE_V_DRAFT=q8_0 \
BENCH_CACHE_RAM=0 BENCH_CACHE_DISK=46000 BENCH_MEMORY_MAX=46G \
BENCH_CACHE_DIR=/mnt/optane-tmp-stage/ik-prompt-cache-growth200k \
BENCH_CORPUS=/mnt/optane-tmp-stage/ik-hybrid-256k.txt \
scripts/run-hybrid-growth-benchmark.sh tmp/bench-growth200k
```

runner 保留 `manifest.txt`、`source.diff`、Git status 和所有脚本/二进制 hash；
`events.jsonl`/`summary.json` 保留逐请求时间线、复用 guard、MTP 分深度接受数和
union/window 口径。`metrics.prom`、`gpu.csv`、`pcie-dmon.txt`、`pidstat.txt`、
`process-io.csv`、`numastat.txt`、`cache-disk.csv` 保留中间状态；
`observability-summary.json` 只是它们的汇总，后续优化不应只保留最终 tok/s。

## 双路 128K 吞吐恢复与 200K 全尺度复验（2026-08-18）

本节取代上节把 `17.65 tok/s` 当作当前结构上限的判断。历史约 `35 tok/s` 的结果
确实是 **约 128K 长上下文**，不是短上下文：单路 `llama-cli` 在 130K 左右、MTP-8、
24,576 hot rows、output projection 在 CUDA 时曾得到 `34.40--34.45 tok/s`。本轮目标
不是要求两请求严格锁步，而是让调度器在多路 runnable work 中消除空泡，使**系统聚合
decode 吞吐至少追平 34.5 tok/s**。

### 实现收敛点

最终提升由三项结构改动共同得到，而不是把某个上下文长度映射到固定参数：

- recurrent speculative checkpoint 不再按 `sequence × depth` 笛卡尔预留。SSM shadow
  只按一次 physical decode 的聚合 draft-row budget `D` 分配，conv 保存 `D +
  max_sequences` 行；调度器始终约束 `width × depth <= D`。两路 MTP-8 不再为不会同时
  使用的组合静态占用约 2.4 GiB，从而能把 output projection 和 24K hot ring 放回 GPU。
- CPU BF16 flash-attention 按 hybrid KV 的真实 mask 检查冷页。如果一个物理页对当前
  所有 query 都被 mask，QK、softmax 和 PV 全部精确跳过。AMX/AVX 微基准分别从
  `3.183 -> 1.368 ms` 和 `8.107 -> 4.029 ms`，约 `2.33x/2.01x`；稠密与裁剪结果
  最大误差为 0。它不猜测“旧页不重要”，只删除数学上全 masked 的工作。
- 自适应调度器以每个 arm 的累计 decoded tokens / 实际秒数学习，并用本地观测方差的
  UCB 处理噪声。候选深度来自 `[0, n_max]`，宽度来自 runnable slot 数，prompt chunk
  来自 batch/ubatch、root 数和实时 KV 连续空洞；每个 arm 至少有两次本地样本后才参与
  稳态比较。没有 Xeon、3090、128K 或固定 depth/chunk 的查表分支。

32K 两路固定深度扫描说明动作空间确实不单调：depth 2/3/4/5/6/7/8 的 decode union
依次为 `41.57/41.69/51.40/35.72/32.98/29.26/36.72 tok/s`，最佳为 depth 4；换成
另一条增长轨迹后固定 depth 4 为 `50.37 tok/s`，在线方差 UCB 在包含探索成本时为
`45.26 tok/s`。因此“永远 MTP-4”会偶然命中这条轨迹，但不是可迁移的生产策略。

### 128K 性能档验收

正式记录为
`tmp/bench-goal-dual128k-variance-ucb-v29-20260818`。两路各有约 128,026 个 prompt
token，并各生成 2,048 token；`ctx=262144`、MTP-8、BF16 target KV、Q8_0 draft KV、
24,576 hot rows、`-ngl 99`、output projection 在 CUDA。

| 版本 | 关键变化 | 双路 decode union |
| --- | --- | ---: |
| v20 | compact checkpoint + GPU output，尚无冷页裁剪 | 24.37 tok/s |
| v22 | 加入 exact all-masked page pruning，旧调度器 | 30.55 tok/s |
| v29 | page pruning + variance-UCB 深度/宽度调度 | **37.93 tok/s** |

v29 的两请求分别为 `19.04/20.07 tok/s`，decode union `107.98 s`，请求结束偏差
`5.94 s`。它比历史单路目标 `34.5` 高 **9.95%**，比 v20 高 **55.65%**，比 v22
高 **24.18%**。全程无 CUDA OOM、无完整 prompt 重算、无 KV retry 或 restore failure。
FA 共检查 356,343,000 页/query 组合并精确跳过 171,643,000，比例 **48.17%**；GPU
平均/P50/P95 利用率为 `91.84/99/99%`，峰值 VRAM `24,026 MiB`。

这也回答了“双路是否没有意义”：旧双路 `17.65 tok/s` 主要被静态 checkpoint、CPU
output、小图同步和未裁剪冷页拖住；消掉这些空泡后，双路系统吞吐超过历史单路，而不是
退回单槽排队。调度器仍可选择 width 1；当第二路不能提高观测到的聚合 token/s 时，
不会为了形式上的并发而等待或强行锁步。

### 为什么没有先做完整 PagedAttention

PagedAttention 的 block table、copy-on-write prefix sharing 和低碎片 allocator 对长期
容量仍有价值，但不是本轮最快的吞吐修复。当前 canonical K/V 在 CPU HBM，最近窗口在
GPU ring，Qwen3Next 另有 position-sensitive recurrent state 和 MTP per-step checkpoint；
只改 target KV block table 不能合并私有 MTP 图，也不能消除 CPU/GPU 小图同步。

本轮的 mask-driven page pruning 已经在 attention 内部提供了更细的**精确计算调度**，
同时保留现有 canonical KV 和磁盘 state 格式；sequence restore 则可 scatter 到任意多个
空闲物理 range。只有需要让更多不同前缀长期同时驻留、或做跨请求 prefix page sharing
时，才值得把 canonical HBM KV、GPU hot ring、磁盘 snapshot 和 recurrent state 一起
迁移到统一 block table，不能只移其中一层。

### 性能档与容量档必须分开

128K 性能档把 output projection 放在 GPU、hot ring 设为 24K。把同一布局直接扩到
三会话 200K 会超出 24 GiB：hot 24K 和 16K 都在 60K 阶段 OOM；hot 8K 虽通过 80K，
到 100K 已逼近 23.8 GiB，按实测斜率无法到 200K。把 ubatch 从 256 降到 128 只节省
约 370 MiB，却使历史 prefill 下降约 24%，仍不足以补齐容量缺口。

因此 200K 使用独立容量档：`-ngl 65` 让 output projection 留在 CPU HBM，hot ring 4K，
`ctx=409600`。这不是写死调度参数；它是由模型权重、KV 几何和实时可用显存决定的启动
容量 envelope，运行中的 MTP depth、decode width 和 prompt chunk 仍完全自适应。

### 修复后的 20K 到 200K 生产曲线

正式复验记录为
`tmp/bench-goal-production-growth200k-hbm-output-v39-20260818`。两个物理 slot 服务三个
不绑槽的逻辑会话；每个会话逐级增长到 200K，前九级各输出 128 token，终级各输出
2,048 token。主 KV 为 BF16、draft KV 为 Q8_0、MTP 上限 8、磁盘 cache 46,000 MiB。

| 逻辑上下文 | 阶段墙钟 | prompt union PP | decode union TG | reuse guard |
| ---: | ---: | ---: | ---: | --- |
| 20K | 134.99 s | 519.18 | 18.43 | 3/3 |
| 40K | 199.04 s | 350.54 | 12.77 | 3/3 |
| 60K | 284.13 s | 238.64 | 7.09 | 3/3 |
| 80K | 412.45 s | 161.01 | 9.27 | 3/3 |
| 100K | 423.49 s | 153.06 | 16.07 | 3/3 |
| 120K | 441.97 s | 149.46 | 10.55 | 3/3 |
| 140K | 466.30 s | 137.83 | 15.45 | 3/3 |
| 160K | 524.82 s | 123.78 | 8.86 | 3/3 |
| 180K | 552.50 s | 116.90 | 16.32 | 3/3 |
| 200K | 986.10 s | 100.06 | **15.86** | 3/3 |

全程 `4,427.86 s`，处理 603,512 prompt token 和 9,600 输出 token；所有 30 个请求
通过 prefix-reuse guard。200K 三请求的单请求 decode 为 `8.18/17.38/7.60 tok/s`，
总计 6,144 token 的 decode union 为 **15.86 tok/s**。相对旧 v13 的 200K
`10.15 tok/s` 提升 **56.23%**，阶段墙钟从 `1,284.44` 降到 `986.10 s`（-23.23%）。

在首次全尺度复验 v34 中，181K local rewind checkpoint 曾因 PARTIAL_ONLY state 不携带
普通 K/V 行、却仍用 allocator `head` 合成虚假连续目标区间而报
`invalid restored cell range [322246, 503447)`，进而全量重算。现在只有完整 state 才
构造/校验 K/V destination ranges；PARTIAL_ONLY 保留已驻留、可能碎片化的 K/V，只恢复
recurrent/compact state。v39 在 60K、140K、161K、181K 的四次 local checkpoint
restore 全部成功，181K 恢复耗时 `41.32 ms`；7 次完整碎片状态恢复也全部成功。

全程 `cuda_oom=0`、`restore_failure=0`、`kv_retry=0`、`ENOSPC=0`。唯一 full-process
计数是 20K 时第三逻辑会话第一次出现的必要冷启动。动态 KV capacity cap 触发 61 次，
没有 allocator retry；FA 检查 1,087,260,000 页/query 组合并跳过 518,330,000，比例
**47.67%**。GPU 平均/P50/P95 为 `75.40/99/100%`，峰值 VRAM `23,916 MiB`；PCIe RX
平均/P95/峰值约 `9.15/26.14/26.72 GiB/s`，仍是 prefill 脉冲而不是全程饱和。磁盘
snapshot 瞬时峰值 `49.89 GiB`，最低余量 `1.15 GiB`，4 次逐出中 3 次用于原子写入
headroom；测试正常退出后临时 blob 已清空。

### 最终复现命令

128K 性能档：

```bash
BENCH_WORKLOAD=growth \
BENCH_CTX_SIZE=262144 BENCH_DRAFT_CTX=131072 \
BENCH_PARALLEL=2 BENCH_SESSIONS=2 BENCH_STICKY_SLOTS=1 \
BENCH_BATCH_SIZE=512 BENCH_UBATCH_SIZE=256 BENCH_HOT_TOKENS=24576 \
BENCH_GPU_LAYERS=99 BENCH_MTP_MAX=8 BENCH_MTP_ADAPTIVE=1 \
BENCH_MILESTONES=128000 BENCH_DECODE_TOKENS=2048 BENCH_FINAL_DECODE_TOKENS=2048 \
BENCH_CACHE_TYPE_K=bf16 BENCH_CACHE_TYPE_V=bf16 \
BENCH_CACHE_TYPE_K_DRAFT=q8_0 BENCH_CACHE_TYPE_V_DRAFT=q8_0 \
BENCH_CACHE_RAM=0 BENCH_CACHE_DISK=26000 BENCH_MEMORY_MAX=46G \
BENCH_CACHE_DIR=/mnt/optane-tmp-stage/ik-prompt-cache-dual128k \
BENCH_CORPUS=/mnt/optane-tmp-stage/ik-hybrid-256k.txt \
scripts/run-hybrid-growth-benchmark.sh tmp/bench-dual128k
```

200K 容量档：

```bash
BENCH_WORKLOAD=growth \
BENCH_CTX_SIZE=409600 BENCH_DRAFT_CTX=204800 \
BENCH_PARALLEL=2 BENCH_SESSIONS=3 BENCH_STICKY_SLOTS=0 \
BENCH_BATCH_SIZE=512 BENCH_UBATCH_SIZE=256 BENCH_HOT_TOKENS=4096 \
BENCH_GPU_LAYERS=65 BENCH_MTP_MAX=8 BENCH_MTP_ADAPTIVE=1 \
BENCH_MILESTONES=20000,40000,60000,80000,100000,120000,140000,160000,180000,200000 \
BENCH_DECODE_TOKENS=128 BENCH_FINAL_DECODE_TOKENS=2048 \
BENCH_CACHE_TYPE_K=bf16 BENCH_CACHE_TYPE_V=bf16 \
BENCH_CACHE_TYPE_K_DRAFT=q8_0 BENCH_CACHE_TYPE_V_DRAFT=q8_0 \
BENCH_CACHE_RAM=0 BENCH_CACHE_DISK=46000 BENCH_MEMORY_MAX=46G \
BENCH_CACHE_DIR=/mnt/optane-tmp-stage/ik-prompt-cache-growth200k \
BENCH_CORPUS=/mnt/optane-tmp-stage/ik-hybrid-256k.txt \
scripts/run-hybrid-growth-benchmark.sh tmp/bench-growth200k
```

最终构建通过；`test-amx-bf16`、`test-amx-int8`、`test-model-load-cancel`、
`test-autorelease` 为 4/4，三份 benchmark Python 驱动可编译，runner `bash -n` 和
`git diff --check` 均通过。

完整 CTest 共 25 项，其中与当前构建环境可独立执行的 21 项为 **21/21**。其余四项均
单独复现为既有环境或 fixture 问题：`test-jinja-py` 缺少 Python `jinja2`，
`test-eval-callback` 所需测试模型在无 libcurl 构建中不能下载，
`test-tokenizer-0-bert-bge` 的 vocab/期望 token 不一致，`test-chat-template` 的
ChatGLM4 期望与实际输出存在尾部换行差异；它们不经过本次修改的调度、KV restore 或
CPU flash-attention 路径。
