# ReasoningBank SWE 适配说明

适配保留仓库自带 mini-swe-agent 1.10.0 的 Bash ReAct 解题循环、原生 selected_memory 注入、成功/失败经验提炼提示。每题产生一条解题轨迹，执行结束后自评并总结，供下一题检索。没有多角色协作或 MaTTS。

## 流程

```text
当前公开 issue → 检索先前任务经验 → 原生解题 agent ↔ 断网 Docker
                                             ↓
                                         实际代码修改
                                             ↓
                                 宿主独立导出 diff 和完整轨迹
                                             ↓
                          LLM 自评 → 原版提示提炼经验 → 原子写回记忆
                                             ↓
                              独立官方评分，结果不进入记忆生成
```

生成容器可以读写源文件、执行 shell 和本地测试；仓库精确恢复到题目基线并隔离未来 Git 历史。输出补丁由宿主读取实际工作树，即使 agent 超时或漏交 untracked 文件也独立尝试导出。凭据只在宿主，生成/评分容器 network=none。

## 与官方实现的差异

| 部分 | 当前配置 |
|---|---|
| 解题 agent | 保留 vendored DefaultAgent 1.10.0；添加保存轨迹的子类，不替换原生 run/query/parse_action/execute_action |
| 模型 | OpenAI 兼容接口，DeepSeek-V4-Flash-0731；解题 temperature 0.2、自评 0、提炼 1 |
| 检索 | 保留当前 issue 对历史 query 的 cosine top-1；使用原检索指令构造 query，取所选经历的全部记忆条目 |
| Embedding | 本地 BAAI/bge-small-en-v1.5 的 FastEmbed ONNX 实现，384 维、512 输入 token 截断；替代官方 Gemini embedding，权重文件 hash 固定 |
| 记忆生成 | 保留官方 SUCCESSFUL_SI / FAILED_SI；严格解析 success/fail，按 Memory Item 标题解析最多三项完整条目；提炼输入明确标记历史消息与指令边界，格式错误最多纠正一次，共用原有提炼预算 |
| 状态管理 | 全新批次空记忆起步，固定题目顺序；记忆和对应 query 向量一次原子写入；重复任务拦截 |
| 预算 | 解题 1200 秒、250 步、单次输出 16384；命令 120 秒、API 300 秒并受阶段剩余时间约束；自评/提炼各另有 180 秒 |
| 异常 | 模型临时错误最多三次请求尝试，等待计入阶段预算；错误类型与 usage 记录；记忆生成失败后停止流，仍评分已导出的补丁 |

上述设置是适配实验，不是原论文分数复现。两题只能检验跨任务记忆链路；第二题仅有一个候选经历，不能证明语义排序效果或准确率提升。无模型测试另用多个候选验证排序。memory=none 关闭检索、自评、提炼与写回，是同一 agent 的无记忆入口，但本次真实冒烟以带记忆流程为主。

自评、提炼和 embedding 消耗须与解题消耗分别报告。当前默认是给记忆阶段额外预算，不宣称与无记忆组总预算相同。官方评分位于另一个解释器，隐藏测试和 gold patch 不进入生成请求。

## 入口与环境

主环境为本项目 `.venv`，依赖见 `swe_adapter/requirements.lock`。通过 `swe_adapter` 初始化将本仓库 `third_party/src` 放入导入路径，避免误用其他环境中的 mini 2.x。没有修改官方 agent 和记忆提示源码。

```bash
# 首次准备 embedding，下载只发生在这一步
.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='.models', threads=2)"

# 连续两题，自动冻结源码、数据、配置和 embedding 摘要
.venv/bin/python -u -m swe_adapter.run \
  --data-path /home/xhgong/project/GPTSwarm/outputs/swebench/swebench_verified_test_154.json \
  --instance-ids django__django-10999 django__django-11141 \
  --eval-python /home/xhgong/project/GPTSwarm/.venv/bin/python \
  --env-file /home/xhgong/project/GPTSwarm/.env \
  --output-dir outputs/swebench/reasoningbank_smoke_NEW
```

加 `--check` 只核对输入、解释器与本地镜像，不调用模型。加 `--memory none` 使用无记忆模式。新输出目录必须不存在；当前不支持自动续跑和失败重试批次，避免重复任务写入记忆。

## 实验记录

- 批次 `status.json` 每 15 秒更新任务、阶段、已完成列表及耗时。
- `manifest.json`、`source_at_run/`、`source_hashes.json` 固定本次代码、输入及模型参数。
- `memory.json` 是本批次记忆；各题 `memory_before.json`、`retrieval.json`、`memory_after.json` 固定可见经验。
- `request.json`、`trajectory.json`、`events.jsonl` 保存实际输入、交互及模型/命令记录；每次解题请求记录 memory_in_system。
- `judge_request.json`、`self_judgment.json`、`extract_request.json`、`memory_entry.json` 保存经验生成链路。
- `memory_raw_attempt_N.md` 保存每次提炼正文；若格式纠正，另存 `extract_request_retry.json`，事件记录 `memory_format_error`。这些日志从 2026-10-13 修复版本开始提供。
- `selected.patch`、`generation.json`、`report.json`、`evaluation.log` 保存真实补丁与独立评分。

公共工作区和评分代码来源见 `swe_adapter/vendor/SOURCES.json`。本次不修改 CAMEL/GPTSwarm 的代码或实验结果。

## 已完成的真实验证

见 [2026-10-13 冒烟报告](SWE_SMOKE_20261013.md)：两题官方评分 0/2，真实容器编辑、测试、补丁导出、前题记忆注入及独立评分均已执行。原始批次末尾提炼格式错误，修复后仅单独重放提炼成功；没有将这一补验记作新的完整两题成功运行。当前没有无记忆对照结果，不能据此声称记忆提升准确率。
