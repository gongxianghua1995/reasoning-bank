# ReasoningBank Mind2Web 适配

## 当前入口（2026-10-15 更新）

最新实验分为论文流程组与 SwarmEvolution 协议对照组，见 [两组实验计划与进度](MIND2WEB_TWO_PROTOCOLS_20261015.md)。`mind2web_adapter.campaign` 负责完整队列；`mind2web_adapter.aligned` 负责官方反馈的批次训练；`run --mode stream` 保留自评持续学习。两组都保留 ReasoningBank 单 agent 与经验机制。

参考论文的持续学习流程，新增 `--mode stream`：每个 split 从空记忆开始，按固定顺序执行一次，每题结束后自评、提炼并写入经验，后面的题可检索之前的经验。使用 `config_stream.json`，包括逐步经验适用性判断、适配本地 BGE 的 plain query、动作 temperature 0.7 和离线因果约束。详情及命令见 [持续学习协议与验证](MIND2WEB_STREAM_20261015.md)。

下文的 `train` → `eval` 是此前的**训练后冻结评测协议**，保留用于历史对照。旧两轮批次已停止，一轮记忆的验证集对照已完成；不应再按旧批次命令启动第二轮。冻结只意味着评测期间经验库只读，并不是不调用模型或固定模型回答。

## 方法范围

使用单 Web agent 做固定观察下的动作预测，保留 ReasoningBank 的任务级经验检索、成功/失败自评和经验提炼。执行器为此任务新增，未直接复用 mini-swe-agent 的 Bash 循环或 WebArena 的浏览器操作。三角色协作、SwarmEvolution 的归因/WriteLocal/Distill/Select 均不参与。

每个完整任务开始时按当前 query 检索一个历史任务的全部经验条目，注入该任务所有步骤。当前 case 的历史条目不参与它自己的检索。每步只提供当前精简 DOM、候选局部 ID 与此前人工动作；模型输出 `{element_id, operation, value, reason}`，操作为 CLICK/TYPE/SELECT。局部 ID 转回原 ID 后独立评分。

每步一次模型动作请求，格式/字段错误最多修复一次，两次共用步骤预算；再次失败计为该步无效动作。没有额外规划角色、Verifier 或动作效果重试。下一步读取数据集固定观察，不执行预测动作、不打开浏览器。API/预算失败使运行标为不完整，不把未执行步骤补零伪装成完整评测。

训练时在完整任务结束后自评、提炼 1–3 条经验并原子写回。自评和提炼只读取公开输入与 agent 轨迹，评分在记忆步骤结束后计算。离线轨迹中的后续页面和人工历史不能证明预测动作成功，提示中明确说明该限制；自评与独立分数分开记录。

验证/测试从训练记忆复制一个冻结快照，只检索、不自评、不提炼、不写入，并校验副本与原文件哈希。无记忆组完全跳过检索与学习。官方测试集不用来调试或构建记忆。

## 与原版的差异

- 新增 Mind2Web 离线单 agent 与固定观察动作协议；官方仓库没有直接可用的 Mind2Web 入口。
- 保留原版 Web 成功/失败提炼 system 提示；新增离线历史说明与有界格式修复。
- 默认 `global` 汇集所有训练网站的经历；`local` 只检索同网站，`none` 无记忆。官方 WebArena 流水线按网站分别保存记忆，global 是显式跨网站适配。
- 固定训练/验证/测试切分并冻结测试记忆，区别于原论文测试时持续学习。支持多 epoch，episode 身份和 case 身份分离；新训练必须空记忆起步。
- 检索为任务 query 的 instruction-aware cosine top-1，使用本地 BAAI/bge-small-en-v1.5（384 维、512 token 截断），替代 Gemini embedding。模型权重摘要随运行记录，不在运行时下载。
- 不启用 MaTTS。额外记忆学习调用单独统计，不能宣称与无记忆组总成本相同。

## 公共数据与评分

来源固定为 agent_evolving `b08a5e1559d279ccbaed278d2bfff08e33c6bd38`。原样复用其 dataset/cache/metrics 与固定官方 DOM 工具，来源及 SHA-256 见 `mind2web_adapter/vendor/SOURCES.json`。只替换包初始化以隔离上游执行/学习模块。官方 DOM 工具的许可随代码保留。

官方原始数据位于 `/home/xhgong/project/agent_evolving/SwarmEvolution/dataset/mind2web/raw`。全量四 split 的完整性、seed 42 的 303/101/605 划分与官方三个测试集均核对；不是下载少量样本后重新随机切分。转换缓存与 SwarmEvolution 共享，转换协议、候选排名和输入哈希相同。

准备共享转换缓存（输出必须为新文件）：

```bash
.venv/bin/python -m mind2web_adapter.prepare \
  --data-root /home/xhgong/project/agent_evolving/SwarmEvolution/dataset/mind2web/raw \
  --ranks /home/xhgong/project/agent_evolving/SwarmEvolution/dataset/mind2web/raw/scores_all_data.pkl \
  --cache-root /home/xhgong/project/agent_evolving/SwarmEvolution/.cache/mind2web \
  --output outputs/mind2web_prepared_NEW.json
```

指标使用任务宏平均 Step Success Rate、Element Accuracy、Operation F1、严格 Task Success Rate 与候选召回率。模型没有见到 gold 动作或正例标记；没有按答案补入候选。完整任务不截断步骤。

## 运行入口

运行环境为本项目 `.venv`，版本见 `mind2web_adapter/requirements.lock`。首次转换后，将准备记录中的 `inventory` 路径传给运行器；它是内容校验过的统一内部缓存，并非另一份数据划分。

```bash
# 本机已准备好的完整共享缓存
MIND2WEB_INVENTORY=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/mind2web_prepared_20261013.json"))["inventory"])')

# 全新目录；训练冒烟选 seed 清单前两条，默认一轮
.venv/bin/python -u -m mind2web_adapter.run --mode train --scope global \
  --inventory "$MIND2WEB_INVENTORY" --seed 42 --max-cases 2 \
  --env-file /home/xhgong/project/GPTSwarm/.env \
  --output outputs/mind2web/train_NEW

# 同一划分的验证清单第一条；只读训练记忆
.venv/bin/python -u -m mind2web_adapter.run --mode eval --scope global \
  --inventory "$MIND2WEB_INVENTORY" --seed 42 --split validation --max-cases 1 \
  --memory-from outputs/mind2web/train_NEW/memory.json \
  --env-file /home/xhgong/project/GPTSwarm/.env \
  --output outputs/mind2web/frozen_NEW

# 无记忆对照：使用同一验证 case；不要传 --memory-from
.venv/bin/python -u -m mind2web_adapter.run --mode eval --scope none \
  --inventory "$MIND2WEB_INVENTORY" --seed 42 --split validation --max-cases 1 \
  --env-file /home/xhgong/project/GPTSwarm/.env \
  --output outputs/mind2web/none_NEW
```

历史批次曾配置两轮训练，现已按用户要求停止第二轮。其一轮记忆的完整验证结果见 [验证集报告](MIND2WEB_VALIDATION_EPOCH1_20261015.md)。新的论文流程使用上面的 `stream` 入口，固定每题一次。原始失败记录保留，新运行必须使用新目录。

针对动作执行中断或最终学习失败的训练故障，可用 `run --resume-from <失败训练目录>` 或 `batch --resume-train-from <失败训练目录>` 恢复。数据、模型权重、成功前缀与记忆逐项核对；已完成动作复用，剩余动作扣除此前已消耗的任务时间，失败请求归档并保留成本。学习恢复重新分配阶段预算。配置及源码变化需通过固定迁移清单校验。该入口不支持任意阶段中断，也不自动重启已退出的批次。

历史 `mind2web_adapter.batch` 入口将两轮训练、四个 split 的冻结评测及无记忆对照串成固定队列，当前已停止；它不是新的持续学习入口。批次启动时冻结所有阶段的源码与配置，自动汇总每阶段结果。`--continue-case-errors` 使可恢复的单题失败留档并继续，进程级严重错误停止。

`case_outcomes.json` 记录所有已处理 episode；`failures.json` 记录动作未完成或学习失败。动作不完整的题不补零、不生成完整评分；学习失败保留动作评分但不写经验。summary 报告完整评分子集指标与覆盖率，存在失败时为 `finished_with_failures`。必须连同未完成数量解释准确率，避免误报全量成绩。冻结／无记忆评测也使用同一失败处理政策。原始失败文件与调用成本保留。

## 日志与预算

默认模型 DeepSeek-V4-Flash-0731、temperature 0.2、最大输出 16384 tokens。完整任务动作阶段最多 1200 秒，每步、自评和提炼的基础预算各 180 秒，各最多一次格式纠正。2026-10-14 起启用 timeout_recoveries=2：临时传输错误用尽阶段预算时最多续给两次基础预算，单次响应最多三次传输尝试；动作仍受任务累计 1200 秒硬上限约束，学习阶段含恢复最多约 540 秒。格式纠正不重置延期额度，额度耗尽不重试。新增调用和阶段延期单独写入事件日志。

每次运行冻结自身源码（含公共数据代码与原版 Web 提示）、配置和完整切分。保存模型实际请求、响应、usage、逐步动作、独立评分、自评、记忆检索/写入与哈希。`status.json` 每 15 秒心跳；每步更新任务、步骤与阶段。凭据仅从宿主环境读取，不复制到快照。

```bash
.venv/bin/python -m unittest mind2web_adapter.tests.test_contract -v
.venv/bin/python -m mind2web_adapter.summarize outputs/mind2web/train_NEW
```

5 项契约测试全部通过，包括完整的模拟训练两题 → 冻结评测链路；它们不产生真实模型准确率。真实冒烟也已完成：训练两题写入 6 条经验，验证有／无记忆均 4/9 步正确。详见 [真实冒烟报告](MIND2WEB_SMOKE_20261013.md) 和 [适配计划](MIND2WEB_ADAPTATION_PLAN.md)。
