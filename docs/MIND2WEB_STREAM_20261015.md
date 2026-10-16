# ReasoningBank Mind2Web：参考论文的持续学习协议

更新时间：2026-10-15。用户要求参考论文实验步骤修改。历史训练和冻结验证结果保留，不覆盖、不与新协议合并统计。

## 冻结评测和持续学习

冻结评测：`训练题 → 构建经验库 → 固定经验库 → 验证/测试只检索`。每道评测题读取同一份库，题目结束后不自评、不提炼、不写回。它测的是训练经验对未见任务的迁移。被冻结的是经验库，不是推理过程；这里两个协议都不更新模型权重。

持续学习：`空库 → 第 1 题执行、自评、提炼、写回 → 第 2 题检索前题经验并执行、写回 → …`。当前题只使用先前任务的记忆；当前题的经验在回答后才加入库。每题只跑一次，不需要第二轮。评分器不向 agent、自评器或提炼器提供参考答案或分数。

论文 §3 与附录 A.2 描述从空记忆出发的顺序测试时学习、query 检索 top-1、逐步判断经验适用性、LLM 自评和成功/失败提炼。来源：[ReasoningBank v2](https://arxiv.org/html/2509.25140v2)。官方公开代码没有可直接执行的 Mind2Web 入口；其具体 Mind2Web 任务排序、跨 split 初始化等不能声称已经逐行复现。

## 修改点

| 部分 | 旧实验 | 新的 `stream` 默认配置 |
|---|---|---|
| 学习流程 | train 构库，eval 冻结 | 当前 split 从空库开始，按题学习一次 |
| 记忆使用 | 检索内容直接注入 | 每步对每条经验给出 use/skip 和简短适用理由，允许全部跳过 |
| 本地 BGE 查询 | 长 Instruct 前缀 + query | plain query，与经验 query 的编码形式一致 |
| 动作采样 | temperature 0.2 | temperature 0.7，参考论文；自评 0、提炼 1 |
| 离线学习 | 提示说明静态观察 | 强化约束：预测动作未执行，后续页面不能作为预测成功的因果证据 |
| 审计 | 请求、轨迹、独立评分、哈希 | 额外记录协议、顺序哈希、与论文差异及逐条经验使用判断 |

`run.py`、`agent.py`、`memory.py`、`resume.py` 负责上述行为；新配置为 `config_stream.json`。旧 `config.json` 和旧输出不改写。plain query 是为本地 BGE 修正的适配选择，不应声称是论文 Gemini embedding 的完全复现。先前检索探针表明该修改减少了检索集中，但它本身不证明动作准确率提高。

## 顺序与公平对照

- 固定 seed 42 的既有 split 清单顺序；每个 split 单独空库起步，不能导入训练记忆，也不读取之后任务的经验。
- 同一条持续学习流串行执行，禁止按评测分片并发后合并记忆。独立实验可以并行，但不能共享可写库。
- `--scope none` 使用同一新配置和同一顺序，跳过检索、自评和提炼，作为无记忆对照。旧 temperature 0.2 的无记忆成绩不能直接作为新协议的严格对照。
- 每题结束先自评和提炼，再由独立评分器计算指标。完整评分覆盖率、失败数、调用和耗时都必须一起报告。
- 支持已有故障恢复机制：`--resume-from` 只恢复同一流中已审计的前缀，必须通过配置、来源、顺序和记忆核对；不是任意导入已有经验。已完成题目不重复学习。
- 官方测试题可以在这个预先约定的测试时学习协议下提供无标签执行经历；不能用其 gold 分数调参。当前只用验证集前两题调试。

## 与论文仍有的差异

1. 当前可用模型是 DeepSeek-V4-Flash-0731，非论文的 Gemini 2.5 Flash/Pro；未配置 Gemini 服务凭据。
2. 检索用本地 BAAI/bge-small-en-v1.5，非 gemini-embedding-001。
3. 沿用 SwarmEvolution 的精简 DOM、top-50 候选、固定人工历史和独立评分器；预测不会实际驱动页面。Mind2Web 固定观察预测与论文同属离线任务，但具体输入处理未证明完全一致。
4. 使用本地固定顺序和独立 split 空库初始化；论文公开材料不足以逐项核对原始 Mind2Web runner 的这些细节。
5. JSON 中的简短适用性判断、离线因果保护提示和有界错误恢复是本适配的实现选择。保留 1200 秒动作任务预算；未启用 MaTTS。

因此新结果应标为“ReasoningBank 持续学习适配”，并附模型和协议；不称为论文原始数值复现。

## 命令

在 ReasoningBank 仓库执行，输出目录必须尚不存在：

```bash
MIND2WEB_INVENTORY=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/mind2web_prepared_20261013.json"))["inventory"])')

# 两题真实冒烟：默认加载 config_stream.json
.venv/bin/python -u -m mind2web_adapter.run \
  --mode stream --scope global --split validation --seed 42 --max-cases 2 \
  --inventory "$MIND2WEB_INVENTORY" \
  --env-file /home/xhgong/project/GPTSwarm/.env \
  --output outputs/mind2web/stream_smoke_NEW --continue-case-errors

# 完整验证流：去掉 --max-cases；独立无记忆对照使用 --scope none 和新目录。
# 正式测试时分别选择 test_task / test_website / test_domain，每组新目录、空记忆。

.venv/bin/python -m unittest discover -s mind2web_adapter/tests -v
```

## 验证记录

25 项测试通过，包含顺序学习、空库初始化、独立流重置、禁用导入/第二轮/分片、每步经验适用性字段、plain query、自评前无评分泄漏、无记忆组无学习调用，以及中断恢复不重复学习。

真实两题冒烟目录：`outputs/mind2web/stream_smoke_20261015T073432Z`。状态 `finished`，耗时 405.54 秒；13 次动作请求、2 次自评、2 次提炼，全部有返回，没有模型错误、格式修复或未完成任务。冻结运行源码哈希全部一致。运行快照早于随后加入的 stream 恢复支持，恢复路径由独立测试覆盖。

| 验证题 | 检索 | 正确步骤 | 自评 | 写入条目 |
|---|---|---|---|---|
| `5f09e15c-c987-4bdb-ab6e-5db39b18317d`（Resy） | 空库，无经验 | 4/9 | success | 3 |
| `0245df99-2909-465a-861e-7fbca948e82f`（Budget） | 第 1 题的经验 | 3/4 | fail | 2 |

第二题 4 步都包含对 3 条经验的适用性判断，共 12 个 `use=false`，理由指出租车任务与活动票务、城市切换和票数选择的条件不匹配。证明前题写入、后题检索、逐步判断及允许跳过的链路实际工作；自报 `use=false` 不证明经验在模型内部完全没有影响。

最终经验库含 2 份任务经历、5 条经验。任务宏平均 Step SR / Element Accuracy 均为 59.72%，Operation F1 83.33%，Task SR 0/2；按全部步骤计为 7/13（53.85%），与任务宏平均不同。两题没有无记忆配对，不能据此判断增益。

第一题自评 success 与严格评分不符，说明增加离线证据约束后自评噪声依然存在；这不是已解决的问题。未将 gold 反馈加入学习来修正标签。完整审计见运行目录下 `audit.json`、`stream_protocol_audit.json` 及每题请求和轨迹。
