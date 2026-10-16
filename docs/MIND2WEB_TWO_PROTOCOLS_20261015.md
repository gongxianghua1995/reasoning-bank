# ReasoningBank 两组实验计划与进度

用户于 2026-10-15 确认开展两组实验：A 保留论文自评持续学习流程；B 对齐 SwarmEvolution 的训练、官方反馈和冻结评测，保持单 agent 及 ReasoningBank 的经验提炼、表示、检索。旧实验不覆盖。

## 固定协议

| 项目 | A 论文流程适配 | B SwarmEvolution 协议对照 |
|---|---|---|
| 数据 | 三个官方测试 split 各自空库，固定顺序一次 | seed42 训练303、验证101，三个官方测试集 |
| 反馈 | LLM 自评，无官方评分输入 | 训练执行完后接收官方任务及逐步评分，无正确动作和 gold 值 |
| 更新 | 每题结束后追加 | 每 batch 固定库，执行全部完成后提炼，批末一次提交 |
| 训练 | 不预训练，不重跑题 | 2 epoch，batch8；与当前 SwarmEvolution 一样按网站分组构造训练列表，再用 seed+epoch-1 打乱 |
| 验证 | 独立验证流仅用于冒烟，不带入测试 | 每轮快照在完整101题上只读评分；选择最佳宏平均 Step SR，平分选早轮 |
| 测试 | 每个 split 各自连续学习，禁止分片共享库 | 选定快照冻结，三个 split 分开报告 |
| 对照 | 相同模型、输入和配置的单 agent 无记忆 | 使用同一无记忆结果，避免重复收费 |

两组统一 DeepSeek-V4-Flash-0731、动作 temperature0.7、本地 BGE-small plain query、cosine top1、每份经历1–3条 Title/Description/Content 经验、原版成功/失败提炼 system prompt。保留逐条适用性判断和离线因果提示。模型及 embedding 服务是明确替换，A 不称原论文数值完全复现。B 不增加 Planner/Verifier、WriteLocal 或 Distill，不导入正确轨迹。

官方反馈 B 与 SwarmEvolution 保持相同信息层级：official_score、FULL/PARTIAL/NO、action_metrics（各步 EA/Operation F1/Step SR/候选召回等），无 gold element/value。反馈仅在该题动作全部结束后供提炼读取。A 继续先自评提炼，再独立评分，绝不读取该字段。

B 第二轮可以检索第一轮同一训练题的经验，符合重复训练语义；同批新经验仍不可见。A 每题仅一次；冻结评测的题与训练互斥。验证快照按宏平均 Step SR 选优，平分选早轮；不把 SwarmEvolution 的角色候选提炼机制加入 ReasoningBank。

SwarmEvolution 每角色验证候选的次数多于 B 的每轮快照选择，成本分别报告，不能宣称总训练算力相等。它当前模型别名为 deepseek-flash，最终横向比较还需核验实际运行模型、预算和输出清单；本次不启动或修改其他人的 SwarmEvolution 进程。

## 执行顺序与门槛

1. 实现官方反馈白名单、批次冻结、顺序一致性、事务提交、每轮快照及恢复。
2. 运行协议测试，验证评分不进入 actor、无答案泄漏、同批经验不可见、下批可见、恢复不重复学习。
3. B 用训练清单中的小规模完整任务冒烟（两个批次）；A 已有两题成功冒烟，并在新测试中继续检查隔离。
4. 冒烟通过后冻结代码与配置，启动 A 三个独立官方 split 流、B 训练→两份快照验证→选择→冻结测试、无记忆对照。流内串行，独立组可并行，限制总请求并发。
5. 每15秒心跳；保存原始请求、响应、token、评分、反馈来源、记忆/源码/配置哈希及失败覆盖。额度或配置异常停止对应任务并留档。

完整官方测试数量：Cross-Task252、Cross-Website177、Cross-Domain912。测试集不用于调试、配置选择或 B 经验构建。A 的测试轨迹写库属于预先固定的无标签测试时学习协议。

## 进度

- 已完成：原流程及 SwarmEvolution 数据顺序、反馈字段、2轮/batch8 配置核对。
- 已完成：B 官方反馈白名单、批次快照、批末事务提交、每轮记忆快照、训练中断恢复、验证快照选择与完整队列。
- 已完成：33 项测试通过，含评分隔离、无正确动作泄漏、同批经验隔离、下一批可见、第二轮同题经验可检索、提交中断恢复不重复学习、验证缺测重试和完整调度依赖。
- 已完成：B 真实三题训练冒烟，两个批次（2+1），25个完整动作步骤、3次提炼、零自评调用、零接口错误，写入3份经历共8条经验，耗时476.18秒。两题首批均空库，第二批检索首批经历；三题官方标签均为 fail，与独立评分一致。训练宏平均 Step SR 64.47%，不作为泛化结论。
- 已完成：冻结验证一题4步，有经验与无经验均3/4正确；仅动作调用，无自评/提炼，原始及复制经验库哈希不变。该小样本只验证链路。
- 已启动：完整两组与共享无记忆对照，目录 `outputs/mind2web/two_protocols_20261015T085651Z`，启动 PID 555202。当前进度以运行目录 `status.json` 为准，不以文档中的启动 PID 判断存活。

此前负增益结果及持续学习两题冒烟保留在各自原目录，不混入新正式实验。

本轮训练冒烟：`outputs/mind2web/aligned_smoke_20261015T084549Z`；冻结冒烟：`outputs/mind2web/aligned_frozen_smoke_20261015T085534Z`，审计在 `smoke_audit.json`。完整实验的 `smoke_gate.json` 保存验收哈希，`protocol_tests.log` 保存33项测试输出，`alignment_reference.json` 保存 SwarmEvolution 源码哈希及两轮完整任务顺序的核对证据。

## 运行与产物

```bash
MIND2WEB_INVENTORY=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/mind2web_prepared_20261013.json"))["inventory"])')

# 完整队列：两轮训练 + 两轮快照验证 + A/B 三测试 split + 共享无记忆对照
.venv/bin/python -u -m mind2web_adapter.campaign \
  --inventory "$MIND2WEB_INVENTORY" --output outputs/mind2web/two_protocols_NEW \
  --workers 8 --training-workers 4 --env-file /home/xhgong/project/GPTSwarm/.env

# 单独恢复 B 训练，使用该训练目录内冻结源码、原配置和事务记录
.venv/bin/python -u -m mind2web_adapter.aligned \
  --output outputs/mind2web/two_protocols_NEW/aligned_train --resume \
  --env-file /home/xhgong/project/GPTSwarm/.env

.venv/bin/python -m unittest discover -s mind2web_adapter/tests -v
```

共计划 4932 次任务执行：B 训练606、快照验证202、A 测试1341、B 测试1341、无记忆验证/测试1442。验证缺测重试另计。总并发8，B 训练并发4（调度参数，batch8不变），A 各 split 串行。三个 A 流始终使用独立经验库。所有正式阶段从同一源码快照启动，配置和输入清单哈希固定。

运行目录 `plan.json` 保存阶段、依赖、清单与命令；`status.json` 每15秒更新子进程进度；`REPORT.md` / `comparison.json` 汇总覆盖率、宏平均指标及共同完成题的配对差值；`selection.json` 保存只基于验证集的版本选择；`usage.json` 按阶段记录返回 token 与调用量。失败请求可能产生未返回 usage 的费用。验证最多3次只重跑缺测，仍缺分则阻断 B 正式测试，不按完成子集选快照。训练与测试的可恢复单题失败留档、报告覆盖率，配额/权限/完整性异常停止队列。

B 与 A 共用的单 agent 无记忆对照不会重复收费。当前 SwarmEvolution 的默认解码及预算不保证与本组完全一致；本次对齐的是数据、训练组织、反馈和冻结边界，后续与其实际成绩合表前必须对齐实际模型调用配置并报告预算差异。
