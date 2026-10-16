# Mind2Web 并发调整记录

2026年10月16日，用户要求提高当前实验并行度。原队列总容量8，B训练占4，A三个域各占1，独立评测仅剩1。调整只涉及调度和评测断点恢复，A域内经验顺序、B批次大小、模型、提示、数据和评分均保持原设置。

## 已执行调整

1. 新增可以接管已有进程的调度入口，保留原实验子进程及冻结源码。9项调度测试通过，包括整条任务依赖链、缺测验证重试、禁止重复任务和 PID 身份核验。
2. 01:47:50 UTC 将总容量从8提高到16，新增8个独立评测进程；核验原有5个实验进程 PID 保持不变，冻结源码、原配置和任务清单哈希一致。
3. 16并发触发明显限流：一个60秒观测窗口内31次请求发生17次 RateLimitError，并出现重试耗尽的单题缺测。未继续提高容量。
4. 回调到12，将4个新增无记忆评测进程保存断点后排队；它们共保留9步预测。恢复沿用原冻结 agent，保留中断前已消耗的动作时间，排队时间不计入任务预算。2项恢复测试通过，验证已完成步骤复用、未提交步骤截断、原始轨迹归档及拒绝修改有记忆学习流。
5. 12并发下仍持续限流，后续60秒窗口中40次请求发生27次 RateLimitError。因此继续回调到10，再将2个无记忆进程保存断点排队，额外保留8步预测。当前共6个恢复进程等待名额，实际执行容量10。

## 当前并行分配

| 部分 | 原先 | 调整后 |
|---|---:|---:|
| B训练执行并发 | 4 | 4 |
| A三个测试域 | 各1 | 各1 |
| 独立评测可用并发 | 1 | 3 |
| 总执行容量 | 8 | 10 |

B训练和A流保留原进程。训练结束后，释放的4个名额用于独立评测。冻结评测可以继续分片并行；A同一域内按题更新记忆的顺序依赖仍是整体完工时间的重要限制，因此不能将并发比例当成总耗时缩短比例。

## 排队和计数说明

回调期间调度器的 `used_slots` 可能仍显示16，其中6个是只等待容量的恢复进程，不发起模型请求，实际执行负载为10。它们等待已有任务结束、预留容量降至10后再恢复。相应子任务状态为 `queued / capacity_wait`，不是模型卡死。

原始 `plan.json` 保留 workers=8，作为启动时配置。实际并发以 `runtime_concurrency/capacity.json` 为准，当前 workers=10；原输入哈希不改写。`latest_two_protocols.json` 已更新为新调度器 PID，不能再用原 PID 判断运行状态。

试调期间的限流错误和缺测保留在原始事件与失败记录中，不能当成错误动作计分或从报告中隐去。10并发下的稳态吞吐和限流仍需观察，不宣称已获得固定倍数加速。

回调到10后的约95秒核验：14次新请求、12次响应、3次限流和1次接口超时；请求与响应可能跨观察窗口，不能简单相减当作失败题数。最后60秒内有1次限流、1次超时，较高并发试调窗口明显减少，但并非零错误。11项调度及恢复测试全部通过；A／B原进程、科学执行源码及原始输入哈希保持一致。

## 审计与代码

- [11项调度与恢复测试记录](../outputs/mind2web/two_protocols_20261015T085651Z/runtime_concurrency/tests.log)。

- [容量配置](../outputs/mind2web/two_protocols_20261015T085651Z/runtime_concurrency/capacity.json)。
- [初次接管核验](../outputs/mind2web/two_protocols_20261015T085651Z/runtime_concurrency/verification.json)。
- [限流回调与步骤恢复记录](../outputs/mind2web/two_protocols_20261015T085651Z/runtime_concurrency/backoff_audit.json)。
- [回调至10的恢复记录](../outputs/mind2web/two_protocols_20261015T085651Z/runtime_concurrency/backoff10_audit.json)。
- [最终核验及请求统计](../outputs/mind2web/two_protocols_20261015T085651Z/runtime_concurrency/verification_final.json)。
- [当前运行状态](../outputs/mind2web/two_protocols_20261015T085651Z/status.json)。
- 调度实现：`mind2web_adapter/campaign.py`、`campaign_live.py`；断点恢复：`eval_resume.py`。新增控制器源码另行冻结，实验执行仍使用原 `source_at_launch` 和各运行的 `source_at_run`。
