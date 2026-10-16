# Mind2Web 全方法 7 项统一底座对齐与问题修复清单

> **目的**：解决当前 Mind2Web 实验表中“各方法底座不一致（Top-5 vs Top-50 vs 原始 HTML）”以及“Swarm/Ours 低于 ReAct”的问题。  
> **核心原则**：**控制变量（同一数据切分、同一 Top-50 网页观测、同一历史动作、同一模型参数、同一算分函数）**。所有方法只能在“Agent 协作架构”与“经验/工作流/技能归纳与注入机制”上存在差异，底层环境输入与评测口径必须 100% 对齐。  
> **基准参考实现**：以 `SwarmEvolution/swarmevo/benches/mind2web/dataset.py` 与 `metrics.py` 为唯一权威底座标准。

---

## 一、 全局必须 100% 统一的 7 项底座配置（所有方法共用铁律）

| 序号 | 对齐维度 | 统一标准（以 `SwarmEvolution` 为准） | 当前各仓库存在的不一致问题（必须清零） |
|---|---|---|---|
| **1** | **候选元素数量 (`top_k`)** | **一次性展示 `Top-50` 候选元素**<br>• 统一读取官方 `scores_all_data.pkl`，筛选 `rank < 50` 的候选元素；<br>• 严禁偷看 `pos_candidates` 强行塞入未进 Top-50 的正例；<br>• 严禁分批提前 `break` 丢弃后序候选。 | • **`AWM`**：当前默认 `top_k_elements = 5`（Recall 上限仅 ~55%，且 `env.py:163` 存在直接取 `pos_candidates[:1]` 的逻辑隐患）；<br>• **`SeeAct`**：当前将 Top-50 切成 `16+17+17` 三批，且第 1 批无 `None` 选项导致 `run_multimodal_mind2web.py:264` 在第 1 批就提前 `break`，后 34 个候选永远看不见；<br>• **`PolySkill`**：当前 `run_polyskill_offline.py` 完全未接入 `scores_all_data.pkl`，未做 Top-50 筛选。 |
| **2** | **网页 DOM 裁剪与压缩表示** | **统一使用 `prune_tree` + `get_tree_repr` 生成紧凑 HTML**<br>• 参数固定为：`max_depth=5`, `max_children=50`, `max_sibling=3`, `max_value_length=5`, `max_length=20`, `keep_html_brackets=True`；<br>• 建议直接读取或复用 `SwarmEvolution` 的 `compact_observation()` 输出。 | • **`PolySkill`**：当前 `run_polyskill_offline.py:234` 直接喂入未裁剪的整页原始 `raw_html`（数万字符），且 HTML 内属性是 `backend_node_id="128"`，而 Prompt 却让模型找 `[128]`，导致准确率仅 6.75%；<br>• **`SeeAct`**：除 HTML 外还拼入了每个候选前 10 个词的 `A..P` 选项列表。 |
| **3** | **前序历史动作 (`previous_actions`)** | **统一使用全量 Teacher-Forcing 真实人类历史动作**<br>• 第 `t` 步统一输入 `task["action_reprs"][:target_action_index]`；<br>• 前一步预测错误不影响下一步的历史输入。 | • **`PolySkill`**：当前 `run_polyskill_offline.py:535` 把模型自己上一步预测的错误动作拼进 `trace` 传给下一步（非 Teacher-Forcing，一步错步步错）；<br>• **`AWM`**：当前 `memory.py:83` 把前序步骤的 HTML 观察也拼进多轮对话，导致上下文迅速膨胀。 |
| **4** | **训练/验证/测试数据切分 (`Splits`)** | **测试集 100% 官方全量 + 训练/验证按网站分层切分**<br>• **测试集一题不删**：固定为 Multimodal-Mind2Web 全量 `test_task` (177 题)、`test_website` (142 题)、`test_domain` (694 题)；<br>• **训练集（~320–350 题）**：从官方 73 个训练网站中，按训练任务数选取最多的 **25–28 个网站**作为训练网站（`Seen Websites`），每站留出 2 题做验证，其余 **每站 11–13 题全部用于训练**（不丢弃 60%）；<br>• **验证集（~100 题 = ~50 Seen + ~50 Unseen）**：由上述 25–28 个训练网站每站留出的 2 题（`Seen Val`），加上从剩余未训练网站中整站抽出的 **4–5 个完整网站**（`Unseen Val`）组成；<br>• `AWM`、`PolySkill`、`Ours` 统一使用同一份 `split_path` JSON。 | • **`SwarmEvolution`**：当前 `dataset.py:44` 从 1009 题跨 73 个网站均匀随机抽 30%（303 题），导致平均每站仅 ~4.1 题，且验证集几乎全是已见网站、无未见整站留出。需修改 `make_holdout()` 后重新导出 `split-seed42.json` 并同步给 `AWM` 与 `PolySkill`。 |
| **5** | **输入模态 (`input_mode`)** | **同一张对比表内必须统一模态（纯文本 `text` 或多模态 `multimodal`）**<br>• 纯文本模式：只输入紧凑 HTML + 任务 + 历史动作，不附带截图；<br>• 多模态模式：统一附带当前步骤动作前截图 `screenshot`。 | • **`SwarmEvolution` (`configs/mind2web.yaml`)** 与 **`AWM`** 当前跑的是 **纯文本 (`text`)**；<br>• **`SeeAct`** 与 **`PolySkill`** 当前跑的是 **带截图 (`multimodal`)**。需明确统一。 |
| **6** | **模型与推理超参 (`LLM Config`)** | **统一模型与生成上限**<br>• 同一骨干模型（如 `DeepSeek-V4.1-Flash`）；<br>• `temperature = 0.0`，`thinking = disabled`；<br>• 单次输出上限 `max_tokens >= 4096`，上下文窗口上限 `>= 64000`。 | • **`AWM`**：`memory.py:109` 硬编码了 `MODEL_CONTEXT_WINDOW = 16385`（旧 GPT-3.5 限制），改 Top-50 后会触发超长跳过并直接判 0 分；<br>• **`PolySkill`**：`run_polyskill_offline.py:396` 硬编码了 `max_tokens = 1024`，模型输出 `<think>` 后容易截断 `<action>`。 |
| **7** | **评测算分口径 (`Evaluation Metric`)** | **统一按任务宏平均（Task-level Macro Average）算分**<br>• `Element Acc (EA)`：预测元素原始 `backend_node_id` 命中 `pos_candidates`；<br>• `Operation F1 (AF1)`：`CLICK` 忽略空值/`None`，统一分词与标点/大小写规则；<br>• `Step SR (SSR)`：`EA == 1.0 and AF1 == 1.0`；<br>• `Task SR (SR)`：任务内全部步骤 `SSR == 1.0`；<br>• Top-50 未召回的正例步骤保留在分母中记 0 分。 | • **`SwarmEvolution/swarmevo/benches/mind2web/metrics.py`**：`operation_f1` 区分大小写且**不过滤标点**；<br>• **`AWM` / `SeeAct`**：`token_f1` 区分大小写但**过滤 `string.punctuation`**；<br>• **`PolySkill`**：`operation_f1` **转小写 `.lower()`** 且不过滤标点。<br>👉 必须统一调用同一套 `score_case` 实现！ |

---

## 二、 各代码库具体修改清单（按仓库分工）

### 仓库 1：`PolySkill`（修复 Cross-Task SSR 仅 `5.48%` 的 Bug）
- **责任文件**：`PolySkill/scripts/run_polyskill_offline.py`、`PolySkill/polyskill/prompts/llm_based/prompts_20250723.py`
- **必改项**：
  1. **接入 `scores_all_data.pkl` 与 Top-50 紧凑 DOM**：
     - 不要在 `load_tasks()`（第 234 行）里直接读取未裁剪的 `row["raw_html"]`；
     - 改为与 `SwarmEvolution/swarmevo/benches/mind2web/dataset.py` 完全一致的 `compact_observation(cleaned_html, top50_candidate_ids)`，只传入 Top-50 候选剪枝后的紧凑 HTML（或直接读取 `SwarmEvolution/.cache/mind2web` 中预处理好的 case JSON）。
  2. **对齐元素 ID 格式与解析**：
     - 在紧凑 HTML 中元素带有 `id=0..49`（或原始 `backend_node_id`），确保 Prompt 中对 `bid` 的说明与 HTML 中的 `id=` 属性一致，预测后映射回原始 `backend_node_id` 算分。
  3. **修复历史动作为 Teacher-Forcing**：
     - 在 `run_eval()`（第 516–536 行）中，不要把模型上一步预测的 `raw_action` 累加到 `trace`；改为传入真实前序人类动作 `row["action_reprs"][:target_action_index]`。
  4. **放开输出 Token 限制**：
     - 将 `make_offline_agent()` 第 396 行的 `max_tokens=1024` 调大至 `4096`（或更高），防止 `<think>` 耗尽长度导致解析不到 `<action>`。
  5. **统一算分函数**：
     - `score_step()` 统一对齐 `candidate_recall` 逻辑（若正例不在 Top-50 候选内则记 0 分），`operation_f1` 与主仓库保持完全一致。

---

### 仓库 2：`AWM`（解除 Top-5 召回锁死与 16k 窗口截断）
- **责任文件**：`AWM/mind2web/run_mind2web.py`、`AWM/mind2web/memory.py`、`AWM/mind2web/utils/env.py`、`AWM/mind2web/utils/llm.py`
- **必改项**：
  1. **将候选元素数从 Top-5 对齐到 Top-50**：
     - 将 `run_mind2web.py` 第 53 行的 `--top_k_elements` 默认值从 `5` 改为 `50`（包括 `run_all_benchmarks.py` 和 `run_multimodal_online_benchmarks.py` 中的对应参数）。
  2. **修复 `get_top_k_obs` 的候选选取逻辑（关键！）**：
     - 在 `AWM/mind2web/utils/env.py` 第 160–170 行，原代码写的是 `pos_ids = [c["backend_node_id"] for c in pos_candidates][:1]` 再加 `top_k - 1` 个负例（这会无视正例本身的 rank 强行塞入正例）。
     - 必须改为：**将 `pos_candidates + neg_candidates` 合并后统一按 `c["rank"] < 50` 过滤**（与 `SwarmEvolution` 的 `normalize_task` 完全一致），不根据正负例标签作弊。
  3. **解除 16385 上下文窗口限制**：
     - 在 `AWM/mind2web/memory.py` 第 108–124 行及 `utils/llm.py` 中，将 `MODEL_CONTEXT_WINDOW` 默认值从 `16385` 调大到 `64000`（或 `128000`），避免 Top-50 DOM 变长后触发 `Too many tokens in acting, skipping...` 被直接判 0 分。
  4. **对齐训练切分 `--split_path`**：
     - `offline_induction.py` 使用的 `--split_path` 必须与 `SwarmEvolution` 最终使用的训练任务清单一致。

---

### 仓库 3：`SeeAct`（修复分 3 批在第 1 批提前终止的 Bug，对齐一次性 Top-50）
- **责任文件**：`SeeAct/src/offline_experiments/run_multimodal_mind2web.py`
- **必改项**：
  1. **改为一次性展示 Top-50 候选（不再拆成 3 批提前 `break`）**：
     - 当前 `group_top50()`（第 129–136 行）将 Top-50 切成 `[candidates[:16], candidates[16:33], candidates[33:50]]` 三批，而 `evaluate_one()`（第 263 行）写的是 `if batch["candidate_id"] is not None: break`。由于 Prompt 没有给 `"None of the above"` 选项，模型在第 1 批（页面最顶部的 16 个元素）必然选出一个字母，导致第 2、3 批的 34 个元素永远被跳过！
     - **修改方法**：将 `group_top50(row)` 改为直接返回单批 `[candidates[:50]]`（`parse_answer` 第 146 行已原生支持 `A..Z` 及 `AA..AX` 双字母解析，可直接覆盖 50 个选项），或者直接复用与 `ReAct` 相同的紧凑 DOM + Top-50 列表。

---

### 仓库 4：`SwarmEvolution`（修复 `Swarm < ReAct` 以及 `Ours` 演化不涨分问题）

#### A. 修复零经验 `Swarm` 基线（对齐 `ReAct` 单步推理能力，消除 Verifier 盲审误杀）
- **责任文件**：
  - `SwarmEvolution/data/mind2web/graph/nodes/planner.yaml`
  - `SwarmEvolution/data/mind2web/graph/nodes/executor.yaml`
  - `SwarmEvolution/data/mind2web/graph/nodes/verifier.yaml`
  - `SwarmEvolution/swarmevo/benches/mind2web/adapter.py`
- **必改项**：
  1. **给 `Executor` 增加 `"reason"` 字段（对齐 `ReAct`）**：
     - 在 `executor.yaml` 中允许输出 `{"reason": "...", "element_id": "...", "operation": "CLICK|TYPE|SELECT", "value": "..."}`，让 `Executor` 在输出 `element_id` 之前先用一句话对比候选元素（与 `react.yaml` 对齐）。
  2. **将 `Planner → Executor` 改为 Coarse-to-Fine（粗筛到精选）协作**：
     - 不要在 `planner.yaml` 中禁止 `Planner` 看候选元素；让 `Planner` 结合总任务、`previous_actions` 和当前 DOM，指出当前步骤的子目标以及 **2–3 个最相关的候选元素方向**，`Executor` 在此基础上结合具体标签属性做最终精确选择。
  3. **将 `Verifier` 限制为“静态硬约束审计员”（禁止主观误杀元素 ID）**：
     - 在 `verifier.yaml` 中明确：由于离线快照无页面跳转反馈，`Verifier` **默认放行（`passed: true`）**，**仅当发现以下 3 类确凿静态错误时才允许 `passed: false`**：
       1. `operation` 类型或 `value` 取值明显违反任务要求（如对 `<input>` 用了 `CLICK` 未填值，或填错词）；
       2. 选中的 `element_id` 是外层非交互包装容器（如包裹着按钮的 `<div>/<td>`），且候选列表中存在对应的内层交互子元素（`<button>/<a>/<input>/<select>`）；
       3. 跳过了任务要求但 `previous_actions` 尚未完成的前置条件，或重复了已完成的动作。

#### B. 修复演化算法 `Ours`（对齐论文 `01-formulation.md` 设计）
- **责任文件**：
  - `SwarmEvolution/swarmevo/optimize/epoch_loop.py`
  - `SwarmEvolution/swarmevo/benches/mind2web/bench.py` & `swarmevo/attribute/case.py`
  - `SwarmEvolution/swarmevo/benches/mind2web/dataset.py` & `swarmevo/optimize/selection.py`
  - `SwarmEvolution/swarmevo/optimize/distillation.py` & `swarmevo/runtime/compose.py`
- **必改项**：
  1. **关闭成功样本强行归因**：
     - 在 `epoch_loop.py` 第 227 行，将 `learn_success=True` 改为 `learn_success=False`，只对存在错误步骤（`official_score < 1.0`）的训练任务进行归因，避免往本地层塞入冗余套话。
  2. **训练期归因开放失败步骤的“标准答案对比诊断信号”**：
     - 在训练阶段（非测试阶段），当某一步 `step_success_rate < 1.0` 时，在传给归因器的 `action_metrics` 中提供该失败步骤的真实对比信息：**模型预测的候选元素属性 vs 标准答案正例（`gold` 元素的局部 ID、tag、属性及正确 `operation/value`）**，或者按论文 §0.7 执行单步反事实重放区分 `Planner` 与 `Executor`。杜绝归因器在不知道正确元素是谁的情况下盲猜编造错误规则。
  3. **按论文 §4.1 重构“少网站、多任务”训练切分与“Seen + Unseen”双域验证集**：
     - 修改 `dataset.py` 的 `make_holdout()`：不再将 303 道训练题分散在全部 73 个网站上（平均每站仅 4 题）；改为选取任务最密集的 **25–30 个网站**作为训练网站（每站 12–15 题，留出 2 题作为 $\mathcal{D}^{\text{val}}_{\text{seen}}$），并从剩余网站中整站留出 6–8 个网站作为未见网站验证集 $\mathcal{D}^{\text{val}}_{\text{unseen}}$；
     - 在 `selection.py` 中按论文公式 $J(\Omega) = \frac{1}{2}\text{Score}(\mathcal{D}^{\text{val}}_{\text{seen}}) + \frac{1}{2}\text{Score}(\mathcal{D}^{\text{val}}_{\text{unseen}})$ 计算验证集得分。
  4. **非破坏性提炼与软化规则注入语气**：
     - 在 `distillation.py` 中，当组被提炼为 `SHARED` 且模型未输出某来源网站的 `local` 补充时，默认保留该来源网站的原本土规则，防止因抽象丢失站点专属 DOM 细节；
     - 在 `compose.py` 第 141 行中，将 `"If the current situation matches When, you MUST follow Flow"` 调整为引导式优先级表述，避免近似匹配的规则强行覆盖当前页面真实 DOM 证据。

---

## 三、 验收核对表（改完重跑前逐项打勾）

- [ ] **Check 1**：所有 5 个方法在 `test_task` 上的 `candidate_recall`（Top-50 召回率）是否完全相同（约 `86.4%`）？如果不相同，说明候选集还没对齐。
- [ ] **Check 2**：`AWM`、`PolySkill`、`Ours` 训练时读取的训练任务 ID 列表（`train` split）是否 100% 一致？
- [ ] **Check 3**：零经验 `Swarm` 在开启 `Executor.reason` 并收紧 `Verifier` 后，在 20 个任务的冒烟测试上 `EA` 和 `SSR` 是否已经 $\ge$ `ReAct`？
- [ ] **Check 4**：`Ours` 训练跑完后，检查 `sharing-summary.json` 中三个角色的 `shared_rules` 与 `local_rules` 是否非空，且规则内容包含具体可执行的 DOM/任务判据而非空泛套话？
