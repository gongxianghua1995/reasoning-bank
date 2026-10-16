"""图谱上的数据类型。对应 01 §2。

字段名尽量跟论文一致（R、S、z），避免两套词汇。读代码时看本文件注释即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Decision = Literal["Shared", "Local", "Reject"]
# 协作边类型 T。身份是 (src, dst, type)，同两端不同型是两条边（01 §2.2）。
# forward：depends_on，正常往下交；feedback：on_fail.retry_role，退回重做；
# escalate：交给更高层（词表里有，G0 的 workflow 还没抽这类边）。
EdgeType = Literal["forward", "feedback", "escalate"]

# 单元测试兼容用的最小 Executor id；真实 SWE G0 使用三个显式共享节点。
DEFAULT_EXECUTOR = "shared.executor"
Op = Literal["add", "edit", "del"]
ZKind = Literal["experience", "carrier", "prompt_rule"]
AttrTargetKind = Literal[
    "prompt_rule",
    "new_prompt_rule",
    "experience",
    "new_experience",
    "workflow",
    "gap",
]
IssueState = Literal["active", "dormant", "consumed"]
# 归因给出的“修复类型”。一级分类，决定 AttrItem 走哪一级演化以及 Tier 3 内部走哪条通道：
# content_edit → Tier 1/2；gate_config → Tier 3 通道 A（确定性 ChangeGate/ChangeRetry）；
# topo_surgery / gap_fill → Tier 3 通道 B（LLM 两阶段 grow）；runtime_defect → 不进任何缓冲区，
# 单独落盘 runtime_defects.jsonl（引擎 bug，任何 Tier 都修不了）。
FixClass = Literal["content_edit", "gate_config", "topo_surgery", "gap_fill", "runtime_defect"]
FIX_CLASSES: tuple[str, ...] = ("content_edit", "gate_config", "topo_surgery", "gap_fill", "runtime_defect")
# topo_surgery / gap_fill 的二级提示：限制 grow() 编译器允许的算子集合。
TopoOp = Literal["add_data_edge", "add_step", "add_role", "add_fail_path", "rewrite_fallback", "reorder_steps"]
TOPO_OPS: tuple[str, ...] = ("add_data_edge", "add_step", "add_role", "add_fail_path", "rewrite_fallback", "reorder_steps")
# gate_config 的二级提示 gate_patch 里 field 的合法取值（对应 workflow.yaml 每个 step 的 gate 子树）。
GATE_PATCH_FIELDS: tuple[str, ...] = (
    "assert",
    "required_fields",
    "on_fail.action",
    "on_fail.target_step",
    "on_fail.max_attempts",
    "on_fail.max_rounds",
)


def make_edge_id(src: str, dst: str, edge_type: str) -> str:
    """边身份 (src, dst, type) 的稳定字符串。01 §2.2：同名不同型是两条边。"""
    return f"{src}->{dst}:{edge_type}"


# 归因打不到图上任何元素时，slot.z 取这个值。01 记作 ⊥，交给 Grow / Structure 优化。
UNATTRIBUTED_SLOT_Z = "bot"
# 提示词槽。兼容向后引用。
PROMPT_SLOT_Z = "pi"


@dataclass(frozen=True)
class Slot:
    """落位单元：改哪个载体上的哪一处。库、Order、Place 都按这个键走。

    模型侧写 position（prompt / 规则 id / 经验 id / gap），入库映射到 z（rule_id / exp_id / pi / bot）。
    """

    carrier_id: str
    z: str  # 规则 id / 经验 id，或 PROMPT_SLOT_Z，或 'carrier'，或 UNATTRIBUTED_SLOT_Z

    def is_unattributed(self) -> bool:
        """症状找不到该怪谁。"""
        return self.z == UNATTRIBUTED_SLOT_Z

    def is_prompt(self) -> bool:
        """打在常驻提示词/规则上，不是一条情景经验。"""
        return self.z == PROMPT_SLOT_Z or self.z.startswith("rule_") or self.z.startswith("rule-")


@dataclass
class StandingRule:
    """一条常驻硬性规则（无条件始终生效的约束、红线或职责边界）。"""

    id: str
    text: str


@dataclass
class Experience:
    """一条情景经验 (何时/触发条件, 具体操作, 检查断言, 领域实例)。
    
    支持结构化经验契约 (Structured Experience Contract):
    - trigger: 可观测触发条件 (报错特征、diff特征、工具返回等)
    - action: 具体操作规程 (命令范式、工具调用序列)
    - assertion: 检查断言与成功判据
    - domain_instantiations: 跨领域具体实例映射 {domain: specific_instance}
    - when / flow: 经典/兼容字段
    """

    id: str
    when: str = ""  # 何时触发 (When-condition / trigger)
    flow: str = ""  # 做什么 (Action/Flow SOP)
    trigger: str = ""  # 结构化触发条件
    action: str = ""  # 结构化具体操作
    assertion: str = ""  # 结构化检查断言
    domain_instantiations: dict[str, str] = field(default_factory=dict)  # 各领域具体实例 {family_id: instance_text}

    def __post_init__(self) -> None:
        # 兼容性双向同步：
        # 如果提供了结构化 trigger/action 但未提供 when/flow，自动同步
        if (self.trigger or self.action or self.assertion) and not (self.when and self.flow):
            if not self.when:
                self.when = self.trigger.strip()
            if not self.flow:
                parts = []
                if self.action:
                    parts.append(f"Action: {self.action.strip()}")
                if self.assertion:
                    parts.append(f"Verify Assertion: {self.assertion.strip()}")
                self.flow = " | ".join(parts)
        # 反向：如果提供了 when/flow 但未提供 trigger/action
        elif (self.when or self.flow) and not (self.trigger or self.action):
            if not self.trigger:
                self.trigger = self.when.strip()
            if not self.action:
                self.action = self.flow.strip()


@dataclass
class Node:
    id: str
    description: str = ""
    pi: str = ""  # 基础不可变骨架 prompt (π_skeleton)
    rules: dict[str, StandingRule] = field(default_factory=dict)  # 共享常驻规则库 (Π_rules)
    psi_in: tuple[str, ...] = ()  # 输入接口
    psi_out: tuple[str, ...] = ()  # 输出接口
    R: dict[str, Experience] = field(default_factory=dict)  # 共享情景经验集 (R)
    # 来源标记：""（冷启动/技能目录）| "blueprint"（LLM 蓝图创生且 prompt 干净）| "local_only"
    # （模板兜底或 prompt 含议题原文；仍注册以保持技能绑定合法，但不进入共享人才库）
    # "controlled-sharing" 表示 S_004 固定模板，同时让 checkpoint 保留两层经验的渲染契约。
    origin: str = ""

    @property
    def shared_rules(self) -> dict[str, StandingRule]:
        return self.rules

    @property
    def shared_experiences(self) -> dict[str, Experience]:
        return self.R


@dataclass
class Edge:
    """协作边。身份是 (src, dst, type)；共享内容只有 R^ctrl。交接字段在两端节点的 ψ。"""

    src: str
    dst: str
    type: EdgeType
    description: str = ""
    R_ctrl: dict[str, Experience] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return make_edge_id(self.src, self.dst, self.type)

    @property
    def shared_experiences(self) -> dict[str, Experience]:
        return self.R_ctrl


@dataclass
class Overlay:
    """领域局部特化层 (Domain Specialization Λ_f)：单领域的专有附加经验与局部常驻规则插槽（非破坏性增量拼装，非覆盖覆写）。"""

    rules_loc: dict[str, StandingRule] = field(default_factory=dict)  # 领域私有常驻规则 (Π_rules_loc)
    R_loc: dict[str, Experience] = field(default_factory=dict)  # 领域私有情景经验 (R_loc)
    pi_loc: str = ""  # 填进共享骨架 $local_prompt；G0 从 roles.md 读入
    max_steps: int | None = None  # 可选本地覆盖；真实 G0 不设置角色级预算

    @property
    def local_rules(self) -> dict[str, StandingRule]:
        return self.rules_loc

    @property
    def local_experiences(self) -> dict[str, Experience]:
        return self.R_loc


@dataclass
class SwarmSkill:
    """领域团队蓝图 (Domain SwarmSkill κ_f)：严格对齐标准 Agent Skill 规范（name & description），绑定角色集 V_f、工作流拓扑 E_f、领域特化 overlays 及私有 WorkflowSpec。"""

    family_id: str
    name: str = ""  # 标准 Skill 名称（如 "swe-bench-team"）
    description: str = ""  # 标准 Skill 描述契约（从 skill.md frontmatter 读取，供自主语义路由与动态反哺）
    node_ids: list[str] = field(default_factory=list)  # 本族绑定的节点 V_f
    edge_ids: list[str] = field(default_factory=list)  # 本族拓扑 E_f，来自 WorkflowSpec；不从并集边推断
    overlays: dict[str, Overlay] = field(default_factory=dict)  # carrier_id → 本族附加
    skill_dir: str = ""  # G0 目录：skill.md / workflow.yaml / roles / workflow.aop.yaml
    workflow: Any = None  # 本族独立 clone 的 WorkflowSpec；结构优化只改这里

    @property
    def id(self) -> str:
        return self.family_id


@dataclass
class Graph:
    """G = (V, E, Θ)。nodes / edges 上的共享内容即 Θ。"""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)


@dataclass
class AttrItem:
    """例级反馈：一条失败轨迹对某个编辑区域的批评。

    target 只说改哪里，不给新增经验起名。所有 AttrItem 都来自坏 case，
    所以不再保存恒为负的 sign。
    - theme: 归因问题类型/因果主题标签 (例如 "missing_test_verification", "locate_timeout", "gate_false_positive", "tool_misuse")，供跨载体共识聚类
    """

    target: "AttrTarget"
    symptom: str  # 这个位置错在哪；不是轨迹现象，也不是补丁
    family_id: str
    evidence: str = ""  # 轨迹上对得上的那一段
    scope: Literal["local", "shared"] = "local"  # 缺陷责任作用范围 (local 局部缺失 vs shared 全局规则误导)
    case_id: str = ""  # 归因来源；实例级适应必须按 case 精确隔离
    theme: str = ""  # 问题类型/因果主题标签
    # 修复类型（见 FixClass）。旧数据为空串，由 _parse_attr_row 按 target.kind 补默认值。
    fix_class: str = ""
    # 仅 topo_surgery / gap_fill：建议的拓扑算子（见 TopoOp）。
    topo_op: str = ""
    # 仅 gate_config：{"step_id": str, "field": GATE_PATCH_FIELDS 之一, "value": Any}。
    gate_patch: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRecord:
    """真实用例单次执行历史轨迹记录 (Execution Record in Replay Buffer H)。"""

    case_id: str
    family_id: str
    case: Any
    trace: Any
    score: float
    output: Any = None
    cost_tokens: int = 0
    duration_seconds: float = 0.0
    timestamp: float = 0.0


class ReplayBuffer:
    """全局真实执行历史回放池 (Execution History Buffer H)。

    在前向优化执行过程中动态记录每个用例的四元组 (x_i, kappa_k, tau_i, s_i)。
    在周期性共识提拔与争议降级时，直接提取使用过相关角色/技能的真实历史记录作为靶向回放检验集 H_rel。
    """

    def __init__(self, max_records: int = 500) -> None:
        self.max_records = max_records
        self._records: list[ExecutionRecord] = []

    def add(self, record: ExecutionRecord) -> None:
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records.pop(0)

    def all_records(self) -> list[ExecutionRecord]:
        return list(self._records)

    def relevant_records(
        self,
        family_ids: set[str] | frozenset[str] | list[str] | None = None,
        max_k: int | None = None,
    ) -> list[ExecutionRecord]:
        """获取与指定领域技能相关的真实历史记录子集 H_rel。"""
        if family_ids is None:
            records = list(self._records)
        else:
            fset = set(family_ids)
            records = [r for r in self._records if r.family_id in fset]
        if max_k is not None and len(records) > max_k:
            records = records[-max_k:]
        return records

    def failed_records(
        self,
        family_id: str | None = None,
    ) -> list[ExecutionRecord]:
        """获取历史失败样本集 H_fail (score < 1.0)，用于结构重构与定向自愈。"""
        records = [r for r in self._records if r.score < 1.0]
        if family_id is not None:
            records = [r for r in records if r.family_id == family_id]
        return records

    def sensitive_records_for_carrier(
        self,
        carrier_id: str,
        family_id: str,
        mode: str = "related",
        max_k: int = 1,
    ) -> list[ExecutionRecord]:
        """获取指定领域针对特定载体的敏感回放历史记录。

        - mode == 'proposer' (提出域): 优先选择历史做错的 (score < 1.0)，按与 carrier_id 交互深度排序，用于验证修复有效性与语义提炼保真度；
        - mode == 'related' (关联域): 优先选择历史做对或临界通过的 (score >= 1.0 或 0 < score < 1.0)，按脆弱度/敏感度打分排序，用于排查跨域破坏与负迁移。
        """
        family_records = [r for r in self._records if r.family_id == family_id]
        if not family_records:
            return []

        def _is_carrier_involved(r: ExecutionRecord) -> bool:
            if carrier_id in ("workflow", "gap"):
                return True
            events = getattr(r.trace, "events", []) if hasattr(r.trace, "events") else []
            if events:
                return any(getattr(e, "carrier_id", "") == carrier_id for e in events)
            return True

        carrier_records = [r for r in family_records if _is_carrier_involved(r)]
        if not carrier_records:
            carrier_records = family_records

        if mode == "proposer":
            failed = [r for r in carrier_records if r.score < 1.0]
            pool = failed if failed else carrier_records

            def _proposer_depth_score(r: ExecutionRecord) -> float:
                events = getattr(r.trace, "events", []) if hasattr(r.trace, "events") else []
                carrier_steps = sum(1 for e in events if getattr(e, "carrier_id", "") == carrier_id)
                return float(carrier_steps) * 2.0 + (1.0 - r.score) * 5.0

            sorted_pool = sorted(pool, key=_proposer_depth_score, reverse=True)
            return sorted_pool[:max_k]
        else:
            successful = [r for r in carrier_records if r.score >= 1.0]
            pool = successful if successful else carrier_records

            def _related_sensitivity_score(r: ExecutionRecord) -> float:
                score = 0.0
                events = getattr(r.trace, "events", []) if hasattr(r.trace, "events") else []
                retries = sum(
                    1
                    for e in events
                    if getattr(e, "kind", "") in ("retry", "rework", "feedback", "gate_fail", "aop_retry")
                )
                score += retries * 5.0
                warnings = sum(
                    1
                    for e in events
                    if getattr(e, "kind", "") in ("aop_warning", "aop_violation", "warning")
                )
                score += warnings * 3.0
                carrier_steps = sum(1 for e in events if getattr(e, "carrier_id", "") == carrier_id)
                score += float(carrier_steps) * 1.5
                if r.cost_tokens > 2000 or r.duration_seconds > 30.0:
                    score += 2.0
                if 0.0 < r.score < 1.0:
                    score += 4.0
                return score

            sorted_pool = sorted(pool, key=_related_sensitivity_score, reverse=True)
            return sorted_pool[:max_k]

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True)
class DefectRecord:
    """全局共享节点/规则的负向缺陷与争议证据打点。"""

    carrier_id: str
    rule_id: str
    family_id: str
    symptom: str
    evidence: str = ""
    target_kind: str = "experience"
    case_id: str = ""
    timestamp: float = 0.0


class SharedDefectLedger:
    """共享缺陷看板 (Shared Defect Ledger B_shared_neg)。

    仅记录已有全局规则/经验引发的冲突与负向反馈证据，当场不改动
    全局参数，留待周期级对账统一处理争议降级与剪枝。新增内容缺口
    (new_prompt_rule/new_experience) 不属于本账本。
    """

    def __init__(self) -> None:
        self._defects: list[DefectRecord] = []

    def record_defect(
        self,
        carrier_id: str,
        rule_id: str,
        family_id: str,
        symptom: str,
        evidence: str = "",
        target_kind: str = "prompt_rule",
        case_id: str = "",
    ) -> None:
        if target_kind in ("new_prompt_rule", "new_experience"):
            raise ValueError(
                f"SharedDefectLedger only accepts existing shared content; got {target_kind}"
            )
        if target_kind not in ("prompt_rule", "experience"):
            raise ValueError(f"Unsupported shared defect target kind: {target_kind}")
        if not rule_id or rule_id == target_kind:
            raise ValueError(
                "SharedDefectLedger requires the concrete ID of existing shared content"
            )
        self._defects.append(
            DefectRecord(
                carrier_id=carrier_id,
                rule_id=rule_id,
                family_id=family_id,
                symptom=symptom,
                evidence=evidence,
                target_kind=target_kind,
                case_id=case_id,
            )
        )

    def defects_for(
        self,
        carrier_id: str,
        rule_id: str | None = None,
        target_kind: str | None = None,
    ) -> list[DefectRecord]:
        return [
            d
            for d in self._defects
            if d.carrier_id == carrier_id
            and (rule_id is None or d.rule_id == rule_id)
            and (target_kind is None or d.target_kind == target_kind)
        ]

    def find_high_negative_contents(
        self, threshold: int = 1
    ) -> list[tuple[str, str, str, int]]:
        """Return disputed existing content, preserving its exact content type."""
        counts: dict[tuple[str, str, str], int] = {}
        for d in self._defects:
            key = (d.carrier_id, d.rule_id, d.target_kind)
            counts[key] = counts.get(key, 0) + 1
        high_neg = [
            (carrier_id, rule_id, target_kind, count)
            for (carrier_id, rule_id, target_kind), count in counts.items()
            if count >= threshold
            and rule_id not in ("", PROMPT_SLOT_Z, UNATTRIBUTED_SLOT_Z)
        ]
        high_neg.sort(key=lambda row: row[3], reverse=True)
        return high_neg

    def find_high_negative_rules(self, threshold: int = 1) -> list[tuple[str, str, int]]:
        """兼容旧调用：按 ID 汇总，不应用于需要区分内容类型的执行路径。"""
        counts: dict[tuple[str, str], int] = {}
        for d in self._defects:
            counts[(d.carrier_id, d.rule_id)] = counts.get((d.carrier_id, d.rule_id), 0) + 1
        high_neg = [
            (carrier_id, rule_id, cnt)
            for (carrier_id, rule_id), cnt in counts.items()
            if cnt >= threshold and rule_id not in ("", PROMPT_SLOT_Z, UNATTRIBUTED_SLOT_Z)
        ]
        high_neg.sort(key=lambda x: x[2], reverse=True)
        return high_neg

    def consume_defects(
        self, carrier_id: str, rule_id: str, target_kind: str | None = None
    ) -> None:
        """争议规则降级或修复后消费移除相关缺陷证据。"""
        self._defects = [
            d
            for d in self._defects
            if not (
                d.carrier_id == carrier_id
                and d.rule_id == rule_id
                and (target_kind is None or d.target_kind == target_kind)
            )
        ]

    def all_defects(self) -> list[DefectRecord]:
        return list(self._defects)

    def __len__(self) -> int:
        return len(self._defects)


@dataclass(frozen=True)
class AttrTarget:
    """归因指出的编辑区域，还不是 Candidate 最终写入的 Slot。

    三层正交目标定位：
    - prompt / prompt_rule / new_prompt_rule: 角色常驻提示词规则 (无条件红线/约束)
    - experience / new_experience: 角色情景经验库 (When/Flow SOP 与避坑)
    - workflow / gap: 团队工作流编排、步骤质量门或拓扑缺失 (Workflow DAG / Gate / Role Gap)
    """

    carrier_id: str | None
    kind: AttrTargetKind
    experience_id: str | None = None  # 也作为 rule_id 兼容传递

    def __post_init__(self) -> None:
        if self.kind in ("gap", "workflow") and self.carrier_id not in (None, "workflow"):
            # 允许 carrier_id 为 None 或 "workflow"
            pass
        elif self.kind not in ("gap", "workflow") and not self.carrier_id:
            raise ValueError(f"{self.kind} 必须有 carrier_id")

    @property
    def rule_id(self) -> str | None:
        return self.experience_id

    def provisional_slot(self) -> Slot:
        """给排序和日志用。"""
        if self.kind in ("gap", "workflow"):
            return Slot(UNATTRIBUTED_SLOT_Z, UNATTRIBUTED_SLOT_Z)
        if self.kind in ("prompt_rule", "new_prompt_rule"):
            return Slot(str(self.carrier_id), self.experience_id or PROMPT_SLOT_Z)
        if self.kind in ("experience", "new_experience"):
            return Slot(str(self.carrier_id), self.experience_id or "experience")
        return Slot(str(self.carrier_id or UNATTRIBUTED_SLOT_Z), UNATTRIBUTED_SLOT_Z)


@dataclass(frozen=True)
class IssueAttempt:
    """一次没被接受的改稿。新证据到来前不再重试同一稿。"""

    round_index: int
    evidence_version: int
    candidate_hash: str
    candidate_text: str
    gains: dict[str, float]
    reject_reason: str


@dataclass
class GradientIssue:
    """同一编辑区域里一组语义相同的例级反馈，即可排序的文本梯度。

    issue_id 是库内稳定身份；evidence_ids 精确控制消费范围。
    """

    issue_id: str
    target: AttrTarget
    summary: str
    families: frozenset[str]
    evidence_ids: frozenset[int]
    evidences: tuple[str, ...] = ()
    state: IssueState = "active"
    evidence_version: int = 1
    attempts: tuple[IssueAttempt, ...] = ()
    # 由成员 AttrItem 聚合而来：fix_class 取多数，topo_op 取多数，gate_patch 取首个非空。
    fix_class: str = ""
    topo_op: str = ""
    gate_patch: dict[str, Any] = field(default_factory=dict)

    @property
    def strength(self) -> float:
        # 梯度强度：综合考虑案例数量与跨域共鸣广度
        return float(len(self.evidence_ids) + max(0, len(self.families) - 1))

    @property
    def slot(self) -> Slot:
        """兼容日志和 Ref 计算；新增经验/规则这里只是占位，不用于写回。"""
        return self.target.provisional_slot()


# 代码沿用 TextGradient 这个公开名字；它现在就是持久语义议题。
TextGradient = GradientIssue


@dataclass
class Candidate:
    """一次待判决的更新步（把文本梯度写成具体改动）。落位读提出族和 z_kind。"""

    slot: Slot
    op: Op
    payload: Experience | StandingRule | Any
    S: frozenset[str]  # 提出族；丢掉 Place 就分不出 Shared / Local
    z_kind: ZKind = "experience"
    structure: dict[str, Any] | None = None  # 结构 op；内容候选必须是 None
    intent: Literal["shared", "local"] = "shared"  # 提议意图：通用共享能力 vs 本域特化知识
    rationale: str = ""  # 改动理由与论证
    action_type: Literal["local", "promote", "revise", "demote", "restructure"] = "local"  # 对应三层自演化动作类型
    source_case_id: str = ""  # 实例级 Local 更新的来源 case；普通 Place 候选为空
    source: str = ""  # instance_adaptation / place / consensus / restructure
    theme: str = ""  # 归因因果主题标签
    # Stage-2 promotion 的精确来源：(family_id, z_kind, payload_id)。
    source_local_keys: tuple[tuple[str, str, str], ...] = ()

    @property
    def proposer_families(self) -> frozenset[str]:
        return self.S

    def is_structure_candidate(self) -> bool:
        """改的是载体存废或 WorkflowSpec，不是一条经验或规则。永不进 Shared。"""
        return self.z_kind == "carrier" or self.structure is not None


def validate_skill_in_graph(graph: Graph, skill: SwarmSkill) -> bool:
    """检查 SwarmSkill 绑定的节点和边是否均存在于共享图谱中。"""
    for node_id in skill.node_ids:
        if node_id not in graph.nodes:
            return False
    for edge_id in skill.edge_ids:
        if edge_id not in graph.edges:
            return False
    return True

