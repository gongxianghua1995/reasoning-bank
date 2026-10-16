"""基准必须提供的函数。接 SWE 前向执行时对照这里，不要改 Place。

轨迹事件必须带 carrier_id，否则归因无法打到图上的位置。
"""
from __future__ import annotations

from typing import Any, Protocol

from swarmevo.graph.types import Candidate, Graph, SwarmSkill


class Case:
    __slots__ = ("id", "family_id", "payload")

    def __init__(self, id: str, family_id: str, payload: dict[str, Any] | None = None) -> None:
        self.id = id
        self.family_id = family_id
        if payload is None:
            payload = {}
        self.payload = payload


def classify_timeout_autosubmit(output: Any) -> str:
    """'eval' when a timed-out case still has a patch, 'empty' when it does not.

    Other outputs return ''. A non-empty timeout patch is marked
    ``evaluated_after_timeout`` and keeps ``workflow_status="timeout"``.
    An empty timeout is NO/empty_patch and does not set that flag.
    """
    if not isinstance(output, dict):
        return ""
    status = str(output.get("workflow_status") or "")
    phase = str(output.get("failed_phase") or "")
    if status != "timeout" and phase != "timeout":
        return ""
    if str(output.get("patch") or "").strip():
        output["evaluated_after_timeout"] = True
        output["workflow_status"] = "timeout"
        if not output.get("failed_phase"):
            output["failed_phase"] = "timeout"
        if not output.get("failure_reason"):
            output["failure_reason"] = output.get("exec_error")
        return "eval"
    output["resolved"] = "NO"
    output["empty_patch"] = True
    output["official_score"] = 0.0
    output.pop("evaluated_after_timeout", None)
    if not output.get("workflow_status"):
        output["workflow_status"] = "timeout"
    return "empty"


def case_task_description(case: Case) -> str:
    """Return meaningful task text for semantic routing."""
    for attr in ("description", "problem_statement", "query"):
        value = getattr(case, attr, "")
        if value:
            return str(value)
    if isinstance(case.payload, dict):
        for key in ("spec", "problem_statement", "query", "text"):
            value = case.payload.get(key)
            if value:
                return str(value)
    if case.id:
        return str(case.id)
    raise ValueError("无法为无标签 case 构造语义路由文本")


class TraceEvent:
    """一条轨迹事件。carrier_id 是硬约束。"""

    __slots__ = ("carrier_id", "kind", "data")

    def __init__(self, carrier_id: str, kind: str, data: dict[str, Any] | None = None) -> None:
        self.carrier_id = carrier_id
        self.kind = kind
        if data is None:
            data = {}
        self.data = data


class Trace:
    __slots__ = ("attribution_context", "events")

    def __init__(self, events: list[TraceEvent], attribution_context: dict | None = None) -> None:
        self.events = events
        self.attribution_context = attribution_context or {}


class Bench(Protocol):
    epsilon: float

    def initial_state(self, family_ids: list[str]) -> tuple[Graph, dict[str, SwarmSkill]]:
        """加载该基准真实的初始图和各族 Skill；不允许由运行入口合成替代状态。"""
        ...

    def sample(self, family_id: str, n_cases: int) -> list[Case]:
        """从该族优化集抽 n_cases 条。混域 / sequential 由 loop 选族后再调用。"""
        ...

    def probe_sample(self, family_id: str, n_cases: int) -> list[Case]:
        """从该族探针集取前 n_cases 条。Place 专用，不推进优化游标。"""
        ...

    def test_cases(self, family_id: str) -> list[Case]:
        """该族测试集全部。只在最终汇报时用。"""
        ...

    def validation_cases(self) -> list[Case]:
        """可选的固定完整验证集扩展；没有时由配置显式提供 cases_file。"""
        ...

    def pool_size(self, family_id: str, split: str) -> int:
        """opt / probe / test 各有多少条。"""
        ...

    def split_manifest(self) -> dict[str, Any]:
        """各族三切分的 case id，供落盘与测试 run 对齐。"""
        ...

    def exec(self, plan: Any, case: Case, output_dir: str | None = None) -> tuple[Any, Trace, float]:
        """返回 (产物, 轨迹, 成本)。轨迹每条事件带 carrier_id。output_dir 有则写下执行过程。"""
        ...

    def eval(self, family_id: str, case: Case, output: Any) -> float:
        """产物 → [0, 1]。优化集分数只进 Attr；Place 看的是 gain。"""
        ...

    def gain(self, family_id: str, candidate: Candidate, skills: dict[str, SwarmSkill], graph: Any = None) -> float | None:
        """该 Δ 在族上的分数变化；无探针返回 None，不能把缺测当 0。"""
        ...
