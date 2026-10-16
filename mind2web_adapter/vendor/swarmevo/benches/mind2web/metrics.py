"""Mind2Web multichoice metrics, including its candidate-recall ceiling.

Reference: OSU-NLP-Group/Mind2Web/src/action_prediction/metric.py.
Operation F1 uses case-sensitive sets of whitespace-separated tokens.
"""

from statistics import mean


def valid_action(action: object) -> bool:
    return (
        isinstance(action, dict)
        and isinstance(action.get("element_id"), str)
        and bool(action["element_id"])
        and action.get("operation") in {"CLICK", "TYPE", "SELECT"}
        and isinstance(action.get("value"), str)
    )


def operation_text(operation: str, value: str) -> str:
    # The official action postprocessor captures Value with a single-line regex.
    first_line = value.split("\n", 1)[0].strip()
    return operation + (" " + first_line if operation != "CLICK" else "")


def operation_f1(prediction: str, target: str) -> float:
    predicted, expected = set(prediction.strip().split()), set(target.strip().split())
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    return 2 * len(predicted & expected) / (len(predicted) + len(expected))


def score_step(action: object, step: dict) -> dict:
    candidates = {item["element_id"] for item in step["raw"]["candidates"]}
    positives = set(step["gold"]["element_ids"])
    recalled = bool(candidates & positives)
    result = {
        "element_accuracy": 0.0,
        "operation_f1": 0.0,
        "step_success_rate": 0.0,
        "candidate_recall": float(recalled),
        "invalid_action": not valid_action(action),
    }
    # The official multichoice path assigns all metrics zero when no positive
    # survived top-k filtering, or when there is no valid final prediction.
    if not recalled or not valid_action(action) or action["element_id"] not in candidates:
        return result
    element = float(action["element_id"] in positives)
    gold = step["gold"]
    f1 = operation_f1(
        operation_text(action["operation"], action["value"]),
        operation_text(gold["operation"], gold["value"]),
    )
    result.update(
        element_accuracy=element, operation_f1=f1, step_success_rate=float(element == 1 and f1 == 1)
    )
    return result


def score_case(predictions: list, task: dict) -> dict:
    if len(predictions) != len(task["steps"]):
        raise ValueError("predictions must cover every annotated step")
    steps = [
        score_step(action, step) for action, step in zip(predictions, task["steps"], strict=True)
    ]
    if not steps:
        raise ValueError("Mind2Web case has no steps")
    metrics = {
        key: mean(row[key] for row in steps)
        for key in ("element_accuracy", "operation_f1", "step_success_rate", "candidate_recall")
    }
    return {
        **metrics,
        "task_success_rate": float(all(row["step_success_rate"] == 1 for row in steps)),
        "steps": steps,
        "n_steps": len(steps),
    }


def aggregate(cases: list[dict]) -> dict:
    if not cases:
        return {"n_cases": 0, "n_steps": 0}
    return {
        **{
            key: mean(case[key] for case in cases)
            for key in (
                "element_accuracy",
                "operation_f1",
                "step_success_rate",
                "task_success_rate",
                "candidate_recall",
            )
        },
        "n_cases": len(cases),
        "n_steps": sum(case["n_steps"] for case in cases),
    }
