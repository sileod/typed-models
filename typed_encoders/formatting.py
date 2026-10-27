"""Decision Index text rendering (the kit's own `text`), shared by the data builders."""

from typing import Any

from decision_index.engines.base import text


def state_text(state: Any) -> str:
    return text(state) if state not in ("", None, {}, []) else "(empty)"


def format_option(key: str, desc: Any) -> str:
    desc_str = text(desc).strip() if desc is not None else ""
    key_str = str(key).strip()
    if not desc_str:
        return key_str
    # If key is an artificial index, label, hash or bracketed letter, use description directly
    is_dummy_key = (
        key_str.lower().startswith(("option_", "k_", "choice_", "item_"))
        or (len(key_str) == 1 and key_str.isalpha())
        or (key_str.startswith("(") and key_str.endswith(")") and len(key_str) <= 4)
    )
    if is_dummy_key:
        return desc_str
    if desc_str.lower() == key_str.lower():
        return key_str
    return f"{key_str}: {desc_str}"


def question_options(q: dict) -> tuple[list[str], list[str]]:
    """(answer keys, rendered option texts) for a choice or noul question, as the official engines see it."""
    q_type = q.get("type", "choice")
    crit = q.get("criteria", {}) or {}
    if q_type == "choice":
        keys = list(crit)
        return keys, [format_option(k, crit[k]) for k in keys]
    if q_type == "noul":
        descs = (crit.get("false", crit.get("False", "False")), crit.get("true", crit.get("True", "True")))
        return ["false", "true"], [format_option("false", descs[0]), format_option("true", descs[1])]
    raise ValueError(f"unsupported question type: {q_type}")
