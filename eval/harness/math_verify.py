#!/usr/bin/env python3
"""Answer extraction + equivalence checking for math evaluation.

Two jobs:
  1. extract_final_answer(text): pull the final answer out of a model output.
  2. is_correct(pred, ref): compare two answers with numeric / sympy / set
     equivalence, falling back to normalized string equality.

Supports the usual competition-math answer shapes: integers, fractions,
decimals, percentages, LaTeX (\\frac, \\sqrt, \\boxed), sets, tuples.

Run `python math_verify.py` for a self-test.
"""

from __future__ import annotations

import re
import unicodedata
from fractions import Fraction

try:  # optional on the server; fraction + string fallback still works
    import sympy  # type: ignore
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _SYMPY_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
    _HAS_SYMPY = True
except Exception:  # pragma: no cover - optional dependency
    sympy = None  # type: ignore
    _HAS_SYMPY = False

BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
MARKER_RES = (
    re.compile(r"(?:最终答案|最终结果|答案)\s*[:：]\s*(.+)"),
    re.compile(r"(?:最终答案|最终结果|答案)\s*(?:是|为)\s*(.+)"),
)

LATEX_REPLACEMENTS = (
    (r"\\left|\\right", ""),
    (r"\\,|\\!|\\;|\\ ", ""),
    (r"\\displaystyle", ""),
    (r"\\dfrac|\\tfrac", r"\\frac"),
    (r"\\cdot|\\times", "*"),
    (r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)"),
    (r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),
    (r"\\pi", "pi"),
    (r"\\%", "%"),
    (r"\\text\s*\{([^{}]*)\}", r"\1"),
    (r"\\boxed", ""),
    (r"\\[a-zA-Z]+", ""),  # drop remaining commands
)


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.rstrip("。.;；,，")
    return text.strip()


def _latex_to_plain(text: str) -> str:
    out = text
    for pattern, repl in LATEX_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    return out.strip()


def _answer_core(text: str) -> str:
    """Strip a leading single-variable assignment, e.g. `x=5` -> `5`."""
    text = _strip_wrappers(text)
    m = re.fullmatch(r"[a-zA-Z]\s*=\s*(.+)", text)
    return m.group(1).strip() if m else text


def normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _latex_to_plain(text)
    text = text.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    text = re.sub(r"\s+", "", text)
    text = _strip_wrappers(text)
    text = text.lower()
    return text


def extract_final_answer(text: str) -> str:
    """Extract the final answer string; returns "" when nothing is found."""
    if not text:
        return ""
    boxed = BOXED_RE.findall(text)
    if boxed:
        return _strip_wrappers(boxed[-1])
    best = ""
    for pattern in MARKER_RES:
        matches = pattern.findall(text)
        if matches:
            best = matches[-1]
    if best:
        return _strip_wrappers(best)
    # last non-empty line is the usual fallback
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return _strip_wrappers(lines[-1])
    return ""


def _to_fraction(text: str) -> Fraction | None:
    s = normalize_answer(text)
    if not s:
        return None
    s = re.sub(r"\(([-\d.]+)\)", r"\1", s)  # (1)/(2) -> 1/2
    if s.endswith("%"):
        try:
            return Fraction(s[:-1]) / 100
        except (ValueError, ZeroDivisionError):
            pass
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return Fraction(m.group(1)) / Fraction(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if m:
        try:
            return Fraction(s)
        except ValueError:
            return None
    return None


def _split_set(text: str) -> list[str] | None:
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [part.strip() for part in inner.split(",")]


def _sympy_equal(a: str, b: str) -> bool:
    if not _HAS_SYMPY:
        return False
    try:
        expr_a = parse_expr(_latex_to_plain(a), transformations=_SYMPY_TRANSFORMS)
        expr_b = parse_expr(_latex_to_plain(b), transformations=_SYMPY_TRANSFORMS)
        return bool(sympy.simplify(expr_a - expr_b) == 0)
    except Exception:  # noqa: BLE001 - parse_expr is best-effort
        return False


def is_correct(pred: str | None, ref: str | None) -> bool:
    """True when the predicted answer equals the reference answer."""
    if not pred or not ref:
        return False

    pred_core = _answer_core(str(pred))
    ref_core = _answer_core(str(ref))
    norm_pred = normalize_answer(pred_core)
    norm_ref = normalize_answer(ref_core)
    if not norm_pred or not norm_ref:
        return False
    if norm_pred == norm_ref:
        return True

    # numeric equivalence (fractions, decimals, percentages)
    frac_pred = _to_fraction(pred_core)
    frac_ref = _to_fraction(ref_core)
    if frac_pred is not None and frac_ref is not None:
        return frac_pred == frac_ref

    # set equivalence, order-insensitive
    set_pred = _split_set(norm_pred)
    set_ref = _split_set(norm_ref)
    if set_pred is not None and set_ref is not None:
        return sorted(set_pred) == sorted(set_ref)

    # tuple / interval comparison element-wise
    for open_b, close_b in (("(", ")"), ("[", "]")):
        if norm_pred.startswith(open_b) and norm_pred.endswith(close_b) and \
                norm_ref.startswith(open_b) and norm_ref.endswith(close_b):
            parts_pred = [p.strip() for p in norm_pred[1:-1].split(",")]
            parts_ref = [p.strip() for p in norm_ref[1:-1].split(",")]
            if len(parts_pred) == len(parts_ref):
                return all(
                    is_correct(p, r) or normalize_answer(p) == normalize_answer(r)
                    for p, r in zip(parts_pred, parts_ref)
                )

    return _sympy_equal(pred_core, ref_core)


def verify(pred_text: str, ref_answer: str) -> dict[str, object]:
    """Convenience wrapper: extract from generation text then compare."""
    pred = extract_final_answer(pred_text)
    return {
        "pred": pred,
        "ref": ref_answer,
        "correct": is_correct(pred, ref_answer),
        "empty": not bool(pred),
    }


if __name__ == "__main__":
    cases = [
        ("答案是 42", "42", True),
        ("\\boxed{\\frac{1}{2}}", "0.5", True),
        ("最终答案：13", "13", True),
        ("所以答案是 3/4", "0.75", True),
        ("{2, 1, 3}", "{1,2,3}", True),
        ("x=5", "5", True),
        ("答案是 7", "8", False),
        ("答案：2\\sqrt{2}", "\\sqrt{8}", True),
        ("答案：完全平方数", "完全平方数", True),
    ]
    failed = 0
    for text, ref, expected in cases:
        got = is_correct(extract_final_answer(text), ref)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"[{status}] pred={extract_final_answer(text)!r} ref={ref!r} "
              f"correct={got} expected={expected}")
    if failed:
        raise SystemExit(f"{failed} case(s) failed")
    print("math_verify.py self-test OK")
