#!/usr/bin/env python3
"""Shared text cleaning / normalization helpers for the RECAST data pipeline.

Kept dependency-free so it can also be imported by the harness and by ad-hoc
inspection scripts.

Terms that must not appear in a *training* rationale:
  - DIALOGUE_TERMS: the teacher-student dialogue is only a presentation shell.
  - META_TERMS: the rationale must not describe the reconstruction pipeline.
"""

from __future__ import annotations

import re
import unicodedata

# The teacher answer is a dialogue shell; these words must not leak into the
# reconstructed rationale.
DIALOGUE_TERMS = ("老师", "学生", "同学", "课堂", "师生", "对话")

# Reconstruction-pipeline meta talk.
META_TERMS = (
    "用户提供了",
    "我被要求",
    "思维链还原",
    "还原工程师",
    "解码模型",
    "闭源模型的完整回答",
    "本次任务",
    "长度要求",
    "角色",
    "分角色",
    "呈现",
    "模拟",
    "还原",
    "解码",
)

# Calibrated on DeepSeek usage for Chinese-heavy text (see docs/data_pipeline.md).
TOKENS_PER_CHAR = 0.635

TITLE_LINE_RE = re.compile(r"^\s*#{1,4}\s*(思维链还原|思维链|CoT|思考过程|推理过程)\s*$", re.I)
BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
ANSWER_LINE_RE = re.compile(r"(?:最终答案|最终结果|答案)\s*[:：]\s*(.+)")
CONCLUSION_RE = re.compile(r"(?:因此|所以|故|从而)[，,]?\s*(.+?)\s*[。.]?\s*$")
NUMERIC_RE = re.compile(r"^[\(\-\+\$\\]*\s*-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?\s*[\)\$%]*$")


def estimate_tokens(text: str, tokens_per_char: float = TOKENS_PER_CHAR) -> int:
    """Cheap token estimate for Chinese-heavy text (used only for length floors)."""
    if not text:
        return 0
    return max(1, round(len(text) * tokens_per_char))


def find_terms(text: str, terms: tuple[str, ...] = DIALOGUE_TERMS) -> list[str]:
    return [term for term in terms if term in text]


def find_all_leaks(text: str) -> dict[str, list[str]]:
    return {
        "dialogue": find_terms(text, DIALOGUE_TERMS),
        "meta": find_terms(text, META_TERMS),
    }


def clean_cot(text: str) -> str:
    """Normalize a decoded rationale for training.

    - strip BOM / zero-width chars
    - drop a leading '# 思维链还原'-style title line (the decoder sometimes adds one)
    - normalize line endings, trailing spaces and runs of blank lines
    """
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    while lines and TITLE_LINE_RE.match(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_source_answer(text: str) -> str:
    """Light normalization for the raw teacher answer (used for ablation targets)."""
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in text.split("\n"))).strip()


def canonical_text(text: str) -> str:
    """Normalization used for dedupe keys (questions, near-identical CoTs)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def question_key(question: str) -> str:
    """Dedupe key: normalized question with most punctuation removed."""
    text = canonical_text(question)
    text = re.sub(r"[\s，,。.；;：:！!？?、\"'“”‘’（）()\[\]{}<>《》]", "", text)
    return text


def extract_answer_regex(text: str) -> dict[str, object]:
    """Best-effort final-answer extraction from a teacher answer or a rationale.

    Returns {"answer": str|None, "method": str, "confidence": float}.
    LLM-based extraction (data/extract_answers.py) is preferred for teacher
    answers; regex is the fallback and the primary path for eval generations.
    """
    if not text:
        return {"answer": None, "method": "empty", "confidence": 0.0}

    boxed = BOXED_RE.findall(text)
    if boxed:
        return {"answer": _strip_answer(boxed[-1]), "method": "boxed", "confidence": 0.95}

    answer_lines = ANSWER_LINE_RE.findall(text)
    if answer_lines:
        return {"answer": _strip_answer(answer_lines[-1]), "method": "answer_marker",
                "confidence": 0.9}

    conclusion = CONCLUSION_RE.findall(text)
    if conclusion:
        return {"answer": _strip_answer(conclusion[-1]), "method": "conclusion",
                "confidence": 0.4}

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return {"answer": _strip_answer(lines[-1]), "method": "last_line", "confidence": 0.2}
    return {"answer": None, "method": "none", "confidence": 0.0}


def _strip_answer(text: str) -> str:
    text = text.strip()
    # drop markdown emphasis / quotes / leftover math delimiters at the edges
    text = re.sub(r"^[\s$*_`\"“”'‘’]+|[\s$*_`\"“”'‘’]+$", "", text).strip()
    text = re.sub(r"^[：:，,。.;；\s]+", "", text)
    text = re.sub(r"[。.\s]+$", "", text)
    return text.strip()


def guess_answer_type(answer: str | None) -> str:
    """Coarse answer type: numeric / proof / text."""
    if not answer:
        return "unknown"
    answer = answer.strip()
    if NUMERIC_RE.match(answer):
        return "numeric"
    if re.search(r"完全平方|平方数|证明|成立|恒等|整除|整数", answer):
        return "proof"
    return "text"


if __name__ == "__main__":  # tiny smoke test
    assert clean_cot("# 思维链还原\n\n## 任务识别\n\n内容  \n\n\n\n更多") == "## 任务识别\n\n内容\n\n更多"
    assert clean_cot("## 任务识别\n内容") == "## 任务识别\n内容"
    parsed = extract_answer_regex("推导如下...\n\\boxed{42}")
    assert parsed["answer"] == "42" and parsed["method"] == "boxed"
    parsed = extract_answer_regex("所以最终答案：13")
    assert parsed["answer"] == "13"
    assert guess_answer_type("13") == "numeric"
    assert guess_answer_type("能够整除") == "proof"
    print("text_clean.py self-test OK")
