from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

_MODEL_DIR = Path(__file__).resolve().parent / "biobert-humadex-ner-output"

_LABEL_MAP: Dict[int, str] = {
    0: "O",
    1: "B-PROBLEM",
    2: "I-PROBLEM",
    3: "E-PROBLEM",
    4: "S-PROBLEM",
    5: "B-TREATMENT",
    6: "I-TREATMENT",
    7: "E-TREATMENT",
    8: "S-TREATMENT",
    9: "B-TEST",
    10: "I-TEST",
    11: "E-TEST",
    12: "S-TEST",
}

_CATEGORY_KEYS = {
    "PROBLEM": "medical_problems",
    "TREATMENT": "medical_treatments",
    "TEST": "medical_tests",
}

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not _MODEL_DIR.exists():
            raise FileNotFoundError(f"Medical NER model directory not found: {_MODEL_DIR}")
        logging.info("Loading medical English NER model from %s", _MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(_MODEL_DIR)
        id2label = {idx: tag for idx, tag in _LABEL_MAP.items()}
        label2id = {tag: idx for idx, tag in id2label.items()}
        model.config.id2label = id2label
        model.config.label2id = label2id
        _PIPELINE = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
    return _PIPELINE


def _empty_summary() -> Dict[str, List[str]]:
    return {key: [] for key in _CATEGORY_KEYS.values()}


def _update_summary(summary: Dict[str, List[str]], entities: List[dict]) -> None:
    for entity in entities:
        group = (entity.get("entity_group") or "").upper()
        target_key = _CATEGORY_KEYS.get(group)
        if not target_key:
            continue
        value = (entity.get("word") or "").strip()
        if not value:
            continue
        if value not in summary[target_key]:
            summary[target_key].append(value)


def extract_medical_entities(text: str) -> Dict[str, object]:
    if not isinstance(text, str) or not text.strip():
        return {"entities": [], "summary": _empty_summary()}
    pipe = _load_pipeline()
    entities: List[dict] = pipe(text)
    summary = _empty_summary()
    _update_summary(summary, entities)
    return {"entities": entities, "summary": summary}


__all__ = ["extract_medical_entities"]
