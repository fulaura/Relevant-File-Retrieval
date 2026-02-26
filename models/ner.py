from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langdetect import detect
import logging

try:
    from . import ner_med
except ImportError:  # pragma: no cover - fallback for direct execution
    try:
        import models.ner_med as ner_med  # type: ignore
    except ImportError:  # pragma: no cover
        ner_med = None

logging.basicConfig(level=logging.INFO)

# Load models once and reuse
models = {
    "kk": {
        "id": "yeshpanovrustem/xlm-roberta-large-ner-kazakh",
        "pipe": None
    },
    "ru": {
        "id": "Davlan/xlm-roberta-base-ner-hrl",
        "pipe": None
    },
    "en": {
        "id": "dslim/bert-base-NER",
        "pipe": None
    }
}

def detect_language(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return "en"
        lang = detect(text)
        if lang == "uk":  # common mistake: Kazakh often misclassified as Ukrainian
            return "kk"
        if lang not in models:
            logging.warning(f"Language '{lang}' not supported, fallback to English.")
            return "en"
        return lang
    except Exception as e:
        logging.error(f"Language detection failed ({e}), using fallback.")
        return "en"

def load_model(lang):
    if models[lang]["pipe"] is None:
        logging.info(f"Loading NER model for language: {lang}")
        tokenizer = AutoTokenizer.from_pretrained(models[lang]["id"])
        model = AutoModelForTokenClassification.from_pretrained(models[lang]["id"])
        models[lang]["pipe"] = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return models[lang]["pipe"]

def _detect_collection_language(texts: list[dict]):
    for item in texts:
        metadata = (item.get("metadata") or {})
        if metadata.get("language") in models:
            return metadata["language"]
    sample = []
    total_chars = 0
    for item in texts:
        content = item.get("content") or ""
        if not content:
            continue
        sample.append(content)
        total_chars += len(content)
        if total_chars >= 2000:
            break
    sample_text = " ".join(sample).strip()
    if not sample_text:
        return "en"
    return detect_language(sample_text)

def detect_and_ner(text: str):
    lang = detect_language(text)
    ner_pipeline = load_model(lang)
    return ner_pipeline(text)

def ner_collection(texts: list[dict]):
    if not texts:
        return []
    lang = _detect_collection_language(texts)
    ner_pipeline = load_model(lang)
    medical_enabled = lang == "en" and ner_med is not None
    if lang == "en" and ner_med is None:
        logging.warning("Medical English NER module is unavailable; skipping medical entity extraction.")
    for item in texts:
        content = item.get("content") or ""
        metadata = item.setdefault("metadata", {})
        if not content.strip():
            metadata["entities"] = []
            if medical_enabled:
                metadata["medical_entities"] = []
                metadata["medical_problems"] = []
                metadata["medical_tests"] = []
                metadata["medical_treatments"] = []
            continue
        metadata["entities"] = ner_pipeline(content)
        if medical_enabled:
            med_payload = ner_med.extract_medical_entities(content)
            summary = med_payload.get("summary", {}) if isinstance(med_payload, dict) else {}
            metadata["medical_entities"] = med_payload.get("entities", []) if isinstance(med_payload, dict) else []
            metadata["medical_problems"] = summary.get("medical_problems", [])
            metadata["medical_tests"] = summary.get("medical_tests", [])
            metadata["medical_treatments"] = summary.get("medical_treatments", [])
    return texts

# === Example Usage ===
if __name__ == "__main__":
    samples = [
        "Президент Тоқаев Астанада қабылдау өткізді.",
        "Президент Путин встретился с делегацией.",
        "President Biden met with world leaders in Washington."
    ]

    for txt in samples:
        print(f"\nText: {txt}")
        print("Entities:", detect_and_ner(txt))
