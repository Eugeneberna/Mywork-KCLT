"""Shared constants and helpers for the MediMultiBERT-KCLT pipeline."""
import os, random, json, hashlib
import numpy as np

LANGS = ["en", "es", "fr", "hi", "ta"]
LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi", "ta": "Tamil"}

# The 24 Symptom2Disease labels, mapped to the concept name used to find the disease node in the KG.
DISEASE_CONCEPTS = {
    "Psoriasis": "Psoriasis",
    "Varicose Veins": "Varicose veins",
    "peptic ulcer disease": "Peptic ulcer",
    "drug reaction": "Adverse reaction to drug",
    "gastroesophageal reflux disease": "Gastroesophageal reflux disease",
    "allergy": "Hypersensitivity",
    "urinary tract infection": "Urinary tract infection",
    "Malaria": "Malaria",
    "Jaundice": "Jaundice",
    "Cervical spondylosis": "Cervical spondylosis",
    "Migraine": "Migraine",
    "Hypertension": "Hypertension",
    "Bronchial Asthma": "Asthma",
    "Acne": "Acne",
    "Arthritis": "Arthritis",
    "Dimorphic Hemorrhoids": "Hemorrhoids",
    "Pneumonia": "Pneumonia",
    "Common Cold": "Common cold",
    "Fungal infection": "Mycosis",
    "Dengue": "Dengue fever",
    "Impetigo": "Impetigo",
    "Chicken pox": "Chickenpox",
    "Typhoid": "Typhoid fever",
    "diabetes": "Diabetes mellitus",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def text_group_id(text: str) -> str:
    """Group key for a source narrative: identical (normalised) English texts share one group,
    so exact duplicates in Symptom2Disease can never be split across train/val/test."""
    norm = " ".join(str(text).lower().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# Alternative names used to find each disease in non-UMLS sources (e.g. the Disease Ontology)
DISEASE_ALIASES = {
    "drug reaction": ["drug allergy", "drug hypersensitivity"],
    "allergy": ["allergic disease"],
    "Fungal infection": ["fungal infectious disease"],
    "Dimorphic Hemorrhoids": ["hemorrhoid"],
    "Dengue": ["dengue disease"],
    "peptic ulcer disease": ["peptic ulcer disease"],
    "Jaundice": ["jaundice", "hyperbilirubinemia"],
    "Cervical spondylosis": ["spondylosis", "cervical spondylosis"],
    "Bronchial Asthma": ["asthma"],
    "Typhoid": ["typhoid fever"],
    "diabetes": ["diabetes mellitus"],
    "Chicken pox": ["chickenpox"],
}
