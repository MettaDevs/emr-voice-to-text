"""
Evaluation & Benchmarking Module for Hospital Voice-to-EMR (MedVoice AI)
Menyediakan kalkulasi metrik standar rumah sakit:
- WER (Word Error Rate) & CER (Character Error Rate) untuk STT
- Precision, Recall, F1-Score untuk Ekstraksi Gejala Klinis
- Negation Accuracy untuk deteksi gejala yang disangkal
- Vitals Exact Match & Validation Rate
- Allergy Status Accuracy
- Field Completeness
"""

import re
from typing import List, Dict, Any, Tuple


def _levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Menghitung jarak Levenshtein antara dua urutan token."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                   dp[i][j - 1],      # insertion
                                   dp[i - 1][j - 1])  # substitution
    return dp[m][n]


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Menghitung Word Error Rate (WER).
    WER = (S + D + I) / N
    """
    ref_words = re.findall(r'\w+', reference.lower())
    hyp_words = re.findall(r'\w+', hypothesis.lower())

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    distance = _levenshtein_distance(ref_words, hyp_words)
    return round(distance / len(ref_words), 4)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Menghitung Character Error Rate (CER).
    """
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    distance = _levenshtein_distance(ref_chars, hyp_chars)
    return round(distance / len(ref_chars), 4)


def calculate_clinical_extraction_metrics(gold_symptoms: List[str],
                                          pred_symptoms: List[str]) -> Dict[str, float]:
    """
    Menghitung Precision, Recall, dan F1-Score untuk ekstraksi gejala/entitas klinis.
    """
    gold_set = set(s.lower().strip() for s in gold_symptoms if s.strip())
    pred_set = set(s.lower().strip() for s in pred_symptoms if s.strip())

    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    # Hitung true positives dengan toleransi substring / kemiripan (1-to-1 matching)
    matched_golds = set()
    tp = 0
    for p in pred_set:
        for g in gold_set:
            if g not in matched_golds and (p in g or g in p):
                matched_golds.add(g)
                tp += 1
                break

    precision = round(min(1.0, tp / len(pred_set)), 4) if pred_set else 0.0
    recall = round(min(1.0, tp / len(gold_set)), 4) if gold_set else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_negation_accuracy(gold_negations: List[Tuple[str, bool]],
                                extracted_symptoms: List[Dict[str, Any]]) -> float:
    """
    Menghitung akurasi negasi gejala klinis.
    gold_negations: list of (symptom_name, is_negated)
    extracted_symptoms: list of {"name": str, "status": "present"|"absent"}
    """
    if not gold_negations:
        return 1.0

    correct = 0
    for term, should_be_negated in gold_negations:
        term_lower = term.lower()
        # Cari di extracted_symptoms
        found_status = None
        for sym in extracted_symptoms:
            if term_lower in sym.get("name", "").lower():
                found_status = sym.get("status")
                break

        if should_be_negated:
            # Sukses jika status == "absent" atau tidak dimasukkan sebagai keluhan aktif
            if found_status == "absent" or found_status is None:
                correct += 1
        else:
            # Sukses jika status == "present"
            if found_status == "present":
                correct += 1

    return round(correct / len(gold_negations), 4)


def calculate_vitals_accuracy(gold_vitals: Dict[str, Any],
                              pred_vitals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Menghitung exact match & extraction rate tanda vital.
    """
    total_fields = len(gold_vitals)
    if total_fields == 0:
        return {"accuracy": 1.0, "matched": 0, "total": 0}

    matched = 0
    details = {}
    for key, expected_val in gold_vitals.items():
        extracted_entry = pred_vitals.get(key)
        actual_val = extracted_entry.get("value") if isinstance(extracted_entry, dict) else extracted_entry

        # Normalisasi string perbandingan
        match = str(actual_val).strip().lower() == str(expected_val).strip().lower()
        if match:
            matched += 1
        details[key] = {"expected": expected_val, "actual": actual_val, "match": match}

    return {
        "accuracy": round(matched / total_fields, 4),
        "matched": matched,
        "total": total_fields,
        "details": details
    }


def calculate_allergy_accuracy(gold_status: str,
                               pred_status: str,
                               gold_items: List[str],
                               pred_items: List[str]) -> Dict[str, Any]:
    """
    Menghitung akurasi status alergi dan ekstraksi item alergen.
    """
    status_match = (gold_status.lower() == pred_status.lower())
    gold_set = set(i.lower().strip() for i in gold_items)
    pred_set = set(i.lower().strip() for i in pred_items)

    item_overlap = len(gold_set.intersection(pred_set))
    item_f1 = 1.0 if not gold_set and not pred_set else (
        round(2 * item_overlap / (len(gold_set) + len(pred_set)), 4) if (len(gold_set) + len(pred_set)) > 0 else 0.0
    )

    return {
        "status_match": status_match,
        "item_f1": item_f1,
        "gold_status": gold_status,
        "pred_status": pred_status
    }


def calculate_field_completeness(emr_result: Dict[str, Any]) -> float:
    """
    Menghitung rasio kelengkapan field terstruktur utama.
    """
    core_fields = [
        bool(emr_result.get("chief_complaint", {}).get("value")),
        bool(emr_result.get("duration")),
        bool(emr_result.get("symptoms")),
        any(v is not None for v in (emr_result.get("vitals") or {}).values() if isinstance(v, dict) and v.get("value")),
        emr_result.get("allergies", {}).get("status") not in [None, "not_asked", "uncertain"]
    ]
    return round(sum(1 for f in core_fields if f) / len(core_fields), 4)


def run_benchmark_suite(extractor) -> Dict[str, Any]:
    """
    Menjalankan pengujian evaluasi benchmark sintetis pada sistem Voice-to-EMR.
    """
    # 5 Kasus Uji Sintetis Rumah Sakit
    benchmark_cases = [
        {
            "id": "BENCH-01",
            "name": "Demam + Pilek + Negasi Batuk & Sesak",
            "transcript": "Selamat pagi dok. Badan saya panas dari kemarin malam dok terus pilek. Tidak ada batuk maupun sesak napas. Tensinya 120/80 mmHg, suhunya 38.5 derajat celcius. Tidak ada alergi dok.",
            "gold_chief": "Demam / Panas",
            "gold_symptoms": ["panas", "pilek"],
            "gold_negations": [("batuk", True), ("sesak napas", True), ("pilek", False)],
            "gold_vitals": {"blood_pressure": "120/80", "temperature": 38.5},
            "gold_allergy_status": "denied",
            "gold_allergy_items": []
        },
        {
            "id": "BENCH-02",
            "name": "Nyeri Dada + Mual + Tensi Tinggi",
            "transcript": "Nyeri dada sejak tadi pagi. Ada mual tapi tidak muntah dok. Tensinya 140/90 mmHg.",
            "gold_chief": "Nyeri dada",
            "gold_symptoms": ["nyeri dada", "mual"],
            "gold_negations": [("muntah", True), ("mual", False)],
            "gold_vitals": {"blood_pressure": "140/90"},
            "gold_allergy_status": "not_asked",
            "gold_allergy_items": []
        },
        {
            "id": "BENCH-03",
            "name": "Alergi Obat Spesifik Dilaporkan",
            "transcript": "Dok saya pusing dan mual sudah tiga hari. Saya ada riwayat alergi obat amoxicillin dok.",
            "gold_chief": "Sakit kepala / Pusing",
            "gold_symptoms": ["pusing", "mual"],
            "gold_negations": [("pusing", False)],
            "gold_vitals": {},
            "gold_allergy_status": "reported",
            "gold_allergy_items": ["amoxicillin"]
        }
    ]

    results = []
    total_f1 = []
    total_neg_acc = []
    total_vitals_acc = []
    total_allergy_status_match = []

    for case in benchmark_cases:
        res = extractor.extract(case["transcript"])

        # Ekstraksi gejala
        extracted_sym_names = [s.get("name", "") for s in res.get("symptoms", []) if s.get("status") == "present"]
        if res.get("chief_complaint", {}).get("value"):
            extracted_sym_names.append(res["chief_complaint"]["value"])

        ext_metrics = calculate_clinical_extraction_metrics(case["gold_symptoms"], extracted_sym_names)
        total_f1.append(ext_metrics["f1"])

        # Negasi
        neg_acc = calculate_negation_accuracy(case["gold_negations"], res.get("symptoms", []))
        total_neg_acc.append(neg_acc)

        # Vitals
        vit_acc = calculate_vitals_accuracy(case["gold_vitals"], res.get("vitals", {}))
        total_vitals_acc.append(vit_acc["accuracy"])

        # Alergi
        allergy_res = calculate_allergy_accuracy(
            case["gold_allergy_status"],
            res.get("allergies", {}).get("status", "not_asked"),
            case["gold_allergy_items"],
            res.get("allergies", {}).get("items", [])
        )
        total_allergy_status_match.append(1.0 if allergy_res["status_match"] else 0.0)

        results.append({
            "case_id": case["id"],
            "name": case["name"],
            "extraction_metrics": ext_metrics,
            "negation_accuracy": neg_acc,
            "vitals_accuracy": vit_acc["accuracy"],
            "allergy_accuracy": allergy_res,
            "completeness": calculate_field_completeness(res)
        })

    summary = {
        "cases_evaluated": len(benchmark_cases),
        "mean_symptom_f1": round(sum(total_f1) / len(total_f1), 4),
        "mean_negation_accuracy": round(sum(total_neg_acc) / len(total_neg_acc), 4),
        "mean_vitals_accuracy": round(sum(total_vitals_acc) / len(total_vitals_acc), 4),
        "allergy_status_accuracy": round(sum(total_allergy_status_match) / len(total_allergy_status_match), 4),
        "case_details": results
    }

    return summary


if __name__ == "__main__":
    from medical_extractor import MedicalComplaintExtractor
    ext = MedicalComplaintExtractor()
    print("=" * 65)
    print("MENJALANKAN BENCHMARK EVALUASI KUALITAS SISTEM (HOSPITAL POC)")
    print("=" * 65)
    report = run_benchmark_suite(ext)
    print(f"Total Kasus Diuji       : {report['cases_evaluated']}")
    print(f"Mean Symptom F1-Score   : {report['mean_symptom_f1'] * 100:.1f}%")
    print(f"Mean Negation Accuracy  : {report['mean_negation_accuracy'] * 100:.1f}%")
    print(f"Mean Vitals Accuracy    : {report['mean_vitals_accuracy'] * 100:.1f}%")
    print(f"Allergy Status Accuracy : {report['allergy_status_accuracy'] * 100:.1f}%")
    print("=" * 65)
