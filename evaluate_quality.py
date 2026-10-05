"""
evaluate_quality.py — MedVoice AI Quality Metrics Evaluation Script
===================================================================
Menghitung metrik performa objektif pada synthetic_dataset.py:
  - Clinical Extraction: Precision, Recall, F1-Score
  - Negation Accuracy
  - Vital Signs Extraction Accuracy
  - Allergy Status & Entity Accuracy
  - Duration Self-Correction Accuracy
  - Field Completeness
  - EMR Integration: Transmission Success Rate & Idempotent Duplicate Rate
"""

import json
from synthetic_dataset import SYNTHETIC_DATASET
from medical_extractor import MedicalComplaintExtractor, _extract_standardized_allergies, _extract_structured_duration
from emr_integration import MockHospitalEMRAdapter


def evaluate_all():
    extractor = MedicalComplaintExtractor()
    adapter = MockHospitalEMRAdapter(simulate_network_delay=0.0)

    # 1. Evaluasi Ekstraksi Klinis (Keluhan Utama & Gejala)
    tp_symptoms = 0
    fp_symptoms = 0
    fn_symptoms = 0

    # 2. Negasi
    negation_total = 0
    negation_correct = 0

    # 3. Tanda Vital
    vital_total = 0
    vital_correct = 0

    # 4. Alergi
    allergy_total = 0
    allergy_correct = 0

    # 5. Durasi & Koreksi Ucapan
    duration_total = 0
    duration_correct = 0

    # 6. Kelengkapan Field
    completeness_slots = 0
    filled_slots = 0

    for item in SYNTHETIC_DATASET:
        dialogue = item["dialogue"]
        exp = item["expected"]
        res = extractor.extract(dialogue)

        # ── Evaluasi Keluhan Utama & Gejala
        extracted_symptoms = [s["name"].lower() for s in res.get("symptoms", []) if s.get("status") == "present"]
        chief_val = (res["chief_complaint"]["value"] or "").lower()
        if chief_val and chief_val not in extracted_symptoms:
            extracted_symptoms.append(chief_val)

        def _match_symptom(a: str, b: str) -> bool:
            a_low, b_low = a.lower().strip(), b.lower().strip()
            if a_low in b_low or b_low in a_low:
                return True
            a_tokens = [t.strip() for t in a_low.replace('/', ' ').split() if len(t.strip()) > 2]
            b_tokens = [t.strip() for t in b_low.replace('/', ' ').split() if len(t.strip()) > 2]
            return any(t in b_tokens for t in a_tokens)

        exp_present = [s.lower() for s in exp.get("present_symptoms", [])]
        for sec in exp.get("secondary", []):
            if sec.lower() not in exp_present:
                exp_present.append(sec.lower())
        if "chief_complaint" in exp and exp["chief_complaint"].lower() not in exp_present:
            exp_present.append(exp["chief_complaint"].lower())

        for ep in exp_present:
            if any(_match_symptom(ep, es) for es in extracted_symptoms):
                tp_symptoms += 1
            else:
                fn_symptoms += 1

        for es in extracted_symptoms:
            if not any(_match_symptom(es, ep) for ep in exp_present):
                fp_symptoms += 1

        # ── Evaluasi Negasi
        if "absent_symptoms" in exp:
            for abs_sym in exp["absent_symptoms"]:
                negation_total += 1
                neg_match = next((s for s in res.get("symptoms", []) if abs_sym.lower() in s["name"].lower() and s["status"] == "absent"), None)
                if neg_match is not None:
                    negation_correct += 1

        # ── Evaluasi Tanda Vital
        if item["category"] == "E_vitals":
            vital_total += 1
            vit_res = res.get("vitals", {})
            v_ok = True
            if "blood_pressure" in exp and (not vit_res.get("blood_pressure") or vit_res["blood_pressure"]["value"] != exp["blood_pressure"]):
                v_ok = False
            if "temperature" in exp and (not vit_res.get("temperature") or vit_res["temperature"]["value"] != exp["temperature"]):
                v_ok = False
            if "pulse" in exp and (not vit_res.get("pulse") or vit_res["pulse"]["value"] != exp["pulse"]):
                v_ok = False
            if v_ok:
                vital_correct += 1

        # ── Evaluasi Alergi
        if item["category"] == "D_allergy":
            allergy_total += 1
            alg_res = res.get("allergies", {})
            a_ok = (alg_res.get("status") == exp.get("allergy_status"))
            if "allergen" in exp:
                if not any(exp["allergen"].lower() in it.lower() for it in alg_res.get("items", [])):
                    a_ok = False
            if a_ok:
                allergy_correct += 1

        # ── Evaluasi Durasi & Koreksi Ucapan
        if "duration" in exp:
            duration_total += 1
            exp_dur = exp["duration"]
            act_dur = res["lama_sakit"]
            if exp_dur is None:
                if act_dur == "":
                    duration_correct += 1
            else:
                if exp_dur.lower() in act_dur.lower():
                    duration_correct += 1

        # ── Evaluasi Kelengkapan Field
        completeness_slots += 3
        if res.get("keluhan_utama"): filled_slots += 1
        if res.get("keluhan_tambahan"): filled_slots += 1
        if res.get("lama_sakit") or exp.get("duration") is None: filled_slots += 1

    # Perhitungan Metrik Ekstraksi Klinis
    prec = tp_symptoms / (tp_symptoms + fp_symptoms) if (tp_symptoms + fp_symptoms) > 0 else 0.0
    rec = tp_symptoms / (tp_symptoms + fn_symptoms) if (tp_symptoms + fn_symptoms) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    neg_acc = (negation_correct / negation_total) if negation_total > 0 else 1.0
    vit_acc = (vital_correct / vital_total) if vital_total > 0 else 1.0
    alg_acc = (allergy_correct / allergy_total) if allergy_total > 0 else 1.0
    dur_acc = (duration_correct / duration_total) if duration_total > 0 else 1.0
    comp_rate = (filled_slots / completeness_slots) if completeness_slots > 0 else 1.0

    # Evaluasi Integrasi EMR
    test_payload = {
        "session_metadata": {"session_id": "METRIC-TEST-SESSION-001"},
        "chief_complaint": {"value": "Demam", "source": "patient"},
        "review_status": "approved",
        "clinical_validation": {"valid": True, "issues": []}
    }
    tx1 = adapter.send_emr(test_payload, actor="dr_eval")
    tx2 = adapter.send_emr(test_payload, actor="dr_eval")
    tx_success_rate = 1.0 if tx1["status"] == "SUCCESS" else 0.0
    idempotent_duplicate_rate = 1.0 if tx2["is_duplicate"] else 0.0

    metrics = {
        "dataset_size": len(SYNTHETIC_DATASET),
        "clinical_extraction": {
            "true_positives": tp_symptoms,
            "false_positives": fp_symptoms,
            "false_negatives": fn_symptoms,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        },
        "negation_accuracy": {
            "correct": negation_correct,
            "total": negation_total,
            "rate": round(neg_acc, 4)
        },
        "vitals_extraction_accuracy": {
            "correct": vital_correct,
            "total": vital_total,
            "rate": round(vit_acc, 4)
        },
        "allergy_accuracy": {
            "correct": allergy_correct,
            "total": allergy_total,
            "rate": round(alg_acc, 4)
        },
        "duration_accuracy": {
            "correct": duration_correct,
            "total": duration_total,
            "rate": round(dur_acc, 4)
        },
        "field_completeness_rate": round(comp_rate, 4),
        "emr_integration": {
            "transmission_success_rate": tx_success_rate,
            "idempotent_duplicate_rate": idempotent_duplicate_rate
        }
    }

    print("\n" + "="*80)
    print("HASIL EVALUASI KUALITAS KLINIS (SECTION 10)")
    print("="*80)
    print(f"Dataset Digunakan           : {metrics['dataset_size']} skenario sintetis (A–H)")
    print(f"Ekstraksi Klinis (Precision): {metrics['clinical_extraction']['precision'] * 100:.2f}%")
    print(f"Ekstraksi Klinis (Recall)   : {metrics['clinical_extraction']['recall'] * 100:.2f}%")
    print(f"Ekstraksi Klinis (F1-Score) : {metrics['clinical_extraction']['f1_score'] * 100:.2f}%")
    print(f"Negation Accuracy           : {metrics['negation_accuracy']['rate'] * 100:.2f}% ({negation_correct}/{negation_total})")
    print(f"Vital Signs Accuracy        : {metrics['vitals_extraction_accuracy']['rate'] * 100:.2f}% ({vital_correct}/{vital_total})")
    print(f"Allergy Accuracy            : {metrics['allergy_accuracy']['rate'] * 100:.2f}% ({allergy_correct}/{allergy_total})")
    print(f"Duration & Self-Correction  : {metrics['duration_accuracy']['rate'] * 100:.2f}% ({duration_correct}/{duration_total})")
    print(f"Field Completeness Rate     : {metrics['field_completeness_rate'] * 100:.2f}%")
    print(f"EMR Transmit Success Rate   : {metrics['emr_integration']['transmission_success_rate'] * 100:.2f}%")
    print(f"Idempotent Duplicate Hit    : {metrics['emr_integration']['idempotent_duplicate_rate'] * 100:.2f}%")
    print("="*80)

    with open("quality_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_all()
