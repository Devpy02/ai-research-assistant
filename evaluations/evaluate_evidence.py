"""Evaluación del validador de suficiencia de evidencia."""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_research_assistant.evidence_validator import validate_evidence


DATASET_PATH = Path(__file__).parent / "evidence_dataset.json"
RESULTS_PATH = Path(__file__).parent / "evidence_evaluation_results.json"

EXPECTED_STATUSES = {
    "sufficient",
    "insufficient",
    "ambiguous",
    "contradictory",
}


def load_dataset():
    """Carga y valida el conjunto de evaluación."""

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    cases = dataset.get("cases")

    if not isinstance(cases, list) or not cases:
        raise ValueError("El dataset debe contener una lista de casos.")

    for case in cases:
        required = {
            "id",
            "category",
            "query",
            "retrieved_chunks",
            "expected_status",
            "rationale",
        }

        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("Se encontró un caso con estructura inválida.")

        if case["expected_status"] not in EXPECTED_STATUSES:
            raise ValueError(
                f"Estado esperado inválido en {case['id']}."
            )

    return cases


def evaluate_case(case):
    """Ejecuta y evalúa un caso individual."""

    start = time.perf_counter()

    try:
        result = validate_evidence(
            query=case["query"],
            retrieved_chunks=case["retrieved_chunks"],
        )

        actual_status = result["status"]

        if actual_status not in EXPECTED_STATUSES:
            raise ValueError("El validador devolvió un estado desconocido.")

        error = None

    except Exception as exc:
        actual_status = "error"
        result = {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "evidence": [],
        }
        error = result["reason"]

    elapsed = time.perf_counter() - start

    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_status": case["expected_status"],
        "actual_status": actual_status,
        "correct": actual_status == case["expected_status"],
        "elapsed_seconds": round(elapsed, 3),
        "reason": result.get("reason", ""),
        "evidence": result.get("evidence", []),
        "error": error,
        "rationale": case["rationale"],
    }


def calculate_metrics(results):
    """Calcula métricas globales y desglosadas."""

    total = len(results)
    correct = sum(item["correct"] for item in results)

    sufficient_results = [
        item for item in results
        if item["actual_status"] == "sufficient"
    ]

    sufficient_correct = sum(
        item["expected_status"] == "sufficient"
        for item in sufficient_results
    )

    abstention_cases = [
        item for item in results
        if item["expected_status"] != "sufficient"
    ]

    correct_abstentions = sum(
        item["actual_status"] != "sufficient"
        and item["actual_status"] != "error"
        for item in abstention_cases
    )

    category_results = defaultdict(list)

    for item in results:
        category_results[item["category"]].append(item)

    accuracy_by_category = {}

    for category, items in category_results.items():
        accuracy_by_category[category] = {
            "total": len(items),
            "correct": sum(item["correct"] for item in items),
            "accuracy": round(
                sum(item["correct"] for item in items) / len(items),
                4,
            ),
        }

    times = [item["elapsed_seconds"] for item in results]

    return {
        "total_cases": total,
        "correct_cases": correct,
        "incorrect_cases": total - correct,
        "global_accuracy": round(correct / total, 4),
        "sufficient_precision": round(
            sufficient_correct / len(sufficient_results), 4
        ) if sufficient_results else 0.0,
        "correct_abstention_rate": round(
            correct_abstentions / len(abstention_cases), 4
        ) if abstention_cases else 0.0,
        "average_time_seconds": round(sum(times) / len(times), 3),
        "minimum_time_seconds": round(min(times), 3),
        "maximum_time_seconds": round(max(times), 3),
        "accuracy_by_category": accuracy_by_category,
    }


def print_report(results, metrics):
    """Presenta los resultados de la evaluación."""

    print("\n" + "=" * 72)
    print("EVALUACIÓN DEL VALIDADOR DE EVIDENCIA")
    print("=" * 72)

    for item in results:
        status = "OK" if item["correct"] else "ERROR"

        print(f"\n[{item['id']}] {status} - {item['category']}")
        print(f"Pregunta: {item['query']}")
        print(f"Esperado: {item['expected_status']}")
        print(f"Obtenido: {item['actual_status']}")
        print(f"Tiempo: {item['elapsed_seconds']:.3f} s")
        print(f"Razón: {item['reason']}")

        for evidence in item["evidence"]:
            print(
                f"  Evidencia: {evidence['chunk_id']} "
                f"| Cita: {evidence['quote']}"
            )

    print("\n" + "-" * 72)
    print("MÉTRICAS GLOBALES")

    for key in (
        "total_cases",
        "correct_cases",
        "incorrect_cases",
        "global_accuracy",
        "sufficient_precision",
        "correct_abstention_rate",
        "average_time_seconds",
        "minimum_time_seconds",
        "maximum_time_seconds",
    ):
        print(f"{key}: {metrics[key]}")

    print("\nEXACTITUD POR CATEGORÍA")

    for category, values in metrics["accuracy_by_category"].items():
        print(
            f"{category}: {values['correct']}/{values['total']} "
            f"({values['accuracy']:.2%})"
        )

    print("-" * 72)


def main():
    cases = load_dataset()
    results = []

    print(f"Casos a evaluar: {len(cases)}")

    for position, case in enumerate(cases, start=1):
        print(
            f"\nEvaluando {position}/{len(cases)}: {case['id']}",
            flush=True,
        )

        results.append(evaluate_case(case))

    metrics = calculate_metrics(results)

    report = {
        "evaluation": "Evidence Validator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_PATH.name,
        "model": "qwen2.5:1.5b",
        "metrics": metrics,
        "results": results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_report(results, metrics)

    print(f"\nResultados guardados en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
