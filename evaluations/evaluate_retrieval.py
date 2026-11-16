
import json
import sys
from pathlib import Path

# Permite ejecutar el script directamente desde evaluations/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_research_assistant.retrieval import search


DATASET_PATH = Path(__file__).parent / "retrieval_dataset.json"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_index"

TOP_K_VALUES = (1, 3, 4)


def load_dataset():
    """Carga y valida el conjunto de consultas."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset.get("queries"), list):
        raise ValueError("El dataset debe contener una lista de consultas.")

    return dataset["queries"]


def calculate_recall_at_k(results, relevant_chunks, k):
    """Calcula Recall@k para una consulta."""
    if not relevant_chunks:
        return None

    retrieved = {
        result["chunk_id"]
        for result in results[:k]
    }

    relevant_retrieved = retrieved.intersection(relevant_chunks)

    return len(relevant_retrieved) / len(relevant_chunks)


def calculate_reciprocal_rank(results, relevant_chunks):
    """Calcula el Reciprocal Rank de una consulta."""
    if not relevant_chunks:
        return None

    for position, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant_chunks:
            return 1 / position

    return 0.0


def evaluate_query(query_data):
    """Ejecuta y evalúa una consulta individual."""
    results = search(
        query_data["question"],
        index_dir=INDEX_DIR,
        vector_dir=VECTOR_DIR,
        top_k=4,
    )

    relevant_chunks = set(query_data["relevant_chunks"])

    evaluation = {
        "id": query_data["id"],
        "question": query_data["question"],
        "type": query_data["type"],
        "results": results,
        "recall": {},
        "reciprocal_rank": None,
    }

    if query_data["type"] == "positive":
        for k in TOP_K_VALUES:
            evaluation["recall"][k] = calculate_recall_at_k(
                results, relevant_chunks, k
            )

        evaluation["reciprocal_rank"] = calculate_reciprocal_rank(
            results, relevant_chunks
        )

    return evaluation


def calculate_metrics(evaluations):
    """Calcula las métricas agregadas de consultas positivas."""
    positive = [
        item for item in evaluations
        if item["type"] == "positive"
    ]

    metrics = {}

    for k in TOP_K_VALUES:
        values = [
            item["recall"][k]
            for item in positive
        ]

        metrics[f"recall@{k}"] = (
            sum(values) / len(values) if values else 0.0
        )

    reciprocal_ranks = [
        item["reciprocal_rank"]
        for item in positive
    ]

    metrics["mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

    return metrics


def print_report(evaluations, metrics):
    """Muestra los resultados de la evaluación."""
    print("\n" + "=" * 65)
    print("EVALUACIÓN DE RECUPERACIÓN SEMÁNTICA")
    print("=" * 65)

    for item in evaluations:
        print(f"\n[{item['id']}] {item['type'].upper()}")
        print(f"Pregunta: {item['question']}")

        if item["type"] == "positive":
            for k, value in item["recall"].items():
                print(f"Recall@{k}: {value:.4f}")

            print(
                f"Reciprocal Rank: "
                f"{item['reciprocal_rank']:.4f}"
            )
        else:
            print("Fragmentos recuperados sin evidencia esperada:")

        for position, result in enumerate(item["results"], start=1):
            print(
                f"  {position}. {result['chunk_id']} "
                f"| Score: {result['score']:.4f}"
            )

    print("\n" + "-" * 65)
    print("MÉTRICAS AGREGADAS (CONSULTAS POSITIVAS)")

    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    print("-" * 65)


def main():
    queries = load_dataset()
    evaluations = []

    for query_data in queries:
        print(f"Evaluando {query_data['id']}...")
        evaluations.append(evaluate_query(query_data))

    metrics = calculate_metrics(evaluations)
    print_report(evaluations, metrics)


if __name__ == "__main__":
    main()
