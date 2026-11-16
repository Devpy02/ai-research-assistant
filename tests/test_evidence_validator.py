
import json

import pytest

from ai_research_assistant.evidence_validator import (
    EvidenceValidationError,
    _validate_input,
    _validate_model_response,
    validate_evidence,
)


CHUNKS = [
    {
        "chunk_id": "chunk_001",
        "text": "El servidor utiliza el endpoint /api/generate.",
    },
    {
        "chunk_id": "chunk_002",
        "text": "El endpoint /api/embed permite generar embeddings.",
    },
]


def test_validate_input_accepts_valid_chunks():
    result = _validate_input("¿Qué endpoint genera texto?", CHUNKS)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "chunk_001"


def test_validate_input_rejects_empty_query():
    with pytest.raises(EvidenceValidationError):
        _validate_input(" ", CHUNKS)


def test_validate_input_rejects_non_list_chunks():
    with pytest.raises(EvidenceValidationError):
        _validate_input("Pregunta", None)


def test_validate_input_rejects_duplicate_ids():
    duplicated = [CHUNKS[0], CHUNKS[0]]

    with pytest.raises(EvidenceValidationError):
        _validate_input("Pregunta", duplicated)


def test_validate_input_rejects_empty_chunk_text():
    invalid = [{"chunk_id": "chunk_001", "text": " "}]

    with pytest.raises(EvidenceValidationError):
        _validate_input("Pregunta", invalid)


def test_sufficient_response_with_valid_evidence():
    response = {
        "status": "sufficient",
        "reason": "El fragmento responde explícitamente.",
        "evidence": [
            {
                "chunk_id": "chunk_001",
                "quote": "El servidor utiliza el endpoint /api/generate.",
                "supports": "Identifica el endpoint de generación.",
            }
        ],
    }

    result = _validate_model_response(
        json.dumps(response),
        {
            "chunk_001": "El servidor utiliza el endpoint /api/generate.",
            "chunk_002": "El endpoint /api/embed permite generar embeddings.",
        },
    )

    assert result["status"] == "sufficient"
    assert len(result["evidence"]) == 1


def test_sufficient_without_evidence_abstains():
    response = {
        "status": "sufficient",
        "reason": "La evidencia es suficiente.",
        "evidence": [],
    }

    result = _validate_model_response(
        json.dumps(response),
        {"chunk_001": "Texto original."},
    )

    assert result["status"] == "insufficient"

def test_unknown_chunk_id_abstains():
    response = {
        "status": "sufficient",
        "reason": "Respuesta respaldada.",
        "evidence": [
            {
                "chunk_id": "chunk_inventado",
                "quote": "Texto inventado.",
                "supports": "Información inventada.",
            }
        ],
    }

    result = _validate_model_response(
        json.dumps(response),
        {"chunk_001": "Texto original."},
    )

    assert result["status"] == "insufficient"


def test_invalid_json_abstains():
    result = _validate_model_response(
        "esto no es JSON",
        {"chunk_001"},
    )

    assert result["status"] == "insufficient"
    assert result["evidence"] == []


def test_unknown_status_abstains():
    response = {
        "status": "maybe",
        "reason": "No está claro.",
        "evidence": [],
    }

    result = _validate_model_response(
        json.dumps(response),
        {"chunk_001": "Texto original."},
    )

    assert result["status"] == "insufficient"


def test_non_sufficient_status_is_preserved():
    response = {
        "status": "ambiguous",
        "reason": "La pregunta admite varias interpretaciones.",
        "evidence": [],
    }

    result = _validate_model_response(
        json.dumps(response),
        {"chunk_001": "Texto original."},
    )

    assert result["status"] == "ambiguous"


def test_empty_retrieval_abstains_without_model_call():
    result = validate_evidence(
        "¿Qué endpoint genera texto?",
        [],
    )

    assert result["status"] == "insufficient"
    assert result["evidence"] == []


def test_fabricated_quote_abstains():
    response = {
        "status": "sufficient",
        "reason": "Respuesta respaldada.",
        "evidence": [
            {
                "chunk_id": "chunk_001",
                "quote": "El servidor permite 100 solicitudes por minuto.",
                "supports": "Indica el límite de solicitudes.",
            }
        ],
    }

    result = _validate_model_response(
        json.dumps(response),
        {"chunk_001": "El servidor utiliza el endpoint /api/generate."},
    )

    assert result["status"] == "insufficient"


def test_empty_quote_abstains():
    response = {
        "status": "sufficient",
        "reason": "Respuesta respaldada.",
        "evidence": [
            {
                "chunk_id": "chunk_001",
                "quote": " ",
                "supports": "Identifica el endpoint.",
            }
        ],
    }

    result = _validate_model_response(
        json.dumps(response),
        {"chunk_001": "El servidor utiliza el endpoint /api/generate."},
    )

    assert result["status"] == "insufficient"



def test_irrelevant_but_authentic_quote_must_not_authorize(monkeypatch):
    from ai_research_assistant import evidence_validator

    original_text = (
        "El endpoint /api/generate recibe solicitudes "
        "de generación de texto. El endpoint /api/embed "
        "permite calcular representaciones vectoriales de textos."
    )

    initial_response = {
        "status": "sufficient",
        "reason": "El fragmento describe el funcionamiento del servidor.",
        "evidence": [
            {
                "chunk_id": "chunk_001",
                "quote": (
                    "El endpoint /api/generate recibe solicitudes "
                    "de generación de texto."
                ),
                "supports": (
                    "Indica la cantidad máxima de solicitudes por minuto."
                ),
            }
        ],
    }

    verification_response = {
        "valid": False,
        "reason": (
            "La cita describe la generación de texto, "
            "pero no contiene información sobre límites de solicitudes."
        ),
    }

    responses = iter([
        json.dumps(initial_response),
        json.dumps(verification_response),
    ])

    monkeypatch.setattr(
        evidence_validator,
        "_call_ollama",
        lambda **kwargs: next(responses),
    )

    result = validate_evidence(
        "¿Qué cantidad máxima de solicitudes por minuto permite el servidor?",
        [
            {
                "chunk_id": "chunk_001",
                "text": original_text,
            }
        ],
    )

    assert result["status"] == "insufficient"
    assert result["evidence"] == []


def test_relevant_evidence_passes_both_validations(monkeypatch):
    from ai_research_assistant import evidence_validator

    initial_response = {
        "status": "sufficient",
        "reason": "El fragmento identifica explícitamente el endpoint.",
        "evidence": [
            {
                "chunk_id": "chunk_001",
                "quote": "El servidor utiliza el endpoint /api/generate.",
                "supports": "Identifica el endpoint utilizado por el servidor.",
            }
        ],
    }

    verification_response = {
        "valid": True,
        "reason": "La cita responde directamente a la pregunta.",
    }

    responses = iter([
        json.dumps(initial_response),
        json.dumps(verification_response),
    ])

    monkeypatch.setattr(
        evidence_validator,
        "_call_ollama",
        lambda **kwargs: next(responses),
    )

    result = validate_evidence(
        "¿Qué endpoint utiliza el servidor?",
        CHUNKS,
    )

    assert result["status"] == "sufficient"
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["chunk_id"] == "chunk_001"
