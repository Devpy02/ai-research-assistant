"""CLI para indexar documentos de texto y verificar la persistencia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import load_index, persist_documents, verify_sources


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Núcleo documental del AI Research Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Fragmenta documentos TXT/MD y los persiste")
    ingest.add_argument("--source", type=Path, required=True, help="Archivo TXT/MD o directorio de documentos")
    ingest.add_argument("--output", type=Path, required=True, help="Directorio de salida del índice")
    ingest.add_argument("--chunk-size", type=int, default=400, help="Palabras aproximadas por fragmento (predeterminado: 400)")
    ingest.add_argument("--overlap", type=int, default=60, help="Solapamiento en palabras aproximadas (predeterminado: 60)")

    show = subparsers.add_parser("show", help="Carga y muestra los fragmentos persistidos")
    show.add_argument("--index", type=Path, required=True, help="Directorio del índice")
    show.add_argument("--limit", type=int, default=10, help="Cantidad máxima de fragmentos que se muestran")

    verify = subparsers.add_parser("verify", help="Valida el índice y compara con los documentos de origen")
    verify.add_argument("--source", type=Path, required=True, help="Archivo o directorio de origen")
    verify.add_argument("--index", type=Path, required=True, help="Directorio del índice")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.command == "ingest":
            manifest = persist_documents(
                args.source,
                args.output,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            print(json.dumps({
                "status": "ok",
                "document_count": manifest["document_count"],
                "chunk_count": manifest["chunk_count"],
                "output": str(args.output.resolve()),
                "manifest": str((args.output.resolve() / "manifest.json")),
                "chunks": str((args.output.resolve() / "chunks.jsonl")),
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "show":
            manifest, chunks = load_index(args.index)
            print(f"Documentos: {manifest['document_count']} | Fragmentos: {len(chunks)}")
            for row in chunks[: max(args.limit, 0)]:
                print("\n" + "=" * 72)
                print(f"chunk_id: {row['chunk_id']}")
                print(f"source: {row['source_path']} | chunk_index: {row['chunk_index']}")
                print(f"char offsets: [{row['char_start']}, {row['char_end']})")
                print(row["text"])
            return 0

        if args.command == "verify":
            manifest, chunks = load_index(args.index)
            errors = verify_sources(args.index, args.source)
            print(f"Índice íntegro: sí | documentos: {manifest['document_count']} | fragmentos: {len(chunks)}")
            if errors:
                print("Verificación de origen: FALLÓ")
                for error in errors:
                    print(f"- {error}")
                return 1
            print("Verificación de origen: OK; hashes de contenido coinciden.")
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
