from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable, List

from database import database_setup
from models import encoder, ner
from parsers.dispatcher import parse_document
from parsers.directory_analyzer import DirectoryAnalyzer


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse documents, chunk, embed, annotate, and load into the database."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["test"],
        help="Directories to scan for documents."
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdf"],
        help="File extensions to include (e.g., .pdf .docx)."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on number of documents to process."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging."
    )
    return parser.parse_args()


def discover_files(paths: Iterable[str], extensions: List[str]) -> List[Path]:
    analyzer = DirectoryAnalyzer(extensions=extensions)
    for directory in paths:
        analyzer.add_directory(directory)
    found = analyzer.analyze()
    files: List[Path] = []
    for directory in paths:
        files.extend(Path(path).resolve() for path in found.get(directory, []))
    return files


def prepare_document_metadata(document: dict) -> dict:
    metadata = document.get("metadata") or {}
    document["metadata"] = metadata
    return document


def process_document(file_path: Path, conn, cur) -> None:
    resolved_path = file_path.resolve()
    str_path = str(resolved_path)
    if database_setup.document_exists(cur, str_path):
        logging.info("Skipping %s (already ingested)", file_path)
        return

    logging.info("Parsing %s", file_path)
    start_total = perf_counter()
    document = parse_document(str_path)
    prepare_document_metadata(document)

    start_chunk = perf_counter()
    chunks = encoder.prepare_chunks_for_db(document, encoder.encode_text_transformer)
    logging.info("Prepared %d chunks for %s (%.2fs)", len(chunks), file_path.name, perf_counter() - start_chunk)

    start_ner = perf_counter()
    chunks = ner.ner_collection(chunks)
    logging.debug("NER completed for %s (%.2fs)", file_path.name, perf_counter() - start_ner)

    database_setup.db_load(conn, cur, chunks)
    logging.info(
        "Loaded %s into database (%d chunks, %.2fs)",
        file_path.name,
        len(chunks),
        perf_counter() - start_total
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    files = discover_files(args.paths, args.extensions)
    if not files:
        logging.warning("No files found for paths=%s extensions=%s", args.paths, args.extensions)
        return

    if args.max_docs is not None:
        files = files[: args.max_docs]

    logging.info("Discovered %d files", len(files))
    for f in files:
        logging.debug("File queued: %s", f)

    conn, cur = database_setup.db_connect()
    if not (conn and cur):
        logging.error("Failed to connect to database")
        return

    database_setup.db_init(conn, cur)

    succeeded = 0
    for file_path in files:
        try:
            process_document(file_path, conn, cur)
            succeeded += 1
        except Exception:  # noqa: BLE001
            logging.exception("Failed to ingest %s", file_path)

    logging.info("Ingestion complete. %d/%d documents succeeded.", succeeded, len(files))


if __name__ == "__main__":
    main()

    