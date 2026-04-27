import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.paths import ROOT_DIR, resolve_project_path


CONFIG_PATH = ROOT_DIR / "config" / "training_pipeline.yaml"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_pipeline_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["pipeline"]


def clean_name(raw_name):
    return raw_name.replace(".jsonl", "_clean.jsonl")


def dedup_name(raw_name):
    return raw_name.replace(".jsonl", "_dedup.jsonl")


def stage_path(step_id, leaf):
    return f"data/stages/{step_id}/{leaf}/"


def get_steps():
    config = load_pipeline_config()
    return sorted(config["steps"], key=lambda item: int(item["order"]))


def get_step(step_id):
    for step in get_steps():
        if step["id"] == step_id:
            return step
    available = ", ".join(step["id"] for step in get_steps())
    raise KeyError(f"Step pipeline non trovato: {step_id}. Disponibili: {available}")


def get_db_path(config=None):
    config = config or load_pipeline_config()
    return resolve_project_path(config["state_db"])


def connect_db():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db():
    config = load_pipeline_config()
    with connect_db() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_steps (
                id TEXT PRIMARY KEY,
                step_order INTEGER NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                objective TEXT NOT NULL,
                requires TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                checkpoint_path TEXT,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_step_datasets (
                step_id TEXT NOT NULL,
                raw_file TEXT NOT NULL,
                dataset_order INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (step_id, raw_file)
            )
            """
        )

        now = utc_now()
        for step in get_steps():
            db.execute(
                """
                INSERT INTO pipeline_steps (
                    id, step_order, kind, title, objective, requires, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    step_order=excluded.step_order,
                    kind=excluded.kind,
                    title=excluded.title,
                    objective=excluded.objective,
                    requires=excluded.requires,
                    updated_at=excluded.updated_at
                """,
                (
                    step["id"],
                    int(step["order"]),
                    step["kind"],
                    step["title"],
                    step["objective"],
                    step.get("requires"),
                    now,
                ),
            )
            db.execute(
                "DELETE FROM pipeline_step_datasets WHERE step_id = ?",
                (step["id"],),
            )
            for dataset_order, raw_file in enumerate(step.get("raw_files", [])):
                db.execute(
                    """
                    INSERT INTO pipeline_step_datasets (
                        step_id, raw_file, dataset_order, updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (step["id"], raw_file, dataset_order, now),
                )

    return {"db": str(get_db_path(config)), "steps": len(config["steps"])}


def mark_step(step_id, status, checkpoint_path=None, message=None):
    init_db()
    now = utc_now()
    started_at = now if status == "running" else None
    completed_at = now if status == "completed" else None

    with connect_db() as db:
        current = db.execute(
            "SELECT started_at FROM pipeline_steps WHERE id = ?",
            (step_id,),
        ).fetchone()
        if current is None:
            raise KeyError(f"Step pipeline non registrato: {step_id}")

        db.execute(
            """
            UPDATE pipeline_steps
            SET status = ?,
                checkpoint_path = COALESCE(?, checkpoint_path),
                started_at = COALESCE(started_at, ?),
                completed_at = COALESCE(?, completed_at),
                updated_at = ?
            WHERE id = ?
            """,
            (status, checkpoint_path, started_at, completed_at, now, step_id),
        )
        db.execute(
            """
            INSERT INTO pipeline_events (step_id, status, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (step_id, status, message, now),
        )

    return {"step_id": step_id, "status": status, "checkpoint_path": checkpoint_path}


def step_status(step_id):
    init_db()
    with connect_db() as db:
        row = db.execute(
            """
            SELECT id, step_order, kind, title, objective, requires, status,
                   checkpoint_path, started_at, completed_at, updated_at
            FROM pipeline_steps
            WHERE id = ?
            """,
            (step_id,),
        ).fetchone()

    if row is None:
        raise KeyError(f"Step pipeline non registrato: {step_id}")

    keys = [
        "id",
        "order",
        "kind",
        "title",
        "objective",
        "requires",
        "status",
        "checkpoint_path",
        "started_at",
        "completed_at",
        "updated_at",
    ]
    return dict(zip(keys, row))


def assert_ready(step_id):
    step = get_step(step_id)
    required = step.get("requires")
    if not required:
        return {"ready": True, "step_id": step_id}

    status = step_status(required)
    if status["status"] != "completed":
        raise RuntimeError(
            f"{step_id} richiede {required}, ma lo stato e {status['status']}. "
            "Riparti dall'ultimo checkpoint ufficiale stabile."
        )

    return {"ready": True, "step_id": step_id, "requires": required}


def env_payload(step_id):
    step = get_step(step_id)
    raw_files = step.get("raw_files", [])
    payload = {
        "id": step["id"],
        "order": int(step["order"]),
        "kind": step["kind"],
        "title": step["title"],
        "objective": step["objective"],
        "requires": step.get("requires"),
        "raw_files": raw_files,
        "raw_include": ";".join(raw_files),
        "clean_include": ";".join(clean_name(name) for name in raw_files),
        "dedup_include": ";".join(dedup_name(name) for name in raw_files),
        "allowed_languages": ",".join(step.get("allowed_languages", ["it"])),
        "data_profile": step.get("data_profile", "auto"),
        "allowed_formats": ",".join(step.get("allowed_formats", [])),
        "cleaned_data_dir": stage_path(step["id"], "cleaned"),
        "shards_dir": stage_path(step["id"], "shards"),
        "official_checkpoint": step.get("official_checkpoint", step["id"]),
    }
    return payload


def list_payload():
    config = load_pipeline_config()
    return {
        "name": config["name"],
        "state_db": str(get_db_path(config)),
        "strict_sequence": bool(config.get("strict_sequence", True)),
        "steps": [
            {
                "id": step["id"],
                "order": int(step["order"]),
                "kind": step["kind"],
                "title": step["title"],
                "objective": step["objective"],
                "requires": step.get("requires"),
                "data_profile": step.get("data_profile", "auto"),
                "allowed_formats": step.get("allowed_formats", []),
            }
            for step in get_steps()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Stato e configurazione pipeline training Jarvis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")
    subparsers.add_parser("init-db")

    env_parser = subparsers.add_parser("env")
    env_parser.add_argument("--step", required=True)

    ready_parser = subparsers.add_parser("assert-ready")
    ready_parser.add_argument("--step", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--step", required=True)

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("--step", required=True)
    mark_parser.add_argument("--status", required=True)
    mark_parser.add_argument("--checkpoint", default=None)
    mark_parser.add_argument("--message", default=None)

    args = parser.parse_args()

    try:
        if args.command == "list":
            print(json.dumps(list_payload(), ensure_ascii=False, indent=2))
        elif args.command == "init-db":
            print(json.dumps(init_db(), ensure_ascii=False, indent=2))
        elif args.command == "env":
            print(json.dumps(env_payload(args.step), ensure_ascii=False))
        elif args.command == "assert-ready":
            print(json.dumps(assert_ready(args.step), ensure_ascii=False))
        elif args.command == "status":
            print(json.dumps(step_status(args.step), ensure_ascii=False, indent=2))
        elif args.command == "mark":
            print(
                json.dumps(
                    mark_step(args.step, args.status, args.checkpoint, args.message),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
