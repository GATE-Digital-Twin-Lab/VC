"""Storage layout: a shared generation cache, and one directory per run.

The two halves are deliberately asymmetric.

``data/raw/`` is SHARED and keyed by ``run.name``. Generation is the expensive
step and the append-only cache is the whole reason downstream phases are pure
functions of it (protocol section 1). Timestamping it would mean every run
re-samples from scratch, which defeats the design; re-running generation against
an existing cache is instead a no-op on rows already present.

``results/runs/<timestamp>__<name>/`` is per-run and holds everything DERIVED.
Those are cheap to recompute and change whenever a knob changes, so each run
gets its own directory and nothing is silently overwritten. Each run directory
carries a ``manifest.json`` and a snapshot of the exact resolved config, because
a results table without the knobs that produced it is not reproducible.

**One directory per phase**, because a single flat ``tables/`` reached 28 files
and stopped being readable:

    manifest.json  config.snapshot.yaml
    processed/              answers.parquet, questions.parquet
    run/                    splits, pitfalls, notes -- not owned by one phase
    phase0_gate/            gate.json, tau_sweep.csv, label_*.csv
    phase1_generation/      parse audits
    phase2_descriptive/     reliability, AUROCs, histograms
    phase3_survival/        KM, hazard, budget, U detection, product rule
    phase4_clm/             LTT headline, discount stability
    phase5_invariance/      temperature and paraphrase sweeps
    phase6_transfer/        isotonic transfer

A phase's FIGURES sit beside its tables rather than in a separate ``figures/``
tree: ``km.csv`` and ``km.png`` are one result in two renderings, and splitting
them meant every question about a figure started by guessing which of two
directories it was in. Directories are numbered by PROTOCOL section, not by
execution order -- the pipeline runs generation before the gate (see
``pipeline``), and a second numbering would be one too many.

``processed/`` stays at the run root and is deliberately NOT under a phase: it
is the handoff between phases (Phase 0 writes correctness onto it, Phase 3 adds
clusters, Phase 4 reads it), so filing it under whichever phase happened to
touch it last would misdescribe what it is.

Phase directories are created on first write, so a run that stopped after the
gate does not grow seven empty folders implying work that never happened.

Phase-by-phase CLI use still works: ``vc_uq generate`` creates a run directory
and later commands attach to the most recent one for that run name unless told
otherwise.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .config import Config
from .schemas import (ANSWER_KEY, ANSWERS_SCHEMA, PER_POSITION_SCHEMA,
                      QUESTIONS_SCHEMA, VC_PRE_KEY, VC_PRE_REPEATS_SCHEMA,
                      check_unique_key, conform, empty_frame)

MANIFEST_NAME = "manifest.json"
CONFIG_SNAPSHOT_NAME = "config.snapshot.yaml"
LATEST_POINTER = "LATEST"

# Keyed by NAME, not number. Phase numbers and pipeline step numbers already
# disagree in this codebase (protocol Phase 0 is step 2); a third numbering
# addressed by callers would be worse than a word.
PHASE_DIRS = {
    "run": "run",
    "gate": "phase0_gate",
    "generation": "phase1_generation",
    "descriptive": "phase2_descriptive",
    "survival": "phase3_survival",
    "clm": "phase4_clm",
    "invariance": "phase5_invariance",
    "transfer": "phase6_transfer",
}


def timestamp(cfg: Config) -> str:
    return datetime.now().strftime(cfg.get("run.timestamp_format"))


def runs_root(cfg: Config) -> Path:
    return cfg.results_root / cfg.get("run.runs_subdir")


def list_runs(cfg: Config, *, name: str | None = None) -> list[Path]:
    """Run directories, oldest first.

    The timestamp format sorts lexicographically, so directory order is
    chronological order and no index file can drift out of sync.
    """
    root = runs_root(cfg)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        if name is not None and d.name.split("__", 1)[1] != name:
            continue
        out.append(d)
    return out


def latest_run(cfg: Config, *, name: str | None = None) -> Path | None:
    runs = list_runs(cfg, name=name)
    return runs[-1] if runs else None


def _git_rev() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parents[2])
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class Store:
    """Paths and IO for one run."""

    def __init__(self, cfg: Config, *, run_id: str | None = None,
                 attach: bool = False, create: bool = True):
        self.cfg = cfg
        self.run = cfg.get("run.name")

        if run_id is not None:
            self.run_id = run_id
            self.run_dir = runs_root(cfg) / f"{run_id}__{self.run}"
        elif attach:
            found = latest_run(cfg, name=self.run)
            if found is None:
                raise FileNotFoundError(
                    f"no existing run directory for run.name={self.run!r} under "
                    f"{runs_root(cfg)}. Run a phase that creates one (e.g. "
                    "`vc_uq generate`) first, or pass --new-run.")
            self.run_dir = found
            self.run_id = found.name.split("__", 1)[0]
        else:
            self.run_id = timestamp(cfg)
            self.run_dir = runs_root(cfg) / f"{self.run_id}__{self.run}"

        self.raw_dir = cfg.raw_dir
        self.processed_dir = self.run_dir / "processed"

        if create:
            self._create()

    # -- lifecycle ---------------------------------------------------------
    def _create(self) -> None:
        # Phase directories are NOT created here. An empty phase4_clm/ in a run
        # that stopped after the gate reads as "Phase 4 produced nothing", which
        # is a different claim from "Phase 4 never ran".
        for d in (self.raw_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not (self.run_dir / CONFIG_SNAPSHOT_NAME).exists():
            self.snapshot_config()
        if not (self.run_dir / MANIFEST_NAME).exists():
            self.write_manifest(status="started")
        self._update_latest_pointer()

    def snapshot_config(self) -> Path:
        """The exact resolved config, overrides included.

        Results without the knobs that produced them are not reproducible, so
        this is written before any phase runs rather than at the end.
        """
        path = self.run_dir / CONFIG_SNAPSHOT_NAME
        path.write_text(yaml.safe_dump(self.cfg.data, sort_keys=False),
                        encoding="utf-8")
        return path

    def config_digest(self) -> str:
        """Short hash of the resolved config, for drift detection."""
        blob = yaml.safe_dump(self.cfg.data, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:10]

    def write_manifest(self, *, status: str, command: str | None = None,
                       extra: dict | None = None) -> Path:
        """Update the manifest.

        Config-derived fields are fixed at CREATION and never rewritten. A later
        command in the same run may carry different ``-o`` overrides, and
        letting it restamp ``n_max`` or the model name would silently
        misdescribe the data already on disk. Instead each command appends to
        ``commands`` with its own digest, and a digest that differs from the
        run's is flagged there -- which is how you find out that `survival` ran
        under different knobs than `generate` did.
        """
        path = self.run_dir / MANIFEST_NAME
        existing = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        from . import __version__

        digest = self.config_digest()
        fixed = {
            "run_id": self.run_id,
            "run_name": self.run,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_digest": digest,
            "seed": self.cfg.get("run.seed"),
            "model": {"backend": self.cfg.get("model.backend"),
                      "name": self.cfg.get("model.name")},
            "embedding_backend": self.cfg.get("embedding.backend"),
            "nli_backend": self.cfg.get("nli.backend"),
            "datasets": self.cfg.get("dataset.datasets"),
            "n_max": self.cfg.get("generation.n_max"),
            "raw_cache": str(self.raw_dir),
            "vc_uq_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_rev": _git_rev(),
        }
        payload = {**existing}
        for k, v in fixed.items():
            payload.setdefault(k, v)

        commands = list(payload.get("commands", []))
        if command is not None:
            commands.append({
                "command": command,
                "at": datetime.now().isoformat(timespec="seconds"),
                "config_digest": digest,
                "matches_run_config": digest == payload.get("config_digest"),
            })
        payload["commands"] = commands
        payload["status"] = status
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload["config_drift"] = any(
            c.get("matches_run_config") is False for c in commands)

        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2, default=_jsonable),
                        encoding="utf-8")
        return path

    def finalize(self, *, status: str = "complete", command: str | None = None,
                 extra: dict | None = None) -> Path:
        return self.write_manifest(status=status, command=command, extra=extra)

    def _update_latest_pointer(self) -> None:
        """Plain text pointer rather than a symlink: symlinks need elevation on
        Windows, and this run targets a Windows workstation."""
        try:
            (runs_root(self.cfg) / LATEST_POINTER).write_text(
                self.run_dir.name, encoding="utf-8")
        except OSError:
            pass

    # -- paths -------------------------------------------------------------
    def raw_path(self, table: str) -> Path:
        # Raw stays name-prefixed: the directory is shared across runs.
        return self.raw_dir / f"{self.run}__{table}.parquet"

    def processed_path(self, table: str) -> Path:
        return self.processed_dir / f"{table}.parquet"

    def phase_dir(self, phase: str) -> Path:
        """The directory for one phase, created on demand."""
        try:
            leaf = PHASE_DIRS[phase]
        except KeyError:
            raise KeyError(
                f"unknown phase {phase!r}; expected one of {sorted(PHASE_DIRS)}"
            ) from None
        d = self.run_dir / leaf
        d.mkdir(parents=True, exist_ok=True)
        return d

    def phase(self, phase: str) -> "PhaseStore":
        """A writer scoped to one phase.

        Each pipeline step opens one of these at the top and writes everything
        through it, so an artifact cannot be filed under the wrong phase by
        forgetting a prefix -- the scope is lexical rather than a name pattern
        matched at write time.
        """
        return PhaseStore(self, phase)

    def table_path(self, name: str, phase: str = "run") -> Path:
        return self.phase_dir(phase) / f"{name}.csv"

    def figure_path(self, name: str, phase: str = "run") -> Path:
        return self.phase_dir(phase) / f"{name}.png"

    def json_path(self, name: str, phase: str = "run") -> Path:
        return self.phase_dir(phase) / f"{name}.json"

    # -- raw (append-only) -------------------------------------------------
    def append_raw(self, table: str, df: pd.DataFrame, schema, key) -> pd.DataFrame:
        """Merge rows into the shared cache. Safe against concurrent writers.

        Generation is sharded across GPUs (``generate --shard i/n``), so two
        processes append to this one file. The merge is read-modify-write, so
        without the lock the second writer would read a snapshot taken before
        the first one's rows landed and then overwrite them -- silently losing
        exactly the draws the checkpointing exists to protect. The lock is held
        across the read AND the write, not just the write.
        """
        with self._raw_lock(table):
            return self._append_raw_locked(table, df, schema, key)

    def _raw_lock(self, table: str):
        from contextlib import contextmanager

        @contextmanager
        def _lock():
            path = self.raw_path(table).with_name(self.raw_path(table).name + ".lock")
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(path, "w")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()
        return _lock()

    def _append_raw_locked(self, table: str, df: pd.DataFrame, schema,
                           key) -> pd.DataFrame:
        df = conform(df, schema, name=table)
        path = self.raw_path(table)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            # Re-running generation must be a no-op on rows already cached.
            combined = combined.drop_duplicates(subset=list(key), keep="first")
        else:
            combined = df
        check_unique_key(combined, key, name=table)
        combined = conform(combined, schema, name=table)
        # Written to a sibling and renamed. Generation checkpoints into this
        # file every few hundred draws, so a process killed mid-write is a
        # realistic event rather than a theoretical one -- and a half-written
        # parquet is not a partial cache, it is an unreadable one. os.replace is
        # atomic within a filesystem, so the old file stands until the new one
        # is complete.
        # Process-unique: the lock already serialises writers, but a shared
        # ".tmp" name means any future unlocked path has two workers writing one
        # file and renaming each other's half-written bytes into place.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        combined.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return combined

    def append_answers(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.append_raw("answers", df, ANSWERS_SCHEMA, ANSWER_KEY)

    def append_vc_pre_repeats(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.append_raw("vc_pre_repeats", df, VC_PRE_REPEATS_SCHEMA, VC_PRE_KEY)

    def append_per_position(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.append_raw("per_position", df, PER_POSITION_SCHEMA, ANSWER_KEY)

    # -- resume: what is already in the cache ------------------------------
    # Generation is the only expensive step, so a run that dies at hour nine of
    # twelve must not start over. Every cache key is derivable WITHOUT calling
    # the model -- `seed` comes from a hash of (run_seed, q_id, draw_idx, T,
    # variant) -- so the set of missing rows is computable up front.

    @staticmethod
    def _align_keys(wanted: pd.DataFrame, schema, key) -> pd.DataFrame:
        """Cast the key columns to the declared dtypes before joining on them.

        The key is a mixed tuple: a float (``temperature``), a nullable Int64
        (``seed``), strings. A float64 column does not join cleanly against a
        Float64 one, and a silent non-match here is indistinguishable from a
        cold cache -- it would quietly re-generate everything, which is the one
        failure this path exists to prevent. So both sides are cast, always.
        """
        out = wanted.copy()
        for col in key:
            out[col] = out[col].astype(schema[col][0])
        return out

    def missing_keys(self, table: str, wanted: pd.DataFrame, schema,
                     key: tuple[str, ...]) -> pd.DataFrame:
        """Rows of ``wanted`` (a key-only frame) that are not yet cached."""
        wanted = self._align_keys(wanted, schema, key)
        path = self.raw_path(table)
        if not path.exists():
            return wanted.reset_index(drop=True)
        have = self._align_keys(pd.read_parquet(path, columns=list(key)),
                                schema, key)
        merged = wanted.merge(have.drop_duplicates(), on=list(key), how="left",
                              indicator=True)
        return (merged.loc[merged["_merge"] == "left_only", list(wanted.columns)]
                .reset_index(drop=True))

    def cached_rows(self, table: str, wanted: pd.DataFrame, schema,
                    key: tuple[str, ...]) -> pd.DataFrame:
        """The cached rows matching ``wanted``, conformed to ``schema``."""
        path = self.raw_path(table)
        if not path.exists():
            return empty_frame(schema)
        have = conform(pd.read_parquet(path), schema, name=table)
        w = self._align_keys(wanted[list(key)].drop_duplicates(), schema, key)
        return have.merge(w, on=list(key), how="inner").reset_index(drop=True)

    def missing_answer_keys(self, wanted: pd.DataFrame) -> pd.DataFrame:
        return self.missing_keys("answers", wanted, ANSWERS_SCHEMA, ANSWER_KEY)

    def missing_vc_pre_keys(self, wanted: pd.DataFrame) -> pd.DataFrame:
        return self.missing_keys("vc_pre_repeats", wanted, VC_PRE_REPEATS_SCHEMA,
                                 VC_PRE_KEY)

    def cached_answers(self, wanted: pd.DataFrame) -> pd.DataFrame:
        return self.cached_rows("answers", wanted, ANSWERS_SCHEMA, ANSWER_KEY)

    def cached_vc_pre_repeats(self, wanted: pd.DataFrame) -> pd.DataFrame:
        return self.cached_rows("vc_pre_repeats", wanted, VC_PRE_REPEATS_SCHEMA,
                                VC_PRE_KEY)

    # -- reads -------------------------------------------------------------
    def read_answers(self, *, splits: list[str] | None = None,
                     datasets: list[str] | None = None) -> pd.DataFrame:
        df = self._read(self.processed_path("answers"), self.raw_path("answers"),
                        ANSWERS_SCHEMA, "answers")
        if splits is not None:
            df = df[df["split"].isin(splits)]
        if datasets is not None:
            df = df[df["dataset"].isin(datasets)]
        return df.reset_index(drop=True)

    def read_questions(self, *, splits: list[str] | None = None,
                       datasets: list[str] | None = None) -> pd.DataFrame:
        df = self._read(self.processed_path("questions"), self.raw_path("questions"),
                        QUESTIONS_SCHEMA, "questions")
        if splits is not None:
            df = df[df["split"].isin(splits)]
        if datasets is not None:
            df = df[df["dataset"].isin(datasets)]
        return df.reset_index(drop=True)

    def read_vc_pre_repeats(self) -> pd.DataFrame:
        return self._read(None, self.raw_path("vc_pre_repeats"),
                          VC_PRE_REPEATS_SCHEMA, "vc_pre_repeats")

    def _read(self, preferred: Path | None, fallback: Path, schema, name) -> pd.DataFrame:
        """This run's processed table wins; the shared raw cache is beneath it."""
        for path in (preferred, fallback):
            if path is not None and path.exists():
                return conform(pd.read_parquet(path), schema, name=name)
        raise FileNotFoundError(
            f"no {name} table found at {preferred or fallback}; run the generation phase first"
        )

    # -- writes ------------------------------------------------------------
    def write_processed(self, table: str, df: pd.DataFrame, schema=None) -> Path:
        if schema is not None:
            df = conform(df, schema, name=table)
        path = self.processed_path(table)
        df.to_parquet(path, index=False)
        return path

    def write_questions(self, df: pd.DataFrame, *, raw: bool = False) -> Path:
        df = conform(df, QUESTIONS_SCHEMA, name="questions")
        path = self.raw_path("questions") if raw else self.processed_path("questions")
        df.to_parquet(path, index=False)
        return path

    def write_table(self, name: str, df: pd.DataFrame, phase: str = "run") -> Path:
        path = self.table_path(name, phase)
        df.to_csv(path, index=False)
        return path

    def write_json(self, name: str, payload: dict, phase: str = "run") -> Path:
        path = self.json_path(name, phase)
        path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
        return path

    def read_json(self, name: str, phase: str = "run") -> dict:
        path = self.json_path(name, phase)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the phase that produces it")
        return json.loads(path.read_text(encoding="utf-8"))

    def find(self, name: str) -> Path | None:
        """Locate an artifact by stem across every phase directory.

        For interactive use and for reading a run written before the layout was
        split by phase. Returns None rather than raising, so a caller can say
        which phase it expected.
        """
        for leaf in list(PHASE_DIRS.values()) + ["tables", "figures"]:
            d = self.run_dir / leaf
            if not d.is_dir():
                continue
            for ext in (".csv", ".json", ".png"):
                p = d / f"{name}{ext}"
                if p.exists():
                    return p
        return None


class PhaseStore:
    """A :class:`Store` scoped to one phase's output directory.

    Holds no state of its own beyond the phase name -- every write goes to the
    underlying store, so a phase view and the run it belongs to can never
    disagree about where the run directory is.
    """

    __slots__ = ("store", "phase")

    def __init__(self, store: "Store", phase: str):
        if phase not in PHASE_DIRS:
            raise KeyError(
                f"unknown phase {phase!r}; expected one of {sorted(PHASE_DIRS)}")
        self.store = store
        self.phase = phase

    @property
    def dir(self) -> Path:
        return self.store.phase_dir(self.phase)

    def table_path(self, name: str) -> Path:
        return self.store.table_path(name, self.phase)

    def figure_path(self, name: str) -> Path:
        return self.store.figure_path(name, self.phase)

    def json_path(self, name: str) -> Path:
        return self.store.json_path(name, self.phase)

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        return self.store.write_table(name, df, self.phase)

    def write_json(self, name: str, payload: dict) -> Path:
        return self.store.write_json(name, payload, self.phase)

    def read_json(self, name: str) -> dict:
        return self.store.read_json(name, self.phase)

    def __repr__(self) -> str:
        return f"PhaseStore({self.phase!r}, {self.store.run_dir.name!r})"


def migrate_layout(run_dir: Path, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Move a flat ``tables/``+``figures/`` run into per-phase directories.

    Idempotent: a run already in the new layout yields no moves. Routing is by
    the ``phaseN_`` prefix the old names carried, plus an explicit table for the
    artifacts that never had one. The prefix is stripped on the way, since the
    directory now says it.

    Anything unrecognised goes to ``run/`` rather than being left behind, so the
    old directories can be removed once empty and no file is ever dropped.
    """
    by_prefix = {
        "phase0_": "gate", "phase2_": "descriptive", "phase3_": "survival",
        "phase4_": "clm", "phase5_": "invariance", "phase6_": "transfer",
    }
    by_name = {
        "dataset_splits": "run", "dataset_split_deviation": "run",
        "pitfalls": "run", "pitfalls_summary": "run", "run_notes": "run",
        "nli_audit": "survival",
        # Figures were never phase-prefixed.
        "km": "survival", "hazard": "survival", "product_rule": "survival",
        "budget_collapse": "survival", "reliability": "descriptive",
        "vc_hist_post": "descriptive", "temperature_sweep": "invariance",
    }
    moves: list[tuple[Path, Path]] = []
    for leaf in ("tables", "figures"):
        src_dir = run_dir / leaf
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.iterdir()):
            if not src.is_file():
                continue
            stem = src.stem
            phase = by_name.get(stem)
            if phase is None and stem.startswith("parse_audit__"):
                phase = "generation"
            if phase is None:
                for pre, ph in by_prefix.items():
                    if stem.startswith(pre):
                        phase, stem = ph, stem[len(pre):]
                        break
            if phase is None:
                phase = "run"
            dest_dir = run_dir / PHASE_DIRS[phase]
            dest = dest_dir / f"{stem}{src.suffix}"
            moves.append((src, dest))
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                src.replace(dest)
        if not dry_run and src_dir.is_dir() and not any(src_dir.iterdir()):
            src_dir.rmdir()
    return moves


def run_summaries(cfg: Config, *, name: str | None = None) -> pd.DataFrame:
    """Every run's manifest as one table, for `vc_uq runs`."""
    rows = []
    for d in list_runs(cfg, name=name):
        path = d / MANIFEST_NAME
        if not path.exists():
            rows.append({"run_id": d.name.split("__", 1)[0], "run_name":
                         d.name.split("__", 1)[1], "status": "no manifest",
                         "path": str(d)})
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run_id": m.get("run_id"), "run_name": m.get("run_name"),
            "status": m.get("status"), "created_at": m.get("created_at"),
            "model": (m.get("model") or {}).get("name"),
            "backend": (m.get("model") or {}).get("backend"),
            "n_max": m.get("n_max"), "seed": m.get("seed"),
            "commands": len(m.get("commands", [])),
            "config_drift": m.get("config_drift"),
            "git_rev": m.get("git_rev"), "path": str(d),
        })
    return pd.DataFrame(rows)


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj)} is not JSON serialisable")
