"""Phase 1 -- generation.

Fixed T, ``N_MAX`` draws per question, **fresh context per draw**: no chat
history, otherwise the draws are genuinely dependent rather than merely modelled
as such, and every independence statement downstream becomes untestable.

Pre-hoc and post-hoc VC are elicited in separate calls. ``vc_pre`` is asked
``R_pre`` times per question because a prospective feeling-of-knowing with no
answer to anchor on has real sampling variance, and averaging over repeats keeps
``vc_pre`` from being one noisy draw. The spread is kept alongside the mean.

Token statistics are recorded twice: once under the VC-augmented prompt that
produced the answer, and once by teacher-forcing the same answer tokens under a
clean prompt. Reporting only the first would confound token entropy with the VC
instruction that was added to the context to elicit VC in the first place.

Generation runs in TWO passes (protocol 6.4). The ``classify`` pass draws
``N_MAX`` on every question and is what decides ``in_U``; the ``downstream``
pass draws a second, independent ``N_MAX`` on the calib/eval questions and is
what everything certified is calibrated and evaluated on. The pass name is
salted into the seed, so the two are genuinely different draws rather than the
same text under two labels.

Everything is cached by ``(model, dataset, q_id, draw_set, draw_idx, T,
prompt_variant, seed)``. Because ``seed`` is a hash of the other components
rather than a counter, the whole key set is computable before a single token is
generated -- which is what lets ``resume=True`` skip work that is already on
disk, instead of generating it and discarding it at write time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from . import prompts
from .backends import build_lm
from .backends.base import SamplingParams
from .config import Config
from .parsing import parse_answer_and_vc, parse_audit_from_status, parse_vc_only
from .schemas import (ANSWER_KEY, ANSWERS_SCHEMA, DRAW_SETS, VC_PRE_KEY,
                      VC_PRE_REPEATS_SCHEMA, empty_frame)
from .store import Store

_SEED_MODULUS = 2**31 - 1


def derive_seed(run_seed: int, *parts: object) -> int:
    """Deterministic per-call seed. Reproducible and independent across draws."""
    key = "::".join([str(run_seed), *[str(p) for p in parts]])
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % _SEED_MODULUS


def _mock_meta(messages: list[dict], meta: dict[str, object]) -> list[dict]:
    """Side channel for the simulated backend ONLY.

    Real backends never see this: adding it to a live prompt would leak the
    question id into the context and contaminate every elicitation.
    """
    tag = "".join(f"[[{k}:{v}]]" for k, v in meta.items() if v is not None)
    out = [dict(m) for m in messages]
    out[0]["content"] = out[0]["content"] + "\n" + tag
    return out


@dataclass
class Generator:
    cfg: Config
    store: Store
    last_resume: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Before build_lm: a mistyped variant name should not cost a GGUF load,
        # let alone surface hours later from inside Phase 5.
        prompts.check_config_variants(self.cfg)
        self.lm = build_lm(self.cfg)
        # h_tok is only comparable across the sampling and teacher-forced passes
        # if both used the same chat template, so record which one that was.
        source = getattr(self.lm, "prompt_template_source", None)
        if source is not None:
            self.store.write_manifest(status="running",
                                      extra={"prompt_template_source": source})
        self.is_mock = self.cfg.get("model.backend") == "mock"
        self.model_name = self.cfg.get("model.name")
        self.run_seed = int(self.cfg.get("run.seed"))
        self.sampling = SamplingParams.from_config(self.cfg)
        if not bool(self.cfg.get("generation.fresh_context_per_draw")):
            raise ValueError(
                "generation.fresh_context_per_draw=false is not supported. Carrying "
                "chat history across draws makes them genuinely dependent rather "
                "than merely modelled as such, and every independence statement "
                "downstream (5.7, 6.7) becomes untestable.")

    # -- prompt plumbing ---------------------------------------------------
    def _messages(self, spec: prompts.PromptSpec, row: pd.Series,
                  **extra: object) -> list[dict]:
        msgs = spec.build(question=row["question"])
        if self.is_mock:
            msgs = _mock_meta(msgs, {"qid": row["q_id"], "ds": row["dataset"],
                                     "variant": spec.variant, "kind": spec.kind,
                                     **extra})
        return msgs

    def _call(self, messages: list[dict], *, temperature: float, seed: int):
        # Truncation knobs travel with every call. p_q is definitionally a
        # function of the decoder, so the sweep in 8.1 only measures T if T is
        # the only thing that changes between calls.
        return self.lm.generate(messages, seed=seed,
                                params=self.sampling.replace(temperature=temperature))

    # -- cache keys, computable without generating -------------------------
    def answer_seed(self, q_id: object, draw_idx: int, temperature: float,
                    variant: str, draw_set: str) -> int:
        """The pass name is part of the seed, not just a label on the row.

        Without it the downstream pass would replay the classify pass token for
        token -- same prompt, same seed, same decoder -- and the "fresh draws"
        of 6.4 step 3 would be a copy under a second name.
        """
        return derive_seed(self.run_seed, draw_set, q_id, draw_idx, temperature,
                           variant)

    def answer_keys(self, questions: pd.DataFrame, *, n_max: int,
                    temperature: float, variant: str,
                    draw_set: str = "classify") -> pd.DataFrame:
        rows = [{
            "model": self.model_name,
            "dataset": q["dataset"],
            "q_id": q["q_id"],
            "draw_set": draw_set,
            "draw_idx": draw_idx,
            "temperature": temperature,
            "prompt_variant": variant,
            "seed": self.answer_seed(q["q_id"], draw_idx, temperature, variant,
                                     draw_set),
        } for _, q in questions.iterrows() for draw_idx in range(n_max)]
        return pd.DataFrame(rows, columns=list(ANSWER_KEY))

    def vc_pre_keys(self, questions: pd.DataFrame, *, r_pre: int,
                    variant: str) -> pd.DataFrame:
        rows = [{
            "model": self.model_name,
            "dataset": q["dataset"],
            "q_id": q["q_id"],
            "repeat_idx": repeat_idx,
            "prompt_variant": variant,
            "seed": derive_seed(self.run_seed, "pre", q["q_id"], repeat_idx,
                                variant),
        } for _, q in questions.iterrows() for repeat_idx in range(r_pre)]
        return pd.DataFrame(rows, columns=list(VC_PRE_KEY))

    def _keep_per_position(self, q_id: str, frac: float) -> bool:
        """Which questions keep full per-position entropy arrays.

        Hashed per question rather than drawn from a shared RNG stream: a
        stream-order draw would pick a different subsample as soon as
        resumption skips a question, so which questions carry per-position data
        would depend on how many times the job had crashed.
        """
        return (derive_seed(self.run_seed, "per_position", q_id) / _SEED_MODULUS) < frac

    @staticmethod
    def _refresh_split(cached: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
        """``split`` is a property of the question in THIS run, not of the draw.

        It is deliberately not part of the cache key -- re-splitting must not
        invalidate expensive generation -- so a cached row still carries
        whichever split it was assigned when it was produced. Re-stamping it
        from the current question table stops a resumed run from mixing two
        split assignments, which would quietly break the calib/eval
        disjointness that every guarantee downstream rests on.
        """
        mapping = questions.set_index("q_id")["split"]
        out = cached.copy()
        mapped = out["q_id"].map(mapping).astype("string")
        out["split"] = mapped.fillna(out["split"])
        return out

    # -- post-hoc draws ----------------------------------------------------
    def draw_answers(self, questions: pd.DataFrame, *, n_max: int | None = None,
                     temperature: float | None = None,
                     prompt_variant: str | None = None,
                     draw_set: str = "classify",
                     store_per_position: bool = True,
                     resume: bool = False) -> pd.DataFrame:
        """``N_MAX`` draws for every question, in one named pass.

        With ``resume=True`` the rows already in the shared cache are read back
        rather than re-generated, and the return value is their union with the
        newly generated ones -- so the contract is the same either way: the
        caller gets every draw it asked for. Only Phase 1 sets it, because only
        Phase 1's output is written to the cache; the Phase 5 sweeps generate
        into frames that are never persisted, so there is nothing to resume from.
        """
        if draw_set not in DRAW_SETS:
            raise ValueError(f"draw_set must be one of {DRAW_SETS}, got {draw_set!r}")
        n_max = n_max if n_max is not None else int(self.cfg.get("generation.n_max"))
        temperature = (temperature if temperature is not None
                       else float(self.cfg.get("generation.temperature")))
        variant = prompt_variant or self.cfg.get("generation.prompt_variant_post")
        spec = prompts.get(variant)
        clean_spec = prompts.get(self.cfg.get("generation.prompt_variant_clean"))
        do_clean = bool(self.cfg.get("generation.token_stats.teacher_force_clean_prompt"))
        keep_frac = float(self.cfg.get("generation.token_stats.store_per_position_fraction"))

        keys = self.answer_keys(questions, n_max=n_max, temperature=temperature,
                                variant=variant, draw_set=draw_set)
        cached = None
        if resume:
            todo = self.store.missing_answer_keys(keys)
            cached = self.store.cached_answers(keys)
        else:
            todo = keys
        pending = set(zip(todo["q_id"].astype(str), todo["draw_idx"].astype(int)))

        # Checkpointing, but ONLY on the persisted path. resume=False is the
        # Phase 5 sweep, which generates at other temperatures and prompt
        # variants into frames that are deliberately never cached -- flushing
        # those would put sweep draws into the table Phases 2-4 read.
        checkpoint_every = int(self.cfg.get("generation.checkpoint_every"))
        rows: list[dict] = []
        per_position: list[dict] = []
        n_flushed = 0

        def _flush() -> None:
            nonlocal rows, per_position, n_flushed
            if rows:
                self.store.append_answers(pd.DataFrame(rows))
                n_flushed += len(rows)
                rows = []
            if per_position:
                self.store.append_per_position(pd.DataFrame(per_position))
                per_position = []

        for _, q in questions.iterrows():
            keep_positions = store_per_position and self._keep_per_position(
                str(q["q_id"]), keep_frac)
            for draw_idx in range(n_max):
                if (str(q["q_id"]), draw_idx) not in pending:
                    continue
                seed = self.answer_seed(q["q_id"], draw_idx, temperature, variant,
                                        draw_set)
                msgs = self._messages(spec, q)
                gen = self._call(msgs, temperature=temperature, seed=seed)
                parsed = parse_answer_and_vc(gen.text)

                row: dict = {
                    "q_id": q["q_id"], "dataset": q["dataset"], "split": q["split"],
                    "draw_set": draw_set,
                    "draw_idx": draw_idx, "answer": parsed.answer,
                    "vc_post": parsed.vc, "vc_post_raw": parsed.raw,
                    "temperature": temperature, "prompt_variant": variant,
                    "seed": seed, "model": self.model_name,
                    "parse_status": parsed.status,
                }
                row.update(gen.stats.summary())
                if do_clean:
                    clean_msgs = self._messages(clean_spec, q)
                    clean = self.lm.teacher_force(clean_msgs, parsed.answer)
                    row["h_tok_mean_clean"] = clean.summary()["h_tok_mean"]
                if keep_positions:
                    per_position.append({
                        "q_id": q["q_id"], "dataset": q["dataset"],
                        "draw_set": draw_set,
                        "draw_idx": draw_idx, "temperature": temperature,
                        "prompt_variant": variant, "seed": seed,
                        "model": self.model_name,
                        "entropies": gen.stats.entropies.tolist(),
                        "chosen_probs": gen.stats.chosen_probs.tolist(),
                    })
                rows.append(row)
                if resume and checkpoint_every and len(rows) >= checkpoint_every:
                    # A 27B model at ~2.4 s/draw is hours of GPU time. Losing it
                    # to a kill signal or an OOM because nothing was written
                    # until the end is not an acceptable failure mode, and the
                    # cache is append-only and keyed, so a partial write is a
                    # valid smaller cache rather than a corrupt one.
                    _flush()

        if resume:
            _flush()
            # Re-read rather than concatenating in memory: the flushed rows are
            # already in the cache, and this way the return value is the cache's
            # own view of exactly the keys that were asked for -- including rows
            # a CONCURRENT shard wrote (see Store.append_raw).
            out = self._refresh_split(self.store.cached_answers(keys), questions)
            out = out.sort_values(["q_id", "draw_idx"]).reset_index(drop=True)
            n_new = n_flushed
        else:
            n_new = len(rows)
            new_df = pd.DataFrame(rows) if rows else empty_frame(ANSWERS_SCHEMA)
            if cached is not None and len(cached):
                out = pd.concat([self._refresh_split(cached, questions), new_df],
                                ignore_index=True)
                out = out.sort_values(["q_id", "draw_idx"]).reset_index(drop=True)
            else:
                out = new_df

        self.last_resume = {"requested": int(len(keys)), "generated": int(n_new),
                            "reused_from_cache": int(len(out)) - int(n_new)}
        audit = parse_audit_from_status(out["parse_status"])
        audit["prompt_variant"] = variant
        audit["temperature"] = temperature
        audit["draw_set"] = draw_set
        audit["resume"] = dict(self.last_resume)
        # The pass name is in the filename: without it the second pass would
        # overwrite the first pass's audit, and the parse-failure rate is
        # reported per pass.
        self.store.write_json(
            f"parse_audit__{variant}__T{temperature}__{draw_set}", audit)
        if per_position:
            # Append-only into the shared raw cache, not the run directory: these
            # arrays cost a decode to recreate, and a resumed run only produces
            # them for the draws it actually made. Appending lets the earlier
            # run's rows stand instead of vanishing. (On the resume path _flush
            # has already emptied this; only the unpersisted Phase 5 path can
            # still reach here with rows, and it has no store to write to.)
            self.store.append_per_position(pd.DataFrame(per_position))
        return out

    # -- pre-hoc elicitation ----------------------------------------------
    def draw_vc_pre(self, questions: pd.DataFrame, *, r_pre: int | None = None,
                    prompt_variant: str | None = None,
                    resume: bool = False) -> pd.DataFrame:
        r_pre = r_pre if r_pre is not None else int(self.cfg.get("generation.r_pre"))
        variant = prompt_variant or self.cfg.get("generation.prompt_variant_pre")
        spec = prompts.get(variant)
        if spec.kind != "pre":
            raise ValueError(f"{variant!r} is not a pre-hoc prompt; vc_pre must be "
                             "elicited with no answer in context")
        temperature = float(self.cfg.get("generation.temperature"))

        keys = self.vc_pre_keys(questions, r_pre=r_pre, variant=variant)
        cached = None
        if resume:
            todo = self.store.missing_vc_pre_keys(keys)
            cached = self.store.cached_vc_pre_repeats(keys)
        else:
            todo = keys
        pending = set(zip(todo["q_id"].astype(str), todo["repeat_idx"].astype(int)))

        rows: list[dict] = []
        for _, q in questions.iterrows():
            for repeat_idx in range(r_pre):
                if (str(q["q_id"]), repeat_idx) not in pending:
                    continue
                seed = derive_seed(self.run_seed, "pre", q["q_id"], repeat_idx, variant)
                msgs = self._messages(spec, q)
                gen = self._call(msgs, temperature=temperature, seed=seed)
                parsed = parse_vc_only(gen.text)
                rows.append({
                    "q_id": q["q_id"], "dataset": q["dataset"],
                    "repeat_idx": repeat_idx, "vc_pre": parsed.vc,
                    "vc_pre_raw": parsed.raw, "prompt_variant": variant,
                    "seed": seed, "model": self.model_name,
                    "parse_status": parsed.status,
                })

        new = pd.DataFrame(rows) if rows else empty_frame(VC_PRE_REPEATS_SCHEMA)
        if cached is not None and len(cached):
            out = pd.concat([cached, new], ignore_index=True)
            out = out.sort_values(["q_id", "repeat_idx"]).reset_index(drop=True)
        else:
            out = new

        self.last_resume = {"requested": int(len(keys)), "generated": int(len(new)),
                            "reused_from_cache": int(len(out)) - int(len(new))}
        audit = parse_audit_from_status(out["parse_status"])
        audit["prompt_variant"] = variant
        audit["resume"] = dict(self.last_resume)
        self.store.write_json(f"parse_audit__{variant}", audit)
        return out


def shard_questions(questions: pd.DataFrame, shard: tuple[int, int] | None
                    ) -> pd.DataFrame:
    """The ``i`` of ``n`` slice of the question table, for parallel generation.

    One 27B instance per GPU is roughly twice the throughput of one instance
    layer-split across both, since llama.cpp's default split runs the two halves
    in sequence with a transfer between them. Each worker pins itself to one
    device with ``CUDA_VISIBLE_DEVICES`` and takes one shard.

    Sliced by POSITION over a q_id-sorted table, so the shards are disjoint,
    exhaustive and identical whatever order the caller built the frame in. They
    write to the same cache, which is safe because Store.append_raw holds a lock
    across its read-modify-write.
    """
    if shard is None:
        return questions
    i, n = shard
    if not (0 <= i < n) or n < 1:
        raise ValueError(f"shard must be i/n with 0 <= i < n, got {i}/{n}")
    return questions.sort_values("q_id").iloc[i::n]


def run_phase1(cfg: Config, store: Store, questions: pd.DataFrame,
               *, splits: list[str] | None = None,
               shard: tuple[int, int] | None = None) -> dict:
    """Generate and cache everything Phase 1 owes downstream phases.

    Two passes of N_MAX (protocol 6.4):

    1. ``classify`` on EVERY question. These draws decide ``in_U``, ``p_hat``,
       ``K_q`` and the whole survival picture, and nothing downstream is
       calibrated or evaluated on them.
    2. ``downstream`` on the ``generation.downstream_splits`` questions only.
       Phase 2, Phase 4 and 6.7 run on these.

    Deciding U with the draws that are later certified is selection on the
    outcome being certified. It voids the LTT guarantee, and it makes the 6.7
    U-curve tautological: "none of the first k draws was correct" is true by
    construction for every question the subset was defined to contain, so the
    observed frequency reads 1.0 at every k no matter what VC claimed.

    Resumable: re-running against a populated cache generates nothing and costs
    a parquet read, so a job that dies at hour nine of twelve restarts where it
    stopped rather than from zero.

    ``splits`` restricts generation to those splits, for staging a long job.
    The Phase 0 gate needs only ``tau_select``, and a gate that fails ends the
    study -- so drawing the other 90% of the questions first is spending most of
    the compute before finding out whether any of it is interpretable. Because
    ``split`` is deliberately NOT part of the cache key and the per-draw seed is
    a hash of ``(draw_set, q_id, draw_idx, T, variant)``, a staged pass is a
    strict subset of the full one: the later unrestricted run reuses every draw
    made here rather than resampling it.

    The question table is still written whole. Only ``vc_pre`` is left null for
    questions this pass did not elicit -- writing the subset would delete the
    other questions from the shared table.
    """
    gen = Generator(cfg, store)

    target = questions
    if splits is not None:
        target = questions[questions["split"].isin([str(s) for s in splits])]
        if target.empty:
            raise ValueError(
                f"no questions in split(s) {sorted(splits)}; the table holds "
                f"{sorted(questions['split'].unique())}")
    target = shard_questions(target, shard)

    classify = gen.draw_answers(target, draw_set="classify", resume=True)
    resume_answers = {"classify": dict(gen.last_resume)}

    downstream_splits = [str(s) for s in cfg.get("generation.downstream_splits")]
    downstream_q = target[target["split"].isin(downstream_splits)]
    if len(downstream_q):
        downstream = gen.draw_answers(downstream_q, draw_set="downstream",
                                      resume=True)
        resume_answers["downstream"] = dict(gen.last_resume)
    else:
        downstream = empty_frame(ANSWERS_SCHEMA)
        resume_answers["downstream"] = {"requested": 0, "generated": 0,
                                        "reused_from_cache": 0}

    answers = store.append_answers(
        pd.concat([classify, downstream], ignore_index=True))

    pre = gen.draw_vc_pre(target, resume=True)
    resume_pre = dict(gen.last_resume)
    pre = store.append_vc_pre_repeats(pre)

    # The FULL question table, never `target`: attach_vc_pre left-joins, so a
    # staged pass leaves vc_pre null on the questions it skipped instead of
    # dropping them from the shared table.
    questions = attach_vc_pre(questions, pre)
    store.write_questions(questions, raw=True)

    return {
        "splits": None if splits is None else sorted(str(s) for s in splits),
        "shard": None if shard is None else f"{shard[0]}/{shard[1]}",
        "n_questions": int(len(questions)),
        "n_questions_generated": int(len(target)),
        "n_answers": int(len(answers)),
        "n_classify_draws": int(len(classify)),
        "n_downstream_draws": int(len(downstream)),
        "downstream_splits": downstream_splits,
        "n_downstream_questions": int(len(downstream_q)),
        "n_pre_repeats": int(len(pre)),
        "answer_key": list(ANSWER_KEY),
        "vc_pre_key": list(VC_PRE_KEY),
        "resume": {**{f"answers[{k}]": v for k, v in resume_answers.items()},
                   "vc_pre_repeats": resume_pre},
    }


def attach_vc_pre(questions: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Summarise the R_pre repeats onto the question table.

    The spread across repeats is kept, not discarded: ``vc_pre_sd`` is what
    tells you whether the mean is describing a stable quantity or averaging
    noise, and Phase 2 reports it alongside the mean.
    """
    agg = (pre.groupby("q_id")["vc_pre"]
           .agg(vc_pre="mean", vc_pre_sd="std").reset_index())
    raw = (pre.sort_values("repeat_idx").groupby("q_id")["vc_pre_raw"]
           .first().rename("vc_pre_raw").reset_index())
    out = questions.drop(columns=[c for c in ("vc_pre", "vc_pre_sd", "vc_pre_raw")
                                  if c in questions.columns], errors="ignore")
    return out.merge(agg, on="q_id", how="left").merge(raw, on="q_id", how="left")
