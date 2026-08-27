"""Phase 1 -- generation.

Fixed T, ``N_MAX`` draws per question, **fresh context per draw**: no chat
history, otherwise the draws are genuinely dependent rather than merely modelled
as such, and every independence statement downstream becomes untestable.

Pre-hoc and post-hoc VC are elicited in separate calls. ``vc_pre`` is asked
``R_pre`` times per question because a prospective feeling-of-knowing with no
answer to anchor on has sampling variance that is itself a measurement (8.6c).

Token statistics are recorded twice: once under the VC-augmented prompt that
produced the answer, and once by teacher-forcing the same answer tokens under a
clean prompt. Reporting only the first would confound token entropy with the VC
instruction that was added to the context to elicit VC in the first place.

Everything is cached by ``(model, dataset, q_id, draw_idx, T, prompt_variant,
seed)``; re-running is a no-op on rows already present.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import prompts
from .backends import build_lm
from .config import Config
from .parsing import parse_answer_and_vc, parse_audit, parse_vc_only
from .schemas import ANSWER_KEY, VC_PRE_KEY
from .store import Store


def derive_seed(run_seed: int, *parts: object) -> int:
    """Deterministic per-call seed. Reproducible and independent across draws."""
    key = "::".join([str(run_seed), *[str(p) for p in parts]])
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**31 - 1)


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

    def __post_init__(self) -> None:
        self.lm = build_lm(self.cfg)
        self.is_mock = self.cfg.get("model.backend") == "mock"
        self.model_name = self.cfg.get("model.name")
        self.run_seed = int(self.cfg.get("run.seed"))
        self.gen_kwargs = self.cfg.section("model.generation")

    # -- prompt plumbing ---------------------------------------------------
    def _messages(self, spec: prompts.PromptSpec, row: pd.Series,
                  **extra: object) -> list[dict]:
        msgs = spec.build(question=row["question"])
        if self.is_mock:
            msgs = _mock_meta(msgs, {"qid": row["q_id"], "ds": row["dataset"],
                                     "variant": spec.variant, "scale": spec.scale,
                                     "kind": spec.kind, **extra})
        return msgs

    def _call(self, messages: list[dict], *, temperature: float, seed: int):
        return self.lm.generate(messages, temperature=temperature, seed=seed,
                                max_tokens=self.gen_kwargs["max_tokens"],
                                stop=self.gen_kwargs.get("stop"))

    # -- post-hoc draws ----------------------------------------------------
    def draw_answers(self, questions: pd.DataFrame, *, n_max: int | None = None,
                     temperature: float | None = None,
                     prompt_variant: str | None = None,
                     store_per_position: bool = True) -> pd.DataFrame:
        n_max = n_max if n_max is not None else int(self.cfg.get("generation.n_max"))
        temperature = (temperature if temperature is not None
                       else float(self.cfg.get("generation.temperature")))
        variant = prompt_variant or self.cfg.get("generation.prompt_variant_post")
        spec = prompts.get(variant)
        clean_spec = prompts.get(self.cfg.get("generation.prompt_variant_clean"))
        do_clean = bool(self.cfg.get("generation.token_stats.teacher_force_clean_prompt"))
        keep_frac = float(self.cfg.get("generation.token_stats.store_per_position_fraction"))
        rng = np.random.default_rng(self.run_seed)

        rows: list[dict] = []
        per_position: list[dict] = []
        parsed_all = []
        for _, q in questions.iterrows():
            keep_positions = store_per_position and (rng.random() < keep_frac)
            for draw_idx in range(n_max):
                seed = derive_seed(self.run_seed, q["q_id"], draw_idx, temperature, variant)
                msgs = self._messages(spec, q)
                gen = self._call(msgs, temperature=temperature, seed=seed)
                parsed = parse_answer_and_vc(gen.text, spec.scale)
                parsed_all.append(parsed)

                row: dict = {
                    "q_id": q["q_id"], "dataset": q["dataset"], "split": q["split"],
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
                        "q_id": q["q_id"], "draw_idx": draw_idx,
                        "entropies": gen.stats.entropies.tolist(),
                        "chosen_probs": gen.stats.chosen_probs.tolist(),
                    })
                rows.append(row)

        df = pd.DataFrame(rows)
        audit = parse_audit(parsed_all)
        audit["prompt_variant"] = variant
        audit["temperature"] = temperature
        self.store.write_json(f"parse_audit__{variant}__T{temperature}", audit)
        if per_position:
            self.store.write_processed(f"per_position__{variant}",
                                       pd.DataFrame(per_position))
        return df

    # -- pre-hoc elicitation ----------------------------------------------
    def draw_vc_pre(self, questions: pd.DataFrame, *, r_pre: int | None = None,
                    prompt_variant: str | None = None) -> pd.DataFrame:
        r_pre = r_pre if r_pre is not None else int(self.cfg.get("generation.r_pre"))
        variant = prompt_variant or self.cfg.get("generation.prompt_variant_pre")
        spec = prompts.get(variant)
        if spec.kind != "pre":
            raise ValueError(f"{variant!r} is not a pre-hoc prompt; vc_pre must be "
                             "elicited with no answer in context")
        temperature = float(self.cfg.get("generation.temperature"))

        rows, parsed_all = [], []
        for _, q in questions.iterrows():
            for repeat_idx in range(r_pre):
                seed = derive_seed(self.run_seed, "pre", q["q_id"], repeat_idx, variant)
                msgs = self._messages(spec, q)
                gen = self._call(msgs, temperature=temperature, seed=seed)
                parsed = parse_vc_only(gen.text, spec.scale)
                parsed_all.append(parsed)
                rows.append({
                    "q_id": q["q_id"], "dataset": q["dataset"],
                    "repeat_idx": repeat_idx, "vc_pre": parsed.vc,
                    "vc_pre_raw": parsed.raw, "prompt_variant": variant,
                    "seed": seed, "model": self.model_name,
                })
        audit = parse_audit(parsed_all)
        audit["prompt_variant"] = variant
        self.store.write_json(f"parse_audit__{variant}", audit)
        return pd.DataFrame(rows)


def run_phase1(cfg: Config, store: Store, questions: pd.DataFrame) -> dict:
    """Generate and cache everything Phase 1 owes downstream phases."""
    gen = Generator(cfg, store)

    answers = gen.draw_answers(questions)
    answers = store.append_answers(answers)

    pre = gen.draw_vc_pre(questions)
    pre = store.append_vc_pre_repeats(pre)

    questions = attach_vc_pre(questions, pre)
    store.write_questions(questions, raw=True)

    return {
        "n_questions": int(len(questions)),
        "n_answers": int(len(answers)),
        "n_pre_repeats": int(len(pre)),
        "answer_key": list(ANSWER_KEY),
        "vc_pre_key": list(VC_PRE_KEY),
    }


def attach_vc_pre(questions: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Summarise the R_pre repeats onto the question table.

    The spread across repeats is kept, not discarded: if the same question
    yields 0.6 and 0.9 on different draws, the feeling of knowing is not a
    stable quantity (protocol 8.6c).
    """
    agg = (pre.groupby("q_id")["vc_pre"]
           .agg(vc_pre="mean", vc_pre_sd="std").reset_index())
    raw = (pre.sort_values("repeat_idx").groupby("q_id")["vc_pre_raw"]
           .first().rename("vc_pre_raw").reset_index())
    out = questions.drop(columns=[c for c in ("vc_pre", "vc_pre_sd", "vc_pre_raw")
                                  if c in questions.columns], errors="ignore")
    return out.merge(agg, on="q_id", how="left").merge(raw, on="q_id", how="left")
