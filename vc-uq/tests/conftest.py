from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vc_uq.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """Config pointed at a scratch directory, sized for fast tests."""
    return load_config(overrides=[
        f"run.data_root={tmp_path.as_posix()}/data",
        f"run.results_root={tmp_path.as_posix()}/results",
        "run.name=test",
        "model.backend=mock", "embedding.backend=mock", "nli.backend=mock",
        "generation.n_max=8", "generation.r_pre=3",
        "dataset.triviaqa.n_questions=30", "dataset.fabricated.n_questions=10",
        "phase2.bootstrap.n_resamples=50",
        "phase0.n_hand_label=60", "phase0.null_band.n_mismatched_pairs=200",
    ])


@pytest.fixture
def answers_frame():
    """A small hand-built answer table with known properties.

    Two questions: one where VC is constant across draws (so within-question
    AUROC must be 0.5) and one where correctness is perfectly correlated across
    draws (so the within-question error correlation must be positive).
    """
    rows = []
    for q_id, corrects, vcs in [
        ("q0", [True, False, True, False], [0.8, 0.8, 0.8, 0.8]),
        ("q1", [False, False, False, False], [0.9, 0.9, 0.9, 0.9]),
        ("q2", [True, True, True, True], [0.7, 0.7, 0.7, 0.7]),
    ]:
        for i, (c, v) in enumerate(zip(corrects, vcs)):
            rows.append({"q_id": q_id, "dataset": "d", "split": "eval",
                         "draw_idx": i, "answer": f"{q_id}-{i}",
                         "vc_post": v, "correct": c, "cluster_id": 0 if c else 1,
                         "h_tok_mean": 1.0, "min_token_p": 0.5,
                         "s_anchor": 0.1, "temperature": 0.8,
                         "prompt_variant": "vc_post_v1", "seed": i,
                         "model": "m", "vc_post_raw": str(v)})
    return pd.DataFrame(rows)
