"""The descriptive tables must be derived from the model, not maintained beside it."""

from __future__ import annotations

from stec.analysis.paper_tables import (
    feature_table,
    hyperparameter_table,
    to_latex,
)
from stec.data.feature_layout import layout_from_feature_control

PAPER_CONFIG = {
    "target": "stec",
    "mode": "finetune",
    "random_seed": 42,
    "model": {
        "model_type": "BayesianResNetSTEC",
        "hidden_dim": 1024,
        "num_layers": 4,
        "prior_sigma": 0.1,
        "dropout_rate": 0.0,
    },
    "finetune": {
        "learning_rate": 2e-4,
        "batchsize": 512,
        "epochs": 50,
        "scheduler": "ReduceLROnPlateau",
    },
    "training": {
        "loss_function": "GaussianNLLLoss",
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "kl_annealing": {
            "enabled": True,
            "start_weight": 0.0,
            "end_weight": 0.1,
            "warmup_epochs": 5,
        },
    },
    "data": {"SH_degree": 5, "train_subset_size": 500000},
    "feature_control": {
        "year": True,
        "doy": True,
        "sod": True,
        "local_time_hours": True,
        "lat_sta": True,
        "lon_sta": True,
        "sm_lat_sta": True,
        "sm_lon_sta": True,
        "satazi": True,
        "satele": True,
        "lat_ipp": True,
        "lon_ipp": True,
        "sm_lat_ipp": True,
        "sm_lon_ipp": True,
        "Kp_index": True,
        "R_Sunspot_No": True,
        "Dst-index,_nT": True,
        "AE-index,_nT": True,
        "ap_index,_nT": True,
        "f107_index": True,
    },
}


def test_feature_table_totals_the_real_model_input_width():
    rows = feature_table(PAPER_CONFIG)
    assert rows[-1]["feature"] == "TOTAL"
    assert rows[-1]["columns"] == 127


def test_feature_table_total_equals_the_layout_it_describes():
    """The table cannot drift from the layout, because it is derived from it."""
    layout = layout_from_feature_control(PAPER_CONFIG["feature_control"], sh_degree=5)
    assert feature_table(PAPER_CONFIG)[-1]["columns"] == layout.total_dim


def test_feature_table_follows_the_tensor_order():
    rows = [r for r in feature_table(PAPER_CONFIG) if r["feature"] != "TOTAL"]
    groups = [r["group"] for r in rows]
    # Space weather is last, after the harmonics - as the collation emits it.
    assert groups[-1] == "Space weather"
    assert "Spherical harmonics" in groups
    assert groups.index("Spherical harmonics") < groups.index("Space weather")


def test_disabling_a_feature_shrinks_the_table_and_the_total():
    reduced = {
        **PAPER_CONFIG,
        "feature_control": {
            **PAPER_CONFIG["feature_control"],
            "local_time_hours": False,
        },
    }
    assert feature_table(reduced)[-1]["columns"] == 127 - 3


def test_hyperparameter_table_carries_the_three_the_manuscript_omits():
    """These are the reason this generator exists rather than a hand-written table."""
    names = {row["parameter"] for row in hyperparameter_table(PAPER_CONFIG)}
    assert "KL weight" in names
    assert "Variance floor" in names
    assert "Output bias init" in names


def test_kl_weight_row_states_the_anneal_not_just_the_endpoint():
    rows = {r["parameter"]: r for r in hyperparameter_table(PAPER_CONFIG)}
    kl = rows["KL weight"]
    assert "0.1" in str(kl["value"])
    assert "5" in str(kl["note"]) and "anneal" in str(kl["note"]).lower()


def test_hyperparameters_come_from_the_run_mode_that_applies():
    rows = {r["parameter"]: r for r in hyperparameter_table(PAPER_CONFIG)}
    assert rows["Learning rate"]["value"] == 2e-4
    assert rows["Epochs"]["value"] == 50


def test_latex_escapes_names_that_would_break_the_document():
    rows = [{"parameter": "Dst-index,_nT", "value": 1, "note": "50% of it"}]
    latex = to_latex(rows, ["parameter", "value", "note"])
    assert r"\_" in latex
    assert r"\%" in latex
