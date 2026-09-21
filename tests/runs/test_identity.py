"""Run identity: same experiment means same id, different experiment means different id."""

from __future__ import annotations

import copy

from stec.runs import identity


def config() -> dict:
    return {
        "mode": "finetune",
        "target": "stec",
        "year": "2024",
        "doy": "132",
        "random_seed": 42,
        "model": {
            "model_type": "BayesianResNetSTEC",
            "hidden_dim": 1024,
            "prior_sigma": 0.1,
        },
        "finetune": {"learning_rate": 1e-4, "batchsize": 2048, "num_workers": 12},
        "training": {
            "loss_function": "GaussianNLLLoss",
            "kl_annealing": {"enabled": True, "warmup_epochs": 5, "end_weight": 0.1},
        },
        "data": {
            "SH_degree": 5,
            "use_SWI": True,
            "GNSS_data_path": "/home/space/data/iono/STEC_DB_CASDCB",
            "scratch_dir": "/scratch2/arrueegg/WP4/PNN_STEC/data/",
        },
        "output_dir": "experiments/Finetune_STEC_2024_132_BayesianResNetSTEC_h1024",
    }


def test_id_is_stable_across_repeated_calls():
    assert identity.run_id(config()) == identity.run_id(config())


def test_machine_specific_paths_do_not_change_identity():
    """The same experiment run on another host is the same experiment."""
    moved = config()
    moved["data"]["GNSS_data_path"] = "/elsewhere/STEC_DB_CASDCB"
    moved["data"]["scratch_dir"] = "/tmp/whatever/"
    moved["output_dir"] = "experiments/some_other_directory_name"
    moved["finetune"]["num_workers"] = 4
    assert identity.run_id(moved) == identity.run_id(config())


def test_a_hyperparameter_change_changes_identity():
    changed = config()
    changed["finetune"]["learning_rate"] = 2e-4
    assert identity.run_id(changed) != identity.run_id(config())


def test_a_nested_training_change_changes_identity():
    """The KL warmup is not in the paper's hyperparameter table; it must still count."""
    changed = config()
    changed["training"]["kl_annealing"]["warmup_epochs"] = 10
    assert identity.run_id(changed) != identity.run_id(config())


def test_seed_is_part_of_identity():
    changed = config()
    changed["random_seed"] = 7
    assert identity.run_id(changed) != identity.run_id(config())


def test_label_is_readable_and_names_the_day():
    assert identity.run_id(config()).startswith(
        "finetune-stec-2024132-BayesianResNetSTEC-"
    )


def test_pretrain_label_omits_the_day():
    pretrain = config()
    pretrain["mode"] = "pretrain"
    assert identity.run_id(pretrain).startswith("pretrain-stec-BayesianResNetSTEC-")


def test_canonical_config_drops_only_volatile_keys():
    canonical = identity.canonical_config(config())
    assert "output_dir" not in canonical
    assert "GNSS_data_path" not in canonical["data"]
    assert canonical["data"]["SH_degree"] == 5
    assert canonical["random_seed"] == 42


def test_deep_copy_is_not_required_by_the_caller():
    original = config()
    snapshot = copy.deepcopy(original)
    identity.run_id(original)
    assert original == snapshot, "computing an id must not mutate the config"
