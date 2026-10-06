"""The positioning VTEC arm must map back with the convention its training VTEC used (SLM)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "positioning" / "scripts" / "generate_stec_corrections.py"


def test_vtec_arm_uses_shared_slm_constant() -> None:
    source = SCRIPT.read_text()
    assert 'MappingFunction(mapping_type="MSLM")' not in source
    assert "MappingFunction(mapping_type=VTEC_MAPPING_TYPE)" in source


def test_shared_vtec_mapping_constant_is_slm() -> None:
    spec = importlib.util.find_spec("stec.inference.run_baselines")
    assert spec is not None
    from stec.inference.run_baselines import VTEC_MAPPING_TYPE

    assert VTEC_MAPPING_TYPE == "SLM"
