import sys
from pathlib import Path

# Make the script importable (it lives in ../).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_vlbi_kband import parse_year_doy_from_filename


def test_new_convention_iso_date():
    # 2024-05-01 is DOY 122.
    assert parse_year_doy_from_filename("20240501-n24jh02h.ion") == (2024, 122)


def test_new_convention_uses_iso_not_expcode_year():
    # Exp code embeds "n23" but ISO date is 2024-01-18 (DOY 18).
    assert parse_year_doy_from_filename("20240118-n23jh02i.ion") == (2024, 18)


def test_legacy_kv():
    # 2017-09-22 is DOY 265.
    assert parse_year_doy_from_filename("17SEP22KV.ion") == (2017, 265)


def test_legacy_q_band_suffix():
    # 2021-04-19 is DOY 109; suffix QL must be tolerated.
    assert parse_year_doy_from_filename("21APR19QL.ion") == (2021, 109)


def test_legacy_pre_2014_still_parses_year():
    # 2002-08-25 is DOY 237 — parsing succeeds; the 2014 cutoff is applied by main().
    assert parse_year_doy_from_filename("02AUG25KV.ion") == (2002, 237)


def test_legacy_invalid_day_returns_none():
    # Feb 30 is not a real date — the datetime guard must yield None.
    assert parse_year_doy_from_filename("14FEB30KV.ion") is None


def test_unparseable_returns_none():
    assert parse_year_doy_from_filename("filelist.txt") is None
    assert parse_year_doy_from_filename("README.md") is None
