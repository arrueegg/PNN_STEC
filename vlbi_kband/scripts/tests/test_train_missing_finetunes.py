import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_missing_finetunes as tmf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def test_session_days_are_2014_plus_and_nonempty():
    days = tmf.session_days(DATA_DIR, min_year=2014)
    assert days, "expected a non-empty set of (year, doy) days"
    # All days respect the floor; structure is (int year, int doy 1..366).
    for year, doy in days:
        assert year >= 2014
        assert 1 <= doy <= 366


def test_session_days_excludes_pre_2014():
    days = tmf.session_days(DATA_DIR, min_year=2014)
    years = {y for y, _ in days}
    assert all(y >= 2014 for y in years)
    # The data dir contains 2002-2008 sessions; none of their years may appear.
    assert not ({2002, 2003, 2004, 2005, 2006, 2007, 2008} & years)


def test_missing_days_filters_out_trained(monkeypatch):
    days = {(2024, 123), (2099, 1)}

    # Pretend only 2024-123 is already trained.
    def fake_is_trained(base_config, year, doy):
        return (year, doy) == (2024, 123)

    monkeypatch.setattr(tmf, "is_trained", fake_is_trained)
    missing = tmf.missing_days(days, base_config="config/config.yaml")
    assert missing == [(2099, 1)]


def test_is_trained_raises_on_missing_base_config():
    with pytest.raises(FileNotFoundError):
        tmf.is_trained("config/does_not_exist.yaml", 2024, 123)
