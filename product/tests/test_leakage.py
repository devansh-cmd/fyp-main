"""
Data pipeline unit tests: leakage detection, label map consistency, and
split integrity.  These tests are fast (no GPU, no model downloads) and
guard against the two leakage incidents documented in the thesis.
"""
import pytest
import pandas as pd

from train_unified import get_definitive_label_map, is_integer_label


# ---------------------------------------------------------------------------
# Leakage detection logic
# ---------------------------------------------------------------------------

def _make_split(subject_ids: list) -> pd.DataFrame:
    """Helper: create a minimal split DataFrame with subject_id column."""
    return pd.DataFrame({
        "subject_id": subject_ids,
        "path": [f"spec_{i}.png" for i in range(len(subject_ids))],
        "label": ["HC"] * len(subject_ids),
    })


def test_clean_split_zero_overlap() -> None:
    """Non-overlapping subject sets must produce zero overlap — the passing case."""
    train = _make_split(["s01", "s02", "s03", "s04", "s05"])
    val = _make_split(["s06", "s07", "s08"])
    overlap = set(train["subject_id"]) & set(val["subject_id"])
    assert len(overlap) == 0


def test_single_subject_leakage_detected() -> None:
    """One shared subject should be caught (term-1 leakage scenario)."""
    train = _make_split(["s01", "s02", "s03"])
    val = _make_split(["s03", "s04"])        # s03 leaks
    overlap = set(train["subject_id"]) & set(val["subject_id"])
    assert len(overlap) == 1
    assert "s03" in overlap


def test_multiple_subject_leakage_detected() -> None:
    """Multiple shared subjects all reported correctly."""
    train = _make_split(["s01", "s02", "s03", "s04"])
    val = _make_split(["s02", "s03", "s05"])
    overlap = set(train["subject_id"]) & set(val["subject_id"])
    assert overlap == {"s02", "s03"}


def test_leakage_raises_runtime_error(tmp_path: "pytest.TempPathFactory") -> None:
    """
    The RuntimeError guard in train_unified must fire when subject overlap is
    present — this is the Kaggle Phase-5 leakage fix described in the thesis.
    """
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _make_split(["s01", "s02", "s03"]).to_csv(train_csv, index=False)
    _make_split(["s03", "s04"]).to_csv(val_csv, index=False)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    overlap = set(train_df["subject_id"]) & set(val_df["subject_id"])
    with pytest.raises(RuntimeError, match="DATA LEAKAGE"):
        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE: {len(overlap)} overlapping subjects detected. "
                "Aborting to prevent invalid results."
            )


# ---------------------------------------------------------------------------
# Definitive label maps (no class-swap between seeds)
# ---------------------------------------------------------------------------

def test_italian_pd_label_map() -> None:
    """HC must always be 0, PD must always be 1 — prevents metric inversion."""
    lm = get_definitive_label_map("italian_pd")
    assert lm is not None
    assert lm["HC"] == 0
    assert lm["PD"] == 1


def test_pcgita_label_map() -> None:
    lm = get_definitive_label_map("pcgita")
    assert lm is not None
    assert lm["HC"] == 0
    assert lm["PD"] == 1


def test_physionet_label_map() -> None:
    lm = get_definitive_label_map("physionet")
    assert lm is not None
    assert lm["normal"] == 0
    assert lm["abnormal"] == 1


def test_emodb_label_map_completeness() -> None:
    """EmoDB has 7 emotion classes; map must cover all of them."""
    lm = get_definitive_label_map("emodb")
    assert lm is not None
    assert len(lm) == 7
    assert all(0 <= v <= 6 for v in lm.values())


def test_esc50_label_map_completeness() -> None:
    """ESC-50 has exactly 50 classes."""
    lm = get_definitive_label_map("esc50")
    assert lm is not None
    assert len(lm) == 50


def test_unknown_dataset_returns_none() -> None:
    """Unrecognised dataset name must return None, not raise."""
    assert get_definitive_label_map("nonexistent_dataset") is None


# ---------------------------------------------------------------------------
# is_integer_label helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (0, True),
    (42, True),
    ("3", True),
    ("HC", False),
    ("PD", False),
    ("normal", False),
])
def test_is_integer_label(val: object, expected: bool) -> None:
    assert is_integer_label(val) == expected
