"""Pure boundary checks for the frozen five-brief training selector."""
from decimal import Decimal

from scripts.thursday_midday_select import brief_valid, objective, ranked


def values(wns: str, lut: str, requirement: str = "1.5") -> dict[str, Decimal]:
    return {"wns": Decimal(wns), "lut": Decimal(lut),
            "requirement": Decimal(requirement)}


def test_exact_guards_use_b2_without_noise_relaxation() -> None:
    start = values("-1.0", "1000")
    boundary = values("-1.1", "1010")
    assert brief_valid("T", start, boundary)
    assert brief_valid("A", start, boundary)
    assert not brief_valid("A0", start, boundary)
    assert not brief_valid("T0", start, boundary)
    assert not brief_valid("B", start, boundary)
    assert not brief_valid("A", start, values("-1.1000001", "1000"))
    assert all(brief_valid(name, start, values("-1.0", "1000"))
               for name in ("T", "A", "A0", "T0", "B"))


def test_balanced_objective_is_percent_fmax_gain_plus_lut_cut() -> None:
    start = values("-0.5", "1000")
    candidate = values("0.0", "900")
    expected = Decimal(100) * (Decimal(2) / Decimal("1.5") - 1) + Decimal(10)
    assert objective("B", start, candidate) == expected


def test_objective_ties_resolve_by_dcp_hash() -> None:
    start = values("-1.0", "1000")
    a = dict(values("-0.8", "980"), hash="a")
    b = dict(values("-0.8", "970"), hash="b")
    assert [item["hash"] for item in ranked("T", start, [b, a], True)] == ["a", "b"]
    assert [item["hash"] for item in ranked("A", start, [a, b], True)] == ["b", "a"]
