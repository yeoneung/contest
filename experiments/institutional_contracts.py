#!/usr/bin/env python3
"""Check public award allocations and illustrate alternative pool accounting.

The inputs describe announced contracts, not realized payments or outcomes.
The accounting counterfactuals do not estimate discovery-speed effects.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def calculate():
    path = ROOT / "data/institutional_awards.csv"
    amounts = ["total_budget_usd", "rank1_usd", "rank2_usd", "rank3_usd",
               "sharing_award_usd", "sharing_award_slots"]
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in amounts:
            row[key] = int(row[key])
            assert row[key] >= 0
        row["edition"] = int(row["edition"])
        row["final_pool_usd"] = sum(row[key] for key in
                                     ["rank1_usd", "rank2_usd", "rank3_usd"])
        row["maximum_sharing_pool_usd"] = row["sharing_award_usd"] * row["sharing_award_slots"]
        assert row["final_pool_usd"] + row["maximum_sharing_pool_usd"] == row["total_budget_usd"]
        row["sharing_budget_percent"] = 100 * row["maximum_sharing_pool_usd"] / row["total_budget_usd"]
    assert {row["edition"] for row in rows} == {2025, 2026}
    assert len({row["total_budget_usd"] for row in rows}) == 1
    initial = next(row for row in rows if row["edition"] == 2025)
    cap, bonus, slots = (initial[key] for key in
                         ["total_budget_usd", "sharing_award_usd", "sharing_award_slots"])
    counterfactuals = []
    for released in range(slots + 1):
        for rule in ["fixed_reservation", "recycling"]:
            final_pool = cap - bonus * (slots if rule == "fixed_reservation" else released)
            total = final_pool + released * bonus
            assert 0 < final_pool <= total <= cap
            counterfactuals.append(dict(rule=rule, paid_release_awards=released,
                                       final_pool_usd=final_pool, total_awards_usd=total))
    return dict(date="2026-11-21", source_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                announced_contracts=rows, pool_accounting_counterfactuals=counterfactuals,
                interpretation="Published allocations are documentary evidence. Counterfactuals vary the treatment of unpaid release awards; they are not observed policy changes, calibrated race outcomes, or causal estimates.")


if __name__ == "__main__":
    result = calculate()
    path = ROOT / "results/institutional_contracts.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
