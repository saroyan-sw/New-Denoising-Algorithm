"""Verify actual preflight artifacts without rerunning scores."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    mechanical = json.loads((ROOT / "results/mechanical_controls.json").read_text())
    for row in mechanical["rows"]:
        if "candidates" in row:
            assert row["averaged_subtraction_mse"] > row["noisy_mse"]
            assert abs(row["averaged_subtraction_mse"] - row["theoretical_average_mse"]) < .0003
        else:
            assert abs(row["difference_expected_sigma_squared"] - .01) < .0003
    scores = json.loads((ROOT / "results/musiq_controls.json").read_text())
    assert len(scores["rows"]) == 16 and scores["clean_used_for_training"] is False
    values = {(r["pattern"], r["variant"]): r for r in scores["rows"]}
    assert values[("gray", "noise_005")]["score_higher_is_better"] > values[("gray", "clean_eval_only")]["score_higher_is_better"]
    assert values[("gray", "noise_005")]["iqa_only_clipped_fraction"] == 0
    checks = json.loads((ROOT / "results/prototype_shared_checks.json").read_text())
    assert checks["noise_conditioning"] == "shared_across_collection"
    assert checks["spline_prior_parameters"] == 2960
    assert checks["heldout_target_perturbation_invariance"] and checks["checkpoint_roundtrip"]
    state = json.loads((ROOT / "experiment_state.json").read_text())
    assert state["full_hybrid_training"] == "not_started"
    assert state["constraints"]["noise_strength"] == "one shared unknown level"
    assert (ROOT / state["checkpoint"]).is_file()
    print("Mechanical controls, 16 MUSIQ scores, shared-KAN wiring, checkpoint, and declared scope verified")


if __name__ == "__main__":
    main()
