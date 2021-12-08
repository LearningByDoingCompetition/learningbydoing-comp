import numpy as np
from pathlib import Path

from lbd_comp.evaluate_track2 import evaluate_track2


def test_robo_eval():

    pw_controller_path = Path(
        "data",
        "ROBO",
        "controllers",
        "prize_winners",
    )
    submissions_to_test = [
        (Path(pw_controller_path / "1_Ajoo" / "controller"), 0.918),
        (Path(pw_controller_path / "2_TeamQ" / "controller"), 16.121),
        (Path(pw_controller_path / "3_jmunozb" / "controller"), 29.539),
    ]
    target_traj_path = Path(
        "data",
        "ROBO",
        "target_trajectories",
    )

    for s in submissions_to_test:
        submission_path, score_gt = s
        robo_eval_results = evaluate_track2(
            submission_path,
            target_traj_path,
            verbose=False,
        )
        score_eval = robo_eval_results['loss']

        assert np.isclose(
            score_eval,
            score_gt,
            rtol=0.,
            atol=5.e-4,
        ), (
            f"Unit test failed: Expected submission \"{submission_path}\""
            f"to have score {score_gt}, but it has score {score_eval}."
        )

    return True
