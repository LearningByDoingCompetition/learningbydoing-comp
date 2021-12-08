import numpy as np
from pathlib import Path

from lbd_comp.evaluate_track1 import evaluate_track1


def test_chem_eval():

    submissions_to_test = [
        (Path("data","CHEM", "prize_winners", "1_Ajoo"), 0.0890),
        (Path("data","CHEM", "prize_winners", "2_TeamQ"), 0.3385),
        (Path("data","CHEM", "prize_winners", "3_GuineaPig"), 0.3386),
    ]

    for s in submissions_to_test:
        submission_path, score_gt = s
        chem_eval_results = evaluate_track1(
            submission_path,
            verbose=False,
        )
        score_eval = chem_eval_results['loss']

        assert np.isclose(
            score_eval,
            score_gt,
            rtol=0.,
            atol=1.e-4,
        ), (
            f"Unit test failed: Expected submission \"{submission_path}\""
            f"to have score {score_gt}, but it has score {score_eval}."
        )

    return True
