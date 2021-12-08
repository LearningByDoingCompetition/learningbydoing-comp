import sys
from pathlib import Path

from lbd_comp.evaluate_track2 import evaluate_track2

def run_robo_winners(input_args):

    this_file_dir = Path(__file__).parent.absolute()
    prize_winners_dir = this_file_dir.joinpath(
        "data",
        "ROBO",
        "controllers",
        "prize_winners",
    )
    trajectories_dir = this_file_dir.joinpath(
        "data",
        "ROBO",
        "target_trajectories",
    )
    output_dir = this_file_dir.joinpath(
        "output",
        "robo_winners",
    )

    prize_winners = [
        "Ajoo",
        "TeamQ",
        "jmunozb",
    ]

    print("LearningByDoing: Track ROBO Final Results")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()

    for idx_w, w in enumerate(prize_winners):

        rank = idx_w+1
        print(f"Rank {rank}: {w}")
        w_dir = f"{rank}_{w}"
        robo_eval_results = evaluate_track2(
            prize_winners_dir.joinpath(w_dir, "controller"),
            trajectories_dir,
            output_dir.joinpath(w_dir),
            force_output=True,
            verbose=False,
        )
        print()


if __name__ == "__main__":
    run_robo_winners(sys.argv[1:])
