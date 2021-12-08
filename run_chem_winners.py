import sys
from pathlib import Path

from lbd_comp.evaluate_track1 import evaluate_track1

def run_chem_winners(input_args):

    this_file_dir = Path(__file__).parent.absolute()
    prize_winners_dir = this_file_dir.joinpath(
        "data",
        "CHEM",
        "prize_winners",
    )
    output_dir = this_file_dir.joinpath(
        "output",
        "chem_winners",
    )

    prize_winners = [
        "Ajoo",
        "TeamQ",
        "GuineaPig",
    ]

    print("LearningByDoing: Track CHEM Final Results")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()

    for idx_w, w in enumerate(prize_winners):
        
        rank = idx_w+1
        print(f"Rank {rank}: {w}")
        w_dir = f"{rank}_{w}"
        chem_eval_results = evaluate_track1(
            prize_winners_dir.joinpath(w_dir),
            output_dir.joinpath(w_dir),
            force_output=True,
            verbose=True,
        )
        print()


if __name__ == "__main__":
    run_chem_winners(sys.argv[1:])
