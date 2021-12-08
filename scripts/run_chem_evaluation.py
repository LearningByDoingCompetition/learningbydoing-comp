import argparse
import sys

from lbd_comp.evaluate_track1 import evaluate_track1


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Specifies the directory of a CHEM submission.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Specifies the output directory to save results. "
            "If no argument is provided, no results will be saved."
        ),
    )
    parser.add_argument(
        "-f",
        "--force-output",
        action="store_true",
        default=False,
        help="Force overwritting of existing results in the output directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enables verbose output.",
    )
    args = parser.parse_args(input_args)
    return args


def run_chem_evaluation(input_args):
    args = parse_arguments(input_args)

    chem_eval_results = evaluate_track1(
        args.input_dir,
        output_dir=args.output_dir,
        force_output=args.force_output,
        verbose=args.verbose,
    )

    return chem_eval_results


if __name__ == "__main__":
    run_chem_evaluation(sys.argv[1:])
