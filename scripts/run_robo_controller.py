import argparse
import sys

from lbd_comp.evaluate_track2 import evaluate_track2


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Specifies the directory of a ROBO controller.",
    )
    parser.add_argument(
        "-t",
        "--traj-dir",
        type=str,
        required=True,
        help="Specifies the directory of the ROBO target trajectories.",
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
        "-d",
        "--debug-output",
        action="store_true",
        default=False,
        help="Generates debug information, such as gif visualizations.",
    )
    parser.add_argument(
        "-z",
        "--show-viz",
        action="store_true",
        default=False,
        help="Displays visualizations during controller evaluation.",
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


def run_robo_controller(input_args):
    args = parse_arguments(input_args)

    robo_eval_results = evaluate_track2(
        args.input_dir,
        args.traj_dir,
        output_dir=args.output_dir,
        force_output=args.force_output,
        debug_output=args.debug_output,
        show_viz=args.show_viz,
        verbose=args.verbose,
    )

    return robo_eval_results


if __name__ == "__main__":
    run_robo_controller(sys.argv[1:])
