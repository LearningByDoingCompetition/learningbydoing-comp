import argparse
import sys

from lbd_comp.track2.training_data import generate_training_data

def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Specifies the output directory to save ROBO training data."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-trajs",
        type=int,
        default=50,
        help=(
            "Specifies the number of training trajectories per system."
        ),
    )
    parser.add_argument(
        "-f",
        "--force-output",
        action="store_true",
        default=False,
        help="Force overwritting of existing files in the output directory.",
    )
    parser.add_argument(
        "-d",
        "--debug-output",
        action="store_true",
        default=False,
        help="Generates debug information, such as gif visualizations.",
    )
    args = parser.parse_args(input_args)
    return args

def gen_robo_training_data(input_args):

    args = parse_arguments(input_args)
    generate_training_data(
        args.output_dir,
        args.num_trajs,
        args.force_output,
        args.debug_output,
    )


if __name__ == "__main__":
    gen_robo_training_data(sys.argv[1:])
