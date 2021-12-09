import argparse
import sys

from lbd_comp.track1.training_data import generate_training_data


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
            "Specifies the output directory to save CHEM training data."
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
        "-z",
        "--zip-data",
        action="store_true",
        default=False,
        help=(
            "If true, outputs training data into a zip archive instead of a directory."
        ),
    )
    args = parser.parse_args(input_args)
    return args


def gen_chem_training_data(input_args):

    args = parse_arguments(input_args)
    generate_training_data(
        args.output_dir,
        args.zip_data,
        args.force_output,
    )
    

if __name__ == "__main__":
    gen_chem_training_data(sys.argv[1:])
