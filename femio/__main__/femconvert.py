import argparse
from datetime import datetime as dt
import pathlib

from .. import fem_data as fd


def main():
    parser = argparse.ArgumentParser(
        description="Convert FEM file format.")
    parser.add_argument(
        'input_type',
        type=str,
        help='Input file type')
    parser.add_argument(
        'output_type',
        type=str,
        help='Output file type')
    parser.add_argument(
        'input_path',
        type=pathlib.Path,
        help='Input file path')
    parser.add_argument(
        '-o', '--output-directory',
        type=pathlib.Path,
        default=None,
        help='Output directory path')

    args = parser.parse_args()

    if args.input_path.is_dir():
        input_directory = args.input_path
        fem_data = fd.FEMData.read_directory(
            args.input_type, input_directory, read_npy=False)
    elif args.input_path.is_file():
        input_directory = args.input_path.parent
        fem_data = fd.FEMData.read_files(
            args.input_type, [args.input_path])
    else:
        raise ValueError(
            f"{args.input_path} is neither directory nor file.")

    if args.output_directory is None:
        args.output_directory = input_directory
    date_string = dt.now().isoformat().replace('T', '_').replace(':', '-')
    fem_data.write(
        args.output_type, args.output_directory / ('out_' + date_string),
        overwrite=False)


if __name__ == '__main__':
    main()
