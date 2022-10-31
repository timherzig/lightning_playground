import argparse
from ast import arg

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cv_dir', type=str)
    parser.add_argument('-log_dir', type=str)
    parser.add_argument('-config', type=str)

    return parser.parse_args()