import argparse

def parse_arguments():
    args = argparse.ArgumentParser()

    args.add_argument('-mnist', type=str)
    args.add_argument('-checkpoint', type=str)

    return args.parse_args()