import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds_root', type=str)
    parser.add_argument('-log_folder', type=str)