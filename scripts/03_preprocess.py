import argparse
import yaml
from preprocessor.preprocessor import Preprocessor

def load_args():
    parser = argparse.ArgumentParser(
        description="preprocess data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_path", type=str,
                        help="filename of preprocess yaml file.")
    parser.add_argument("--num_workers", type=int, default=10,required=False,
                        help="number of workers for multiprocessing.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = load_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    print("start preprocess.")
    Preprocessor(config).build_from_path()
    # print("start augment.")
    # Augmenter(config).build_from_path()