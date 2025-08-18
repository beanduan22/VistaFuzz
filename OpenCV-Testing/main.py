import json
import numpy as np
import cv2
from tool.API_info import *  # Assuming this contains necessary info or functions
import random
from tool.opencv_args_seed_generator import *
from tool.mutation import *
from tool.parser_from_str2funcandinfo import *
from tool.load_json_file import  *
from tool.test import  *
import venn

def take_subset(data, n, random_subset=False, seed=0):

    if n <= 0:
        return data
    if isinstance(data, list):
        n = min(n, len(data))
        if random_subset:
            rng = random.Random(seed)
            idxs = rng.sample(range(len(data)), n)
            idxs.sort()
            return [data[i] for i in idxs]
        return data[:n]
    if isinstance(data, dict):
        keys = list(data.keys())
        n = min(n, len(keys))
        if random_subset:
            rng = random.Random(seed)
            sel = rng.sample(keys, n)
        else:
            sel = keys[:n]
        return {k: data[k] for k in sel}
    return data

def main():
    parser = argparse.ArgumentParser(description="VistaFuzz runner")
    parser.add_argument("--json", default="./API/OpenCV_API_filtered_subset.json",
                        help="Path to API JSON")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Mutation iterations per API")
    parser.add_argument("--max-apis", type=int, default=0,
                        help="Limit number of APIs to run (0 = all)")
    parser.add_argument("--random-subset", action="store_true",
                        help="Randomly sample max-apis from the dataset")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling")
    parser.add_argument("--api", type=str, default="",
                        help="Run a single API by name (e.g., 'cv2.findChessboardCornersSBWithMeta')")
    args = parser.parse_args()

    env_max = int(os.getenv("VISTAFUZZ_MAX_APIS", "0"))
    if args.max_apis == 0 and env_max > 0:
        args.max_apis = env_max

    data = load_json_file(args.json)

    if args.api:
        print(f"[VistaFuzz] Running single API: {args.api} (iterations={args.iterations})")
        test_one(args.api, args.iterations)
        return

    data_subset = take_subset(data, args.max_apis, args.random_subset, args.seed)

    total = len(data) if isinstance(data, list) else (len(data.keys()) if isinstance(data, dict) else 1)
    sub = len(data_subset) if isinstance(data_subset, list) else (len(data_subset.keys()) if isinstance(data_subset, dict) else 1)
    print(f"[VistaFuzz] Total APIs in dataset: {total}")
    print(f"[VistaFuzz] Selected APIs to run: {sub} "
          f"(mode={'random' if args.random_subset else 'head'}, max_apis={args.max_apis})")
    print(f"[VistaFuzz] Mutation iterations per API: {args.iterations}")

    test_list(data_subset, args.iterations)

if __name__ == "__main__":
    main()
