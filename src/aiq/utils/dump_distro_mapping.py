import argparse
import json

from aiq.runtime.loader import get_all_aiq_entrypoints_distro_mapping


def dump_distro_mapping(path: str):
    mapping = get_all_aiq_entrypoints_distro_mapping()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    dump_distro_mapping(args.path)
