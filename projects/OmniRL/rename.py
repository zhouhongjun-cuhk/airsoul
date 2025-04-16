import os
from pathlib import Path
import argparse

def rename_npy_files(directory="."):
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        print(f"[ERROR] Directory {dir_path} does not exist.")
        return

    rename_map = {
        "observation": "observations",
        "action_ref": "actions_ref",
        "reward": "rewards",
        "tag": "tags",
        "action_label": "actions_label",
        "prompt": "prompts"
    }

    for file_path in dir_path.glob("**/*.npy"):
        filename = file_path.stem
        parent_dir = file_path.parent

        if filename in rename_map:
            new_filename = f"{rename_map[filename]}.npy"
            new_file_path = parent_dir / new_filename

            if new_file_path.exists():
                print(f"[WARNING] {new_file_path} already exists, skipping {file_path}")
                continue

            try:
                file_path.rename(new_file_path)
                print(f"[INFO] Renamed {file_path} to {new_file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to rename {file_path} to {new_file_path}: {e}")
        else:
            print(f"[INFO] No rename rule for {file_path}, skipping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename .npy files by adding 's' to specific names.")
    parser.add_argument("--directory", type=str, default=".", help="Directory containing .npy files to rename.")
    args = parser.parse_args()
    rename_npy_files(args.directory)
