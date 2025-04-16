import os
from pathlib import Path
import argparse

def rename_npy_files(directory="."):
    """遍历指定目录下的 .npy 文件，仅重命名 actions_ref 和 actions_label。"""
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        print(f"[ERROR] Directory {dir_path} does not exist.")
        return

    # 定义仅针对 actions 的文件名映射
    rename_map = {
        "actions_ref": "actions_label",      # actions_ref -> actions_label
        "actions_label": "actions_behavior"  # actions_label -> actions_behavior
    }

    # 遍历目录下的所有 .npy 文件
    for file_path in dir_path.glob("**/*.npy"):  # 递归查找所有 .npy 文件
        filename = file_path.stem  # 获取文件名（不含扩展名）
        parent_dir = file_path.parent

        # 检查是否需要改名
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
    parser = argparse.ArgumentParser(description="Rename specific actions-related .npy files.")
    parser.add_argument("--directory", type=str, default=".", help="Directory containing .npy files to rename.")
    args = parser.parse_args()
    rename_npy_files(args.directory)
