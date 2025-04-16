import site
import os

# 获取 site-packages 目录列表
site_packages_dirs = site.getsitepackages()

for site_packages_dir in site_packages_dirs:
    if os.path.isdir(site_packages_dir):
        # 遍历 site-packages 目录下的所有文件
        for root, dirs, files in os.walk(site_packages_dir):
            for file in files:
                if file.endswith('.pth'):
                    pth_file_path = os.path.join(root, file)
                    print(f"Found .pth file: {pth_file_path}")
                    try:
                        with open(pth_file_path, 'r') as pth_file:
                            # 读取 .pth 文件中的每一行路径
                            for line in pth_file:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    print(f"  Path in .pth: {line}")
                    except Exception as e:
                        print(f"Error reading {pth_file_path}: {e}")
