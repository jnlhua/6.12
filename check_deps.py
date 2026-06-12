#!/usr/bin/env python
"""
精简检查依赖包版本
"""

import importlib.metadata
import sys

# 项目所需依赖
DEPS = [
    'torch',
    'torchvision',
    'transformers',
    'diffusers',
    'accelerate',
    'tensorboardX',
    'scipy',
    'Pillow',
    'tqdm',
    'gradio',
    'numpy',
]


def main():
    print("=" * 50)
    print("Claude项目依赖包版本检查")
    print("=" * 50)
    print(f"Python版本: {sys.version.split()[0]}")
    print("-" * 50)

    all_ok = True
    for pkg in DEPS:
        try:
            version = importlib.metadata.version(pkg)
            print(f"{pkg:<15} → {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg:<15} → ❌ 未安装")
            all_ok = False

    print("-" * 50)
    if all_ok:
        print("✅ 所有依赖已安装!")
    else:
        print("⚠️  部分依赖缺失，请使用以下命令安装:")
        print(
            "\n   pip install torch torchvision transformers diffusers accelerate tensorboardX scipy Pillow tqdm gradio numpy")
    print("=" * 50)


if __name__ == "__main__":
    main()
