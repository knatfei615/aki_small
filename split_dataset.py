# -*- coding: utf-8 -*-
# 将合成的 AKI 数据集拆分为 Kaggle 竞赛常见结构
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


def make_kaggle_splits(
    input_path: str,
    test_size: float = 0.2,
    target_col: str = "aki_48h",
    random_state: int = 42,
    output_dir: str = "kaggle_dataset",
    sample_fill: float = 0.0,
) -> dict[str, str]:
    """
    将单一 CSV 拆分为 Kaggle 常见的 data/ 目录结构:
    - train.csv: 包含特征 + 目标
    - test.csv: 仅包含特征（去掉目标列）
    - solution.csv: 仅评测用，包含 test 集的 id + 真实标签
    - sample_submission.csv: 参赛者提交格式模板 (id + 目标列，占位值)

    返回:
        dict，键为文件名，值为输出路径
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size 应位于 (0, 1) 区间内")

    df = pd.read_csv(input_path)
    if "id" not in df.columns:
        raise ValueError("数据集中缺少 Kaggle 常用的 'id' 列")
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据集中")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    solution_path = os.path.join(output_dir, "solution.csv")
    sample_path = os.path.join(output_dir, "sample_submission.csv")

    # 保存 train（保留目标列）
    train_df.to_csv(train_path, index=False)

    # 保存 test（去除目标列）
    test_features = test_df.drop(columns=[target_col])
    test_features.to_csv(test_path, index=False)

    # solution（id + 真实标签，留作私下评测）
    solution_df = test_df[["id", target_col]].sort_values("id")
    solution_df["Usage"] = "public"  # 新增 Usage 列，全部填充为 public
    solution_df.to_csv(solution_path, index=False)

    # sample submission（同列，填充占位值）
    sample_df = solution_df.copy()
    sample_df[target_col] = sample_fill
    sample_df.to_csv(sample_path, index=False)

    return {
        "train": train_path,
        "test": test_path,
        "solution": solution_path,
        "sample_submission": sample_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 nephro_small.csv 拆分为 Kaggle data 结构"
    )
    parser.add_argument(
        "--input",
        default="nephro_small.csv",
        help="输入 CSV 文件路径 (默认: nephro_small.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集占比 (默认: 0.2)",
    )
    parser.add_argument(
        "--target",
        default="aki_48h",
        help="目标列名，用于分层抽样 (默认: aki_48h)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="splits",
        help="输出目录 (默认: ./kaggle_dataset)",
    )
    parser.add_argument(
        "--sample-fill",
        type=float,
        default=0.0,
        help="sample_submission 中目标列的占位值 (默认: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        out_paths = make_kaggle_splits(
            input_path=args.input,
            test_size=args.test_size,
            target_col=args.target,
            random_state=args.random_state,
            output_dir=args.output_dir,
            sample_fill=args.sample_fill,
        )
    except Exception as exc:  # 捕获异常并打印
        print(f"✗ 拆分失败: {exc}", file=sys.stderr)
        sys.exit(1)

    print("✓ Kaggle 数据集已生成")
    for name, path in out_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

