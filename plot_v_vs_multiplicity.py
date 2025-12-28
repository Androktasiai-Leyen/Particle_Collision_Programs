#!/usr/bin/env python3
"""
从已有的 Y_analysis 结果文本中解析每个多重度分箱的 v2,2，
计算 v2 = sqrt(max(v2,2, 0))，并绘制“v 对多重度分箱”的图。

默认读取：Y_analysis_results/Y_analysis_results_pT0.5-5.0_eta1.1_deltaEta2.0.txt
默认输出：Y_analysis_results/v_vs_multiplicity_deltaEta_gt_2p0.png
"""

import argparse
import math
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


MULTI_BINS_ORDER: List[str] = [
    "0-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%",
]


def parse_results(results_path: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """解析结果文件，返回 (avg_n_dict, v22_dict, v33_dict)。

    文件片段样式参考：
        40-60%:
          Events: 999,073
          Average particles per event: 21.25
          Template fit parameters:
            F = ...
            G = ...
            v2_2 = 0.019140
            v3_3 = ...
    """
    avg_n: Dict[str, float] = {}
    v22: Dict[str, float] = {}
    v33: Dict[str, float] = {}

    current: str = ""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            # 匹配分箱标题，如 "40-60%:" 或 "0-20%:"，允许三位数字（如100）
            m_bin = re.match(r"^(\d+-\d+%)\:$", line)
            if m_bin:
                current = m_bin.group(1)
                continue

            if not current:
                continue

            m_avg = re.search(r"Average particles per event:\s*([0-9.]+)", line)
            if m_avg:
                try:
                    avg_n[current] = float(m_avg.group(1))
                except ValueError:
                    pass

            m_v22 = re.search(r"v2_2\s*=\s*([-0-9.]+)", line)
            if m_v22:
                try:
                    v22[current] = float(m_v22.group(1))
                except ValueError:
                    pass

            m_v33 = re.search(r"v3_3\s*=\s*([-0-9.]+)", line)
            if m_v33:
                try:
                    v33[current] = float(m_v33.group(1))
                except ValueError:
                    pass

    return avg_n, v22, v33


def plot_v_vs_multiplicity(
    v22_dict: Dict[str, float],
    output_path: str,
    title_suffix: str = "|Δη| > 2.0",
):
    """绘制 v 对多重度分箱的图。

    - x 轴：多重度分箱类别
    - y 轴：v = sqrt(max(v2,2, 0))
    """
    bins = [b for b in MULTI_BINS_ORDER if b in v22_dict]
    if not bins:
        raise RuntimeError("No multiplicity bins found in parsed results.")

    v_vals = [math.sqrt(max(v22_dict[b], 0.0)) for b in bins]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(bins)))

    ax.plot(x, v_vals, marker="o", linestyle="-", color="tab:red", label="v from v2,2")

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("v")
    ax.set_xlabel("Multiplicity percentile bin")
    ax.set_title(f"v vs multiplicity bins  {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


def plot_v_and_v33_same_figure(
    v22_dict: Dict[str, float],
    v33_dict: Dict[str, float],
    output_path: str,
    title_suffix: str = "|Δη| > 2.0",
):
    """同一张图绘制 v 与 v3,3：左轴 v，右轴 v3,3。"""
    bins = [b for b in MULTI_BINS_ORDER if b in v22_dict and b in v33_dict]
    if not bins:
        raise RuntimeError("No overlapping multiplicity bins between v22 and v33.")

    v_vals = [math.sqrt(max(v22_dict[b], 0.0)) for b in bins]
    v33_vals = [v33_dict[b] for b in bins]

    x = list(range(len(bins)))

    fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
    ax2 = ax1.twinx()

    l1 = ax1.plot(x, v_vals, marker="o", color="tab:red", label="v = sqrt(v2,2)")
    ax1.set_ylabel("v")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins)

    l2 = ax2.plot(x, v33_vals, marker="s", linestyle="--", color="tab:blue", label="v3,3")
    ax2.set_ylabel("v3,3")

    ax1.set_xlabel("Multiplicity percentile bin")
    ax1.set_title(f"v and v3,3 vs multiplicity bins  {title_suffix}")
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot v vs multiplicity bins from Y_analysis results")
    parser.add_argument(
        "--results",
        default="Y_analysis_results/Y_analysis_results_pT0.5-5.0_eta1.1_deltaEta2.0.txt",
        help="Path to the results text file"
    )
    parser.add_argument(
        "--output",
        default="Y_analysis_results/v_vs_multiplicity_deltaEta_gt_2p0.png",
        help="Output figure path"
    )
    parser.add_argument(
        "--output-v33",
        default="Y_analysis_results/v33_vs_multiplicity_deltaEta_gt_2p0.png",
        help="Output figure path for v3,3"
    )
    parser.add_argument(
        "--output-combined",
        default="Y_analysis_results/v_and_v33_vs_multiplicity_deltaEta_gt_2p0.png",
        help="Output figure path for combined v and v3,3"
    )
    parser.add_argument(
        "--title-suffix",
        default="|Δη| > 2.0",
        help="Suffix added to figure title"
    )

    args = parser.parse_args()

    avg_n, v22, v33 = parse_results(args.results)
    # v from v2,2
    plot_v_vs_multiplicity(v22, args.output, args.title_suffix)
    # v3,3 vs bins
    plot_v_vs_multiplicity(v33, args.output_v33, args.title_suffix.replace("|Δη|", "v3,3  \n|Δη|"))
    # combined figure
    plot_v_and_v33_same_figure(v22, v33, args.output_combined, args.title_suffix)


if __name__ == "__main__":
    main()


