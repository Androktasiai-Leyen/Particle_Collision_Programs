#!/usr/bin/env python3
"""
各向异性流分析脚本
计算并绘制v2和v3随多重度区间的变化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_data_file(filepath):
    """解析数据文件，提取v2_2和v3_3参数"""
    data = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取pT范围
    pT_match = re.search(r'pT ∈ \[([0-9.]+), ([0-9.]+)\]', content)
    if pT_match:
        pT_min, pT_max = float(pT_match.group(1)), float(pT_match.group(2))
        data['pT_range'] = (pT_min, pT_max)
    
    # 提取Δη范围
    delta_eta_match = re.search(r'\|Δη\| > ([0-9.]+)', content)
    if delta_eta_match:
        data['delta_eta'] = float(delta_eta_match.group(1))
    
    # 提取多重度区间数据
    multiplicity_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    for mult_range in multiplicity_ranges:
        # 查找该多重度区间的数据
        pattern = rf'{mult_range}:(.*?)(?=\n\d+-\d+%:|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            section = match.group(1)
            
            # 提取v2_2和v3_3
            v2_match = re.search(r'v2_2 = ([+-]?[0-9.]+)', section)
            v3_match = re.search(r'v3_3 = ([+-]?[0-9.]+)', section)
            
            if v2_match and v3_match:
                data[mult_range] = {
                    'v2_2': float(v2_match.group(1)),
                    'v3_3': float(v3_match.group(1))
                }
    
    return data

def calculate_vn(vn_n, pT1, pT2):
    """
    计算v_n值
    使用公式: v_n(pT1) = v_n,n(pT1, pT2) / sqrt(v_n,n(pT2, pT2))
    这里假设pT1 = pT2，所以公式简化为: v_n = v_n,n / sqrt(v_n,n)
    """
    if vn_n == 0:
        return 0
    
    # 由于pT1 = pT2，公式简化为 v_n = sqrt(v_n,n)
    return np.sqrt(abs(vn_n)) * np.sign(vn_n)

def main():
    # 数据文件路径
    data_dir = Path('/Users/androktasiaileyen/ffss/Ypp')
    
    # 所有数据文件
    files = [
        'Y_analysis_results_pT0.2-1.0_eta1.1_deltaEta1.0.txt',
        'Y_analysis_results_pT0.2-1.0_eta1.1_deltaEta1.5.txt',
        'Y_analysis_results_pT0.2-1.0_eta1.1_deltaEta2.0.txt',
        'Y_analysis_results_pT1.0-3.0_eta1.1_deltaEta1.0.txt',
        'Y_analysis_results_pT1.0-3.0_eta1.1_deltaEta1.5.txt',
        'Y_analysis_results_pT1.0-3.0_eta1.1_deltaEta2.0.txt'
    ]
    
    # 解析所有数据
    all_data = {}
    for file in files:
        filepath = data_dir / file
        if filepath.exists():
            data = parse_data_file(filepath)
            key = f"pT{data['pT_range'][0]}-{data['pT_range'][1]}_deltaEta{data['delta_eta']}"
            all_data[key] = data
            print(f"解析文件: {file}")
            print(f"  pT范围: {data['pT_range']}")
            print(f"  Δη: {data['delta_eta']}")
            print(f"  多重度区间数量: {len([k for k in data.keys() if k.endswith('%')])}")
            print()
    
    # 组织数据用于绘图
    pT_02_10_data = {}  # pT 0.2-1.0 GeV的数据
    pT_10_30_data = {}  # pT 1.0-3.0 GeV的数据
    
    for key, data in all_data.items():
        pT_range = data['pT_range']
        delta_eta = data['delta_eta']
        
        if pT_range == (0.2, 1.0):
            pT_02_10_data[delta_eta] = data
        elif pT_range == (1.0, 3.0):
            pT_10_30_data[delta_eta] = data
    
    # 多重度区间
    multiplicity_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    multiplicity_centers = [10, 30, 50, 70, 90]  # 多重度区间的中心值
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 颜色和线型
    colors = ['red', 'blue', 'green']
    linestyles = ['-', '--', '-.']
    delta_etas = [1.0, 1.5, 2.0]
    
    # 绘制pT 0.2-1.0 GeV的图
    ax1.set_title('各向异性流分析 - pT ∈ [0.2, 1.0] GeV', fontsize=14, fontweight='bold')
    
    for i, delta_eta in enumerate(delta_etas):
        if delta_eta in pT_02_10_data:
            data = pT_02_10_data[delta_eta]
            v2_values = []
            v3_values = []
            
            for mult_range in multiplicity_ranges:
                if mult_range in data:
                    v2_2 = data[mult_range]['v2_2']
                    v3_3 = data[mult_range]['v3_3']
                    
                    # 计算v2和v3
                    v2 = calculate_vn(v2_2, 0.6, 0.6)  # 使用pT范围的中点
                    v3 = calculate_vn(v3_3, 0.6, 0.6)
                    
                    v2_values.append(v2)
                    v3_values.append(v3)
                else:
                    v2_values.append(np.nan)
                    v3_values.append(np.nan)
            
            # 绘制v2
            ax1.plot(multiplicity_centers, v2_values, 
                    color=colors[i], linestyle=linestyles[i], 
                    marker='o', linewidth=2, markersize=6,
                    label=f'v₂, Δη > {delta_eta}')
            
            # 绘制v3
            ax1.plot(multiplicity_centers, v3_values, 
                    color=colors[i], linestyle=linestyles[i], 
                    marker='s', linewidth=2, markersize=6,
                    alpha=0.7, label=f'v₃, Δη > {delta_eta}')
    
    ax1.set_xlabel('多重度区间中心 (%)', fontsize=12)
    ax1.set_ylabel('v_n', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim(5, 95)
    
    # 绘制pT 1.0-3.0 GeV的图
    ax2.set_title('各向异性流分析 - pT ∈ [1.0, 3.0] GeV', fontsize=14, fontweight='bold')
    
    for i, delta_eta in enumerate(delta_etas):
        if delta_eta in pT_10_30_data:
            data = pT_10_30_data[delta_eta]
            v2_values = []
            v3_values = []
            
            for mult_range in multiplicity_ranges:
                if mult_range in data:
                    v2_2 = data[mult_range]['v2_2']
                    v3_3 = data[mult_range]['v3_3']
                    
                    # 计算v2和v3
                    v2 = calculate_vn(v2_2, 2.0, 2.0)  # 使用pT范围的中点
                    v3 = calculate_vn(v3_3, 2.0, 2.0)
                    
                    v2_values.append(v2)
                    v3_values.append(v3)
                else:
                    v2_values.append(np.nan)
                    v3_values.append(np.nan)
            
            # 绘制v2
            ax2.plot(multiplicity_centers, v2_values, 
                    color=colors[i], linestyle=linestyles[i], 
                    marker='o', linewidth=2, markersize=6,
                    label=f'v₂, Δη > {delta_eta}')
            
            # 绘制v3
            ax2.plot(multiplicity_centers, v3_values, 
                    color=colors[i], linestyle=linestyles[i], 
                    marker='s', linewidth=2, markersize=6,
                    alpha=0.7, label=f'v₃, Δη > {delta_eta}')
    
    ax2.set_xlabel('多重度区间中心 (%)', fontsize=12)
    ax2.set_ylabel('v_n', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(5, 95)
    
    plt.tight_layout()
    
    # 保存图形
    output_file = '/Users/androktasiaileyen/ffss/anisotropic_flow_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {output_file}")
    
    # 显示图形
    plt.show()
    
    # 打印数据摘要
    print("\n数据摘要:")
    print("="*50)
    
    for pT_range_name, data_dict in [("pT 0.2-1.0 GeV", pT_02_10_data), 
                                   ("pT 1.0-3.0 GeV", pT_10_30_data)]:
        print(f"\n{pT_range_name}:")
        for delta_eta in sorted(data_dict.keys()):
            data = data_dict[delta_eta]
            print(f"  Δη > {delta_eta}:")
            for mult_range in multiplicity_ranges:
                if mult_range in data:
                    v2_2 = data[mult_range]['v2_2']
                    v3_3 = data[mult_range]['v3_3']
                    v2 = calculate_vn(v2_2, 0.6 if pT_range_name.startswith("0.2") else 2.0, 
                                   0.6 if pT_range_name.startswith("0.2") else 2.0)
                    v3 = calculate_vn(v3_3, 0.6 if pT_range_name.startswith("0.2") else 2.0, 
                                   0.6 if pT_range_name.startswith("0.2") else 2.0)
                    print(f"    {mult_range}: v₂ = {v2:.4f}, v₃ = {v3:.4f}")

if __name__ == "__main__":
    main()
