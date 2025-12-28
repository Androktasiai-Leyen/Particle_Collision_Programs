#!/usr/bin/env python3
"""
绘制vn/多重度区间的图
从三个不同的delta eta结果文件中提取数据并生成图表
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os

def parse_results_file(file_path):
    """
    解析结果文件，提取多重度区间和对应的vn值
    """
    results = {}
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return results
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 提取delta eta值
    delta_eta = None
    for line in lines:
        if '|Δη| <' in line:
            delta_eta_match = re.search(r'\|Δη\| < ([\d.]+)', line)
            if delta_eta_match:
                delta_eta = float(delta_eta_match.group(1))
                break
    
    # 多重度区间标签
    multiplicity_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    current_range = None
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检查是否是多重度区间标签
        for range_label in multiplicity_ranges:
            if line.startswith(range_label + ':'):
                current_range = range_label
                results[current_range] = {
                    'events': 0,
                    'avg_particles': 0.0,
                    'F': None,
                    'G': None,
                    'v2_2': None,
                    'v3_3': None
                }
                break
        
        if current_range and current_range in results:
            # 提取事件数和平均粒子数
            if 'Events:' in line:
                events_match = re.search(r'Events:\s*([\d,]+)', line)
                if events_match:
                    results[current_range]['events'] = int(events_match.group(1).replace(',', ''))
            
            elif 'Average particles per event:' in line:
                avg_match = re.search(r'Average particles per event:\s*([\d.]+)', line)
                if avg_match:
                    results[current_range]['avg_particles'] = float(avg_match.group(1))
            
            # 提取拟合参数
            elif 'F =' in line:
                F_match = re.search(r'F = ([\d.-]+)', line)
                if F_match:
                    results[current_range]['F'] = float(F_match.group(1))
            
            elif 'G =' in line:
                G_match = re.search(r'G = ([\d.-]+)', line)
                if G_match:
                    results[current_range]['G'] = float(G_match.group(1))
            
            elif 'v2_2 =' in line:
                v2_match = re.search(r'v2_2 = ([\d.-]+)', line)
                if v2_match:
                    results[current_range]['v2_2'] = float(v2_match.group(1))
            
            elif 'v3_3 =' in line:
                v3_match = re.search(r'v3_3 = ([\d.-]+)', line)
                if v3_match:
                    results[current_range]['v3_3'] = float(v3_match.group(1))
    
    return results, delta_eta

def plot_vn_vs_multiplicity():
    """
    绘制vn/多重度区间的图
    """
    # 文件路径
    base_dir = "Y_analysis_results"
    files = [
        "Y_analysis_results_pT0.5-5.0_eta1.1_deltaEta1.0.txt",
        "Y_analysis_results_pT0.5-5.0_eta1.1_deltaEta1.5.txt", 
        "Y_analysis_results_pT0.5-5.0_eta1.1_deltaEta2.0.txt"
    ]
    
    # 解析所有文件
    all_results = {}
    delta_eta_values = []
    
    for file_path in files:
        full_path = os.path.join(base_dir, file_path)
        results, delta_eta = parse_results_file(full_path)
        
        if results and delta_eta:
            all_results[delta_eta] = results
            delta_eta_values.append(delta_eta)
            print(f"✅ 成功解析 |Δη| < {delta_eta} 的结果")
        else:
            print(f"❌ 解析失败: {file_path}")
    
    if not all_results:
        print("没有成功解析任何结果文件")
        return
    
    # 排序delta eta值
    delta_eta_values.sort()
    
    # 多重度区间标签
    multiplicity_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('vn vs Multiplicity Range for Different Δη Cuts\n' + 
                 'Particle selection: pT ∈ [0.5, 5.0] GeV, |η| < 1.1', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 颜色映射
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd', 'v']
    
    # 1. v2_2 vs Multiplicity Range
    ax1 = axes[0, 0]
    for i, delta_eta in enumerate(delta_eta_values):
        v2_values = []
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['v2_2'] is not None:
                v2_values.append(all_results[delta_eta][range_label]['v2_2'])
            else:
                v2_values.append(np.nan)
        
        ax1.plot(range(len(multiplicity_ranges)), v2_values, 
                marker=markers[i], linewidth=3, markersize=10, 
                label=f'|Δη| < {delta_eta}', color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Multiplicity Range', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'$v_2^2$', fontsize=14, fontweight='bold')
    ax1.set_title(r'$v_2^2$ vs Multiplicity Range', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(multiplicity_ranges)))
    ax1.set_xticklabels(multiplicity_ranges, rotation=45, fontsize=12)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.legend(fontsize=12, framealpha=0.9, loc='best')
    
    # 设置y轴范围，让变化更明显
    v2_all = []
    for delta_eta in delta_eta_values:
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['v2_2'] is not None:
                v2_all.append(all_results[delta_eta][range_label]['v2_2'])
    
    if v2_all:
        v2_min, v2_max = min(v2_all), max(v2_all)
        v2_range = v2_max - v2_min
        ax1.set_ylim(v2_min - 0.1 * v2_range, v2_max + 0.1 * v2_range)
        # 设置y轴刻度，让变化更明显
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 2. v3_3 vs Multiplicity Range
    ax2 = axes[0, 1]
    for i, delta_eta in enumerate(delta_eta_values):
        v3_values = []
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['v3_3'] is not None:
                v3_values.append(all_results[delta_eta][range_label]['v3_3'])
            else:
                v3_values.append(np.nan)
        
        ax2.plot(range(len(multiplicity_ranges)), v3_values, 
                marker=markers[i], linewidth=3, markersize=10, 
                label=f'|Δη| < {delta_eta}', color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Multiplicity Range', fontsize=14, fontweight='bold')
    ax2.set_ylabel(r'$v_3^3$', fontsize=14, fontweight='bold')
    ax2.set_title(r'$v_3^3$ vs Multiplicity Range', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(multiplicity_ranges)))
    ax2.set_xticklabels(multiplicity_ranges, rotation=45, fontsize=12)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend(fontsize=12, framealpha=0.9, loc='best')
    
    # 设置y轴范围，让变化更明显
    v3_all = []
    for delta_eta in delta_eta_values:
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['v3_3'] is not None:
                v3_all.append(all_results[delta_eta][range_label]['v3_3'])
    
    if v3_all:
        v3_min, v3_max = min(v3_all), max(v3_all)
        v3_range = v3_max - v3_min
        ax2.set_ylim(v3_min - 0.1 * v3_range, v3_max + 0.1 * v3_range)
        # 设置y轴刻度，让变化更明显
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 3. F vs Multiplicity Range
    ax3 = axes[1, 0]
    for i, delta_eta in enumerate(delta_eta_values):
        F_values = []
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['F'] is not None:
                F_values.append(all_results[delta_eta][range_label]['F'])
            else:
                F_values.append(np.nan)
        
        ax3.plot(range(len(multiplicity_ranges)), F_values, 
                marker=markers[i], linewidth=3, markersize=10, 
                label=f'|Δη| < {delta_eta}', color=colors[i], alpha=0.8)
    
    ax3.set_xlabel('Multiplicity Range', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F', fontsize=14, fontweight='bold')
    ax3.set_title('F vs Multiplicity Range', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(range(len(multiplicity_ranges)))
    ax3.set_xticklabels(multiplicity_ranges, rotation=45, fontsize=12)
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.legend(fontsize=12, framealpha=0.9, loc='best')
    
    # 设置y轴范围，让变化更明显
    F_all = []
    for delta_eta in delta_eta_values:
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['F'] is not None:
                F_all.append(all_results[delta_eta][range_label]['F'])
    
    if F_all:
        F_min, F_max = min(F_all), max(F_all)
        F_range = F_max - F_min
        ax3.set_ylim(F_min - 0.1 * F_range, F_max + 0.1 * F_range)
        # 设置y轴刻度，让变化更明显
        ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 4. G vs Multiplicity Range
    ax4 = axes[1, 1]
    for i, delta_eta in enumerate(delta_eta_values):
        G_values = []
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['G'] is not None:
                G_values.append(all_results[delta_eta][range_label]['G'])
            else:
                G_values.append(np.nan)
        
        ax4.plot(range(len(multiplicity_ranges)), G_values, 
                marker=markers[i], linewidth=3, markersize=10, 
                label=f'|Δη| < {delta_eta}', color=colors[i], alpha=0.8)
    
    ax4.set_xlabel('Multiplicity Range', fontsize=14, fontweight='bold')
    ax4.set_ylabel('G', fontsize=14, fontweight='bold')
    ax4.set_title('G vs Multiplicity Range', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(range(len(multiplicity_ranges)))
    ax4.set_xticklabels(multiplicity_ranges, rotation=45, fontsize=12)
    ax4.grid(True, alpha=0.4, linestyle='--')
    ax4.legend(fontsize=12, framealpha=0.9, loc='best')
    
    # 设置y轴范围，让变化更明显
    G_all = []
    for delta_eta in delta_eta_values:
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta] and all_results[delta_eta][range_label]['G'] is not None:
                G_all.append(all_results[delta_eta][range_label]['G'])
    
    if G_all:
        G_min, G_max = min(G_all), max(G_all)
        G_range = G_max - G_min
        ax4.set_ylim(G_min - 0.1 * G_range, G_max + 0.1 * G_range)
        # 设置y轴刻度，让变化更明显
        ax4.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(base_dir, "vn_vs_multiplicity_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印数值摘要
    print("\n📊 数值摘要:")
    print("="*80)
    for delta_eta in delta_eta_values:
        print(f"\n|Δη| < {delta_eta}:")
        print("-" * 40)
        for range_label in multiplicity_ranges:
            if range_label in all_results[delta_eta]:
                data = all_results[delta_eta][range_label]
                if data['v2_2'] is not None:
                    print(f"{range_label:8s}: v2_2 = {data['v2_2']:8.6f}, v3_3 = {data['v3_3']:8.6f}, F = {data['F']:8.6f}, G = {data['G']:8.6f}")
                else:
                    print(f"{range_label:8s}: 无拟合参数")

if __name__ == '__main__':
    plot_vn_vs_multiplicity()
