#!/usr/bin/env python3
"""
绘制dN/dpT、dN/dη和dN/dϕ分布图
使用DDDAA文件夹中所有以events开头的20个文件的数据
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_events_data(events_dir="DDDAA", pt_min=0.5, pt_max=5.0, eta_min=-1.1, eta_max=1.1):
    """
    加载所有events文件的数据
    数据格式：event_id particle_id particle_type pt phi eta
    """
    print("📂 正在加载所有events文件数据...")
    
    # 查找所有events文件
    events_files = glob.glob(os.path.join(events_dir, "events_*_converted.txt"))
    events_files.sort()
    
    if not events_files:
        print("❌ 没有找到events文件！")
        return None
    
    print(f"📊 找到 {len(events_files)} 个events文件")
    
    # 存储所有数据
    all_pt = []
    all_eta = []
    all_phi = []
    all_events = []
    
    total_particles = 0
    filtered_particles = 0
    
    for file_path in tqdm(events_files, desc="加载文件"):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 6:
                        continue
                    
                    try:
                        event_id = parts[0]
                        particle_id = int(parts[1])
                        particle_type = int(parts[2])
                        pt = float(parts[3])
                        phi = float(parts[4])
                        eta = float(parts[5])
                        
                        total_particles += 1
                        
                        # 应用筛选条件
                        if (pt_min <= pt <= pt_max and eta_min <= eta <= eta_max):
                            filtered_particles += 1
                            
                            all_pt.append(pt)
                            all_eta.append(eta)
                            all_phi.append(phi)
                            all_events.append(event_id)
                            
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"⚠️ 读取文件 {file_path} 时出错: {e}")
            continue
    
    print(f"\n✅ 数据加载完成:")
    print(f"📈 总粒子数: {total_particles:,}")
    print(f"📊 筛选后粒子数: {filtered_particles:,}")
    print(f"📊 有效事件数: {len(set(all_events)):,}")
    
    return np.array(all_pt), np.array(all_eta), np.array(all_phi), all_events

def plot_dN_dpt(pt_data, pt_min=0.5, pt_max=5.0, output_path="dN_dpt_results.png"):
    """绘制dN/dpT分布图"""
    print("📊 绘制dN/dpT分布图...")
    
    # 创建直方图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 线性坐标
    counts, bin_edges = np.histogram(pt_data, bins=50, range=(pt_min, pt_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # 计算dN/dpT (归一化到bin宽度)
    dN_dpt = counts / bin_widths
    
    # 线性图
    ax1.errorbar(bin_centers, dN_dpt, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1, 
                label=f'总粒子数: {len(pt_data):,}')
    ax1.set_xlabel(r'$p_T$ (GeV/c)', fontsize=14)
    ax1.set_ylabel(r'$dN/dp_T$ (GeV/c)$^{-1}$', fontsize=14)
    ax1.set_title(r'$dN/dp_T$ 分布 (线性坐标)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 对数坐标
    ax2.errorbar(bin_centers, dN_dpt, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1)
    ax2.set_xlabel(r'$p_T$ (GeV/c)', fontsize=14)
    ax2.set_ylabel(r'$dN/dp_T$ (GeV/c)$^{-1}$', fontsize=14)
    ax2.set_title(r'$dN/dp_T$ 分布 (对数坐标)', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ dN/dpT图保存到: {output_path}")
    
    # 保存数据到文本文件
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# pT(GeV/c) dN/dpT error\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.4f} {dN_dpt[i]:.6f} {np.sqrt(counts[i])/bin_widths[i]:.6f}\n")
    
    print(f"💾 数据保存到: {data_file}")
    
    return bin_centers, dN_dpt, np.sqrt(counts)/bin_widths

def plot_dN_deta(eta_data, eta_min=-1.1, eta_max=1.1, output_path="dN_deta_results.png"):
    """绘制dN/dη分布图"""
    print("📊 绘制dN/dη分布图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 创建直方图
    counts, bin_edges = np.histogram(eta_data, bins=44, range=(eta_min, eta_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # 计算dN/dη (归一化到bin宽度)
    dN_deta = counts / bin_widths
    
    # 线性图
    ax1.errorbar(bin_centers, dN_deta, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1,
                label=f'总粒子数: {len(eta_data):,}')
    ax1.set_xlabel(r'$\eta$', fontsize=14)
    ax1.set_ylabel(r'$dN/d\eta$', fontsize=14)
    ax1.set_title(r'$dN/d\eta$ 分布 (线性坐标)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 对数坐标
    ax2.errorbar(bin_centers, dN_deta, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1)
    ax2.set_xlabel(r'$\eta$', fontsize=14)
    ax2.set_ylabel(r'$dN/d\eta$', fontsize=14)
    ax2.set_title(r'$dN/d\eta$ 分布 (对数坐标)', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ dN/dη图保存到: {output_path}")
    
    # 保存数据到文本文件
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# eta dN/deta error\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.4f} {dN_deta[i]:.6f} {np.sqrt(counts[i])/bin_widths[i]:.6f}\n")
    
    print(f"💾 数据保存到: {data_file}")
    
    return bin_centers, dN_deta, np.sqrt(counts)/bin_widths

def plot_dN_dphi(phi_data, output_path="dN_dphi_results.png"):
    """绘制dN/dφ分布图"""
    print("📊 绘制dN/dφ分布图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 创建直方图 (φ范围: -π到π)
    counts, bin_edges = np.histogram(phi_data, bins=44, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # 计算dN/dφ (归一化到bin宽度)
    dN_dphi = counts / bin_widths
    
    # 线性图
    ax1.errorbar(bin_centers, dN_dphi, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1,
                label=f'总粒子数: {len(phi_data):,}')
    ax1.set_xlabel(r'$\phi$ (rad)', fontsize=14)
    ax1.set_ylabel(r'$dN/d\phi$ (rad)$^{-1}$', fontsize=14)
    ax1.set_title(r'$dN/d\phi$ 分布 (线性坐标)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 对数坐标
    ax2.errorbar(bin_centers, dN_dphi, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1)
    ax2.set_xlabel(r'$\phi$ (rad)', fontsize=14)
    ax2.set_ylabel(r'$dN/d\phi$ (rad)$^{-1}$', fontsize=14)
    ax2.set_title(r'$dN/d\phi$ 分布 (对数坐标)', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ dN/dφ图保存到: {output_path}")
    
    # 保存数据到文本文件
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# phi(rad) dN/dphi error\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.4f} {dN_dphi[i]:.6f} {np.sqrt(counts[i])/bin_widths[i]:.6f}\n")
    
    print(f"💾 数据保存到: {data_file}")
    
    return bin_centers, dN_dphi, np.sqrt(counts)/bin_widths

def plot_combined_distributions(pt_data, eta_data, phi_data, output_path="combined_distributions.png"):
    """绘制组合分布图"""
    print("📊 绘制组合分布图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 设置参数
    pt_min, pt_max = 0.5, 5.0
    eta_min, eta_max = -1.1, 1.1
    
    # dN/dpT
    counts_pt, bin_edges_pt = np.histogram(pt_data, bins=50, range=(pt_min, pt_max))
    bin_centers_pt = (bin_edges_pt[:-1] + bin_edges_pt[1:]) / 2
    bin_widths_pt = bin_edges_pt[1:] - bin_edges_pt[:-1]
    dN_dpt = counts_pt / bin_widths_pt
    
    axes[0, 0].errorbar(bin_centers_pt, dN_dpt, yerr=np.sqrt(counts_pt)/bin_widths_pt, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[0, 0].set_xlabel(r'$p_T$ (GeV/c)')
    axes[0, 0].set_ylabel(r'$dN/dp_T$')
    axes[0, 0].set_title(r'$dN/dp_T$ 分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].errorbar(bin_centers_pt, dN_dpt, yerr=np.sqrt(counts_pt)/bin_widths_pt, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[1, 0].set_xlabel(r'$p_T$ (GeV/c)')
    axes[1, 0].set_ylabel(r'$dN/dp_T$')
    axes[1, 0].set_title(r'$dN/dp_T$ 分布 (对数)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # dN/dη
    counts_eta, bin_edges_eta = np.histogram(eta_data, bins=44, range=(eta_min, eta_max))
    bin_centers_eta = (bin_edges_eta[:-1] + bin_edges_eta[1:]) / 2
    bin_widths_eta = bin_edges_eta[1:] - bin_edges_eta[:-1]
    dN_deta = counts_eta / bin_widths_eta
    
    axes[0, 1].errorbar(bin_centers_eta, dN_deta, yerr=np.sqrt(counts_eta)/bin_widths_eta, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[0, 1].set_xlabel(r'$\eta$')
    axes[0, 1].set_ylabel(r'$dN/d\eta$')
    axes[0, 1].set_title(r'$dN/d\eta$ 分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].errorbar(bin_centers_eta, dN_deta, yerr=np.sqrt(counts_eta)/bin_widths_eta, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[1, 1].set_xlabel(r'$\eta$')
    axes[1, 1].set_ylabel(r'$dN/d\eta$')
    axes[1, 1].set_title(r'$dN/d\eta$ 分布 (对数)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # dN/dφ
    counts_phi, bin_edges_phi = np.histogram(phi_data, bins=44, range=(-np.pi, np.pi))
    bin_centers_phi = (bin_edges_phi[:-1] + bin_edges_phi[1:]) / 2
    bin_widths_phi = bin_edges_phi[1:] - bin_edges_phi[:-1]
    dN_dphi = counts_phi / bin_widths_phi
    
    axes[0, 2].errorbar(bin_centers_phi, dN_dphi, yerr=np.sqrt(counts_phi)/bin_widths_phi, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[0, 2].set_xlabel(r'$\phi$ (rad)')
    axes[0, 2].set_ylabel(r'$dN/d\phi$')
    axes[0, 2].set_title(r'$dN/d\phi$ 分布')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].errorbar(bin_centers_phi, dN_dphi, yerr=np.sqrt(counts_phi)/bin_widths_phi, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[1, 2].set_xlabel(r'$\phi$ (rad)')
    axes[1, 2].set_ylabel(r'$dN/d\phi$')
    axes[1, 2].set_title(r'$dN/d\phi$ 分布 (对数)')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ 组合分布图保存到: {output_path}")

def main():
    """主函数"""
    print("="*80)
    print("🚀 绘制dN/dpT、dN/dη和dN/dφ分布图")
    print("="*80)
    
    # 分析参数
    pt_min = 0.5
    pt_max = 5.0
    eta_min = -1.1
    eta_max = 1.1
    
    print(f"🎯 分析参数:")
    print(f"   pT范围: [{pt_min}, {pt_max}] GeV/c")
    print(f"   η范围: [{eta_min}, {eta_max}]")
    
    # 加载数据
    data = load_events_data("DDDAA", pt_min, pt_max, eta_min, eta_max)
    if data is None:
        print("❌ 数据加载失败！")
        return
    
    pt_data, eta_data, phi_data, events = data
    
    print(f"\n📊 数据统计:")
    print(f"   pT: 均值={np.mean(pt_data):.3f}, 标准差={np.std(pt_data):.3f}")
    print(f"   η: 均值={np.mean(eta_data):.3f}, 标准差={np.std(eta_data):.3f}")
    print(f"   φ: 均值={np.mean(phi_data):.3f}, 标准差={np.std(phi_data):.3f}")
    
    # 绘制各个分布图
    print(f"\n🎨 开始绘制分布图...")
    
    # dN/dpT
    plot_dN_dpt(pt_data, pt_min, pt_max, "dN_dpt_results.png")
    
    # dN/dη
    plot_dN_deta(eta_data, eta_min, eta_max, "dN_deta_results.png")
    
    # dN/dφ
    plot_dN_dphi(phi_data, "dN_dphi_results.png")
    
    # 组合图
    plot_combined_distributions(pt_data, eta_data, phi_data, "combined_distributions.png")
    
    print(f"\n✅ 所有分布图绘制完成！")
    print(f"📊 生成的文件:")
    print(f"   - dN_dpt_results.png 和 dN_dpt_results.txt")
    print(f"   - dN_deta_results.png 和 dN_deta_results.txt")
    print(f"   - dN_dphi_results.png 和 dN_dphi_results.txt")
    print(f"   - combined_distributions.png")
    
    # 显示一些统计信息
    print(f"\n📈 统计摘要:")
    print(f"   pT分布: 最小值={np.min(pt_data):.3f}, 最大值={np.max(pt_data):.3f}")
    print(f"   η分布: 最小值={np.min(eta_data):.3f}, 最大值={np.max(eta_data):.3f}")
    print(f"   φ分布: 最小值={np.min(phi_data):.3f}, 最大值={np.max(phi_data):.3f}")

if __name__ == "__main__":
    main()
