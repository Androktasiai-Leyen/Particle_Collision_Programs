#!/usr/bin/env python3
"""
增强的两粒子关联分析程序 - 包含信号和背景直方图绘制
针对 multiplicity_group_0-20_percent.txt 数据格式
pT范围：0.5-5.0 GeV，|η| ∈ [-1.1, 1.1]
支持ROOT文件输出，便于进一步分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.auto import tqdm
import os
import gc
import time
import argparse
import numba
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 启用内存监控
try:
    import psutil
    MEMORY_MONITOR = True
except ImportError:
    MEMORY_MONITOR = False

# ROOT相关导入 - 修复导入问题
ROOT_AVAILABLE = False
try:
    import ROOT
    ROOT_AVAILABLE = True
    print("✅ ROOT库可用，将生成ROOT文件")
except ImportError:
    print("⚠️ ROOT库不可用，将跳过ROOT文件生成")
    print("   请安装ROOT: https://root.cern/install/")
    print("   或者使用conda: conda install -c conda-forge root")
except Exception as e:
    print(f"⚠️ ROOT库导入出错: {e}")
    print("   将跳过ROOT文件生成")

@numba.njit
def calculate_delta_phi(phi1, phi2):
    """快速计算角度差并归一化到[-π/2, 3π/2]范围"""
    delta = phi1 - phi2
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    if delta < -np.pi/2:
        delta += 2 * np.pi
    return delta

@numba.njit(parallel=True)
def vectorized_delta_calculation(eta1, phi1, eta2, phi2):
    """向量化计算eta和phi差异"""
    n = len(eta1)
    delta_eta = np.empty(n, dtype=np.float32)
    delta_phi = np.empty(n, dtype=np.float32)
    
    for i in numba.prange(n):
        delta_eta[i] = eta1[i] - eta2[i]
        delta_phi[i] = calculate_delta_phi(phi1[i], phi2[i])
    
    return delta_eta, delta_phi

def load_data_from_txt(data_file, pt_min=0.5, pt_max=5.0, eta_min=2.0, eta_max=5.0):
    """
    从txt文件加载数据，格式：event_id particle_id particle_type pt phi eta
    """
    print(f"📂 Loading data from: {os.path.basename(data_file)}")
    print(f"🎯 pT range: [{pt_min}, {pt_max}] GeV")
    print(f"🎯 |η| range: [{eta_min}, {eta_max}]")
    
    # 存储事件数据
    event_data = {}
    total_particles = 0
    filtered_particles = 0
    
    # 获取文件大小
    file_size = os.path.getsize(data_file)
    file_size_mb = file_size / (1024**2)
    print(f"📁 File size: {file_size_mb:.2f} MB")
    
    # 读取数据
    with open(data_file, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            
            try:
                event_id = parts[0]  # 保持为字符串
                particle_id = int(parts[1])
                particle_type = int(parts[2])
                pt = float(parts[3])
                phi = float(parts[4])
                eta = float(parts[5])
                
                total_particles += 1
                
                # 应用筛选条件
                if (pt_min <= pt <= pt_max and eta_min <= abs(eta) <= eta_max):
                    filtered_particles += 1
                    
                    if event_id not in event_data:
                        event_data[event_id] = []
                    
                    event_data[event_id].append([eta, phi, pt, particle_type])
                
            except (ValueError, IndexError):
                continue
            
            # 显示进度
            if (line_num + 1) % 100000 == 0:
                print(f"📊 Processed {line_num + 1:,} lines, found {len(event_data):,} events")
    
    # 转换为numpy数组
    for event_id in event_data:
        event_data[event_id] = np.array(event_data[event_id], dtype=np.float32)
    
    print(f"\n✅ Data loading completed:")
    print(f"📈 Total particles: {total_particles:,}")
    print(f"📊 Filtered particles: {filtered_particles:,}")
    print(f"📊 Events with valid particles: {len(event_data):,}")
    
    return event_data

def load_data_from_pbpb_csv(data_file, pt_min=0.5, pt_max=5.0, eta_min=-1.1, eta_max=1.1):
    """
    从逗号分隔的 PbPb_20k.txt 加载数据，字段为：event,track,pt,eta,phi,charge
    映射关系：
      event_id=event (str)
      particle_id=track (int)
      pt=pt (float)
      eta=eta (float)
      phi=phi (float)
      particle_type ← charge (int，占位，不在后续计算中使用)
    """
    print(f"📂 Loading CSV data from: {os.path.basename(data_file)}")
    print(f"🎯 pT range: [{pt_min}, {pt_max}] GeV")
    print(f"🎯 |η| range: [{eta_min}, {eta_max}]")

    event_data: dict[str, list] = {}
    total_particles = 0
    filtered_particles = 0

    file_size = os.path.getsize(data_file)
    print(f"📁 File size: {file_size/(1024**2):.2f} MB")

    with open(data_file, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # 可能存在表头，尝试跳过非数字起始行
            if line_num == 0 and (',' in line):
                parts0 = [p.strip() for p in line.split(',')]
                # 如果第一列不是数字（可能是表头），则跳过
                try:
                    float(parts0[0])
                except Exception:
                    # 跳过表头行
                    continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue

            try:
                event_id = parts[0]  # 作为字符串键
                track_id = int(float(parts[1]))
                pt = float(parts[2])
                eta = float(parts[3])
                phi = float(parts[4])
                charge = int(float(parts[5])) if parts[5] else 0

                total_particles += 1

                # 选择条件：pt 与 |η|
                if (pt_min <= pt <= pt_max) and (eta_min <= eta <= eta_max):
                    filtered_particles += 1
                    if event_id not in event_data:
                        event_data[event_id] = []
                    # 存储顺序与下游一致：[eta, phi, pt, particle_type]
                    event_data[event_id].append([eta, phi, pt, charge])
            except Exception:
                continue

            if (line_num + 1) % 100000 == 0:
                print(f"📊 Processed {line_num + 1:,} lines, found {len(event_data):,} events")

    # 转为 numpy 数组
    for event_id in list(event_data.keys()):
        event_data[event_id] = np.array(event_data[event_id], dtype=np.float32)

    print(f"\n✅ CSV loading completed:")
    print(f"📈 Total particles: {total_particles:,}")
    print(f"📊 Filtered particles: {filtered_particles:,}")
    print(f"📊 Events with valid particles: {len(event_data):,}")

    return event_data

def analyze_multiplicity_distribution(event_data):
    """
    分析多重度分布，用于验证事件混合
    """
    print(f"\n📊 Analyzing multiplicity distribution...")
    
    multiplicities = [len(particles) for particles in event_data.values()]
    multiplicities = np.array(multiplicities)
    
    print(f"📈 Multiplicity statistics:")
    print(f"   Mean: {np.mean(multiplicities):.2f}")
    print(f"   Std: {np.std(multiplicities):.2f}")
    print(f"   Min: {np.min(multiplicities)}")
    print(f"   Max: {np.max(multiplicities)}")
    print(f"   Events with mult >= 3: {np.sum(multiplicities >= 3):,}")
    print(f"   Events with mult >= 5: {np.sum(multiplicities >= 5):,}")
    
    # 绘制多重度分布
    plt.figure(figsize=(10, 6))
    plt.hist(multiplicities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Event Multiplicity')
    plt.ylabel('Number of Events')
    plt.title('Event Multiplicity Distribution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('multiplicity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ Multiplicity distribution saved to: multiplicity_distribution.png")
    
    return multiplicities

def apply_zyam_method(C, eta_bins, phi_bins):
    """
    应用ZYAM方法：找到最小值并归零
    ZYAM = Zero Yield At Minimum
    """
    print(f"\n🔍 Applying ZYAM method...")
    
    C_zyam = C.copy()
    zyam_applied = 0
    
    # 对每个Δη切片应用ZYAM
    for i in range(eta_bins):
        phi_slice = C[i, :]
        min_val = np.min(phi_slice)
        
        if min_val < 0:
            C_zyam[i, :] -= min_val
            zyam_applied += 1
            print(f"   Δη bin {i}: min = {min_val:.4f}, applied ZYAM")
    
    # 计算整体统计
    overall_min = np.min(C)
    overall_max = np.max(C)
    zyam_min = np.min(C_zyam)
    zyam_max = np.max(C_zyam)
    
    print(f"📊 ZYAM statistics:")
    print(f"   Original range: [{overall_min:.4f}, {overall_max:.4f}]")
    print(f"   ZYAM range: [{zyam_min:.4f}, {zyam_max:.4f}]")
    print(f"   ZYAM applied to {zyam_applied}/{eta_bins} Δη bins")
    
    return C_zyam

def calculate_correlation_with_histograms(event_data, eta_bins=22, phi_bins=22, max_pairs=50000000):
    """
    计算关联函数并返回信号和背景直方图
    """
    print(f"\n📊 Calculating correlation function...")
    print(f"🔢 Grid: {eta_bins} × {phi_bins}")
    print(f"🔢 Max pairs: {max_pairs:,}")
    
    # 定义范围
    eta_range = (-2.2, 2.2)  # 修改：Δη 范围设置为[-2.2,2.2]
    phi_range = (-np.pi/2, 3*np.pi/2)
    
    # 初始化直方图
    S_N = np.zeros((eta_bins, phi_bins), dtype=np.float64)
    B_N = np.zeros((eta_bins, phi_bins), dtype=np.float64)
    
    if not event_data:
        print("⚠️ No event data found!")
        return None, None, None, None, None
    
    n_events = len(event_data)
    event_list = list(event_data.values())
    
    print(f"🎯 Total events: {n_events:,}")
    
    # 信号分布 - 向量化计算
    print("🔍 Computing signal distribution...")
    signal_pairs = 0
    np.random.seed(42)
    
    # 预计算事件权重
    event_weights = np.array([len(particles) for particles in event_list], dtype=np.float64)
    event_probs = event_weights / event_weights.sum()
    
    # 采样事件
    sampled_events = np.random.choice(n_events, size=max_pairs, p=event_probs)
    event_counts = np.bincount(sampled_events, minlength=n_events)
    
    with tqdm(total=len(event_counts), desc="Signal pairs", unit='events',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for i, count in enumerate(event_counts):
            if count == 0:
                pbar.update(1)
                continue
                
            particles = event_list[i]
            n_particles = len(particles)
            
            if n_particles < 2:
                pbar.update(1)
                continue
                
            # 计算该事件中需要采样的对数
            n_pairs = min(count, n_particles * (n_particles - 1) // 2)
            if n_pairs == 0:
                pbar.update(1)
                continue
                
            # 生成所有可能的粒子对索引
            indices = np.triu_indices(n_particles, k=1)
            if n_pairs < len(indices[0]):
                selected = np.random.choice(len(indices[0]), size=n_pairs, replace=False)
                i_indices = indices[0][selected]
                j_indices = indices[1][selected]
            else:
                i_indices = indices[0]
                j_indices = indices[1]
            
            # 提取粒子坐标
            particles_arr = particles
            eta_i = particles_arr[i_indices, 0]
            phi_i = particles_arr[i_indices, 1]
            eta_j = particles_arr[j_indices, 0]
            phi_j = particles_arr[j_indices, 1]
            
            # 计算差异
            delta_eta = eta_i - eta_j
            delta_phi = phi_i - phi_j
            
            # 向量化phi归一化
            delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
            delta_phi = np.where(delta_phi < -np.pi/2, delta_phi + 2*np.pi, delta_phi)
            
            # 向量化直方图
            valid_mask = (
                (delta_eta >= eta_range[0]) & 
                (delta_eta <= eta_range[1]) & 
                (delta_phi >= phi_range[0]) & 
                (delta_phi <= phi_range[1])
            )
            
            if np.any(valid_mask):
                hist, _, _ = np.histogram2d(
                    delta_eta[valid_mask], delta_phi[valid_mask],
                    bins=[eta_bins, phi_bins], range=[eta_range, phi_range]
                )
                S_N += hist
                signal_pairs += np.sum(valid_mask)
            
            pbar.update(1)
    
    # 背景分布 - 向量化混合事件
    print("🔍 Computing background distribution...")
    background_pairs = 0

    # 采样事件对，保证不同事件
    np.random.seed(42)
    event_indices1 = np.random.randint(0, n_events, size=max_pairs)
    event_indices2 = np.random.randint(0, n_events, size=max_pairs)
    # 保证 event_indices1 != event_indices2
    mask = event_indices1 != event_indices2
    event_indices1 = event_indices1[mask]
    event_indices2 = event_indices2[mask]
    max_pairs_actual = len(event_indices1)

    # 采样粒子
    indices1 = np.array([np.random.randint(0, len(event_list[i])) for i in event_indices1])
    indices2 = np.array([np.random.randint(0, len(event_list[j])) for j in event_indices2])

    batch_size = 1000000
    with tqdm(total=max_pairs_actual, desc="Mixed pairs", unit='pairs',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

        for start in range(0, max_pairs_actual, batch_size):
            end = min(start + batch_size, max_pairs_actual)
            batch_size_actual = end - start

            eta1 = np.array([event_list[event_indices1[k]][indices1[k], 0] for k in range(start, end)])
            phi1 = np.array([event_list[event_indices1[k]][indices1[k], 1] for k in range(start, end)])
            eta2 = np.array([event_list[event_indices2[k]][indices2[k], 0] for k in range(start, end)])
            phi2 = np.array([event_list[event_indices2[k]][indices2[k], 1] for k in range(start, end)])

            delta_eta = eta1 - eta2
            delta_phi = phi1 - phi2
            delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
            delta_phi = np.where(delta_phi < -np.pi/2, delta_phi + 2*np.pi, delta_phi)

            valid_mask = (
                (delta_eta >= eta_range[0]) & 
                (delta_eta <= eta_range[1]) & 
                (delta_phi >= phi_range[0]) & 
                (delta_phi <= phi_range[1])
            )

            if np.any(valid_mask):
                hist, _, _ = np.histogram2d(
                    delta_eta[valid_mask], delta_phi[valid_mask],
                    bins=[eta_bins, phi_bins], range=[eta_range, phi_range]
                )
                B_N += hist
                background_pairs += np.sum(valid_mask)

            pbar.update(batch_size_actual)
    
    # 计算坐标轴
    eta_centers = np.linspace(eta_range[0], eta_range[1], eta_bins, endpoint=False) + \
                  (eta_range[1] - eta_range[0]) / (2 * eta_bins)
    phi_centers = np.linspace(phi_range[0], phi_range[1], phi_bins, endpoint=False) + \
                  (phi_range[1] - phi_range[0]) / (2 * phi_bins)
    
    print(f"🔢 Signal pairs used: {signal_pairs:,}")
    print(f"🔢 Background pairs used: {background_pairs:,}")
    
    # 归一化和计算关联函数
    S_N_normalized = S_N.copy()
    B_N_normalized = B_N.copy()
    
    if signal_pairs > 0:
        S_N_normalized /= signal_pairs
    if background_pairs > 0:
        B_N_normalized /= background_pairs
    
    B_N_normalized[B_N_normalized == 0] = 1e-9  # 避免除零错误
    C = S_N_normalized / B_N_normalized
    
    # 应用ZYAM方法
    C = apply_zyam_method(C, eta_bins, phi_bins)
    
    # 清理内存
    del event_data
    gc.collect()
    
    return eta_centers, phi_centers, C, S_N, B_N

def plot_signal_histogram(eta_centers, phi_centers, S_N, pt_min, pt_max, eta_min, eta_max, output_path):
    """绘制信号直方图"""
    if eta_centers is None or phi_centers is None or S_N is None:
        print("⚠️ No signal data to plot")
        return
        
    ETA, PHI = np.meshgrid(eta_centers, phi_centers)
    Z = S_N.T
    
    # 裁剪极端值
    z_min, z_max = np.percentile(Z, [2, 98])
    Z_clipped = np.clip(Z, z_min, z_max)
    
    # 创建高质量的2D图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用更好的颜色映射和等高线
    im = ax.contourf(ETA, PHI, Z_clipped, levels=100, cmap='viridis', extend='both')
    
    # 添加等高线
    contour = ax.contour(ETA, PHI, Z_clipped, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # 设置标签和标题
    ax.set_xlabel(r'$\Delta\eta$', fontsize=16)
    ax.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=16)
    title = f'Signal Distribution (Same Event Pairs)\n(pT ∈ [{pt_min}, {pt_max}] GeV, |η| ∈ [{eta_min}, {eta_max}])'
    ax.set_title(title, fontsize=18, pad=20)
    
    # 设置坐标轴范围
    ax.set_ylim(3*np.pi/2, -np.pi/2)
    ax.set_xlim(-2.2, 2.2)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Signal Counts', fontsize=14)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🖼️ Signal histogram saved to: {output_path}")

def plot_background_histogram(eta_centers, phi_centers, B_N, pt_min, pt_max, eta_min, eta_max, output_path):
    """绘制背景直方图"""
    if eta_centers is None or phi_centers is None or B_N is None:
        print("⚠️ No background data to plot")
        return
        
    ETA, PHI = np.meshgrid(eta_centers, phi_centers)
    Z = B_N.T
    
    # 裁剪极端值
    z_min, z_max = np.percentile(Z, [2, 98])
    Z_clipped = np.clip(Z, z_min, z_max)
    
    # 创建高质量的2D图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用更好的颜色映射和等高线
    im = ax.contourf(ETA, PHI, Z_clipped, levels=100, cmap='plasma', extend='both')
    
    # 添加等高线
    contour = ax.contour(ETA, PHI, Z_clipped, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # 设置标签和标题
    ax.set_xlabel(r'$\Delta\eta$', fontsize=16)
    ax.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=16)
    title = f'Background Distribution (Mixed Event Pairs)\n(pT ∈ [{pt_min}, {pt_max}] GeV, |η| ∈ [{eta_min}, {eta_max}])'
    ax.set_title(title, fontsize=18, pad=20)
    
    # 设置坐标轴范围
    ax.set_ylim(3*np.pi/2, -np.pi/2)
    ax.set_xlim(-2.2, 2.2)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Background Counts', fontsize=14)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🖼️ Background histogram saved to: {output_path}")

def plot_3d_correlation_enhanced(eta_centers, phi_centers, C, pt_min, pt_max, eta_min, eta_max, output_path):
    """绘制增强的3D关联图"""
    if eta_centers is None or phi_centers is None or C is None:
        print("⚠️ No data to plot")
        return
        
    ETA, PHI = np.meshgrid(eta_centers, phi_centers)
    Z = C.T
    
    # 裁剪极端值 - 使用更保守的范围
    z_min, z_max = np.percentile(Z, [2, 98])
    Z_clipped = np.clip(Z, z_min, z_max)
    
    # 创建高质量的3D图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用更好的颜色映射
    surf = ax.plot_surface(
        ETA, PHI, Z_clipped,
        cmap='viridis', alpha=0.9, edgecolor='none', 
        rstride=1, cstride=1, antialiased=True,
        linewidth=0.1
    )
    
    # 设置标签和标题
    ax.set_xlabel(r'$\Delta\eta$', fontsize=16, labelpad=20)
    ax.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=16, labelpad=20)
    ax.set_zlabel(r'$C(\Delta\eta, \Delta\phi)$', fontsize=16, labelpad=20)
    
    title = f'Particle Angular Correlation Function\n(pT ∈ [{pt_min}, {pt_max}] GeV, |η| ∈ [{eta_min}, {eta_max}])'
    ax.set_title(title, fontsize=18, pad=25)
    
    # 设置坐标轴范围
    ax.set_ylim(3*np.pi/2, -np.pi/2)
    ax.set_xlim(-2.2, 2.2)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置视角
    ax.view_init(elev=35, azim=45)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, pad=0.1)
    cbar.set_label(r'$C(\Delta\eta, \Delta\phi)$', fontsize=14)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🖼️ Enhanced 3D plot saved to: {output_path}")

def plot_2d_correlation_enhanced(eta_centers, phi_centers, C, pt_min, pt_max, eta_min, eta_max, output_path):
    """绘制增强的2D关联图"""
    if eta_centers is None or phi_centers is None or C is None:
        print("⚠️ No data to plot")
        return
        
    ETA, PHI = np.meshgrid(eta_centers, phi_centers)
    Z = C.T
    
    # 裁剪极端值
    z_min, z_max = np.percentile(Z, [2, 98])
    Z_clipped = np.clip(Z, z_min, z_max)
    
    # 创建高质量的2D图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用更好的颜色映射和等高线
    im = ax.contourf(ETA, PHI, Z_clipped, levels=100, cmap='viridis', extend='both')
    
    # 添加等高线
    contour = ax.contour(ETA, PHI, Z_clipped, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # 设置标签和标题
    ax.set_xlabel(r'$\Delta\eta$', fontsize=16)
    ax.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=16)
    title = f'Particle Angular Correlation Function\n(pT ∈ [{pt_min}, {pt_max}] GeV, |η| ∈ [{eta_min}, {eta_max}])'
    ax.set_title(title, fontsize=18, pad=20)
    
    # 设置坐标轴范围
    ax.set_ylim(3*np.pi/2, -np.pi/2)
    ax.set_xlim(-2.2, 2.2)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(r'$C(\Delta\eta, \Delta\phi)$', fontsize=14)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🖼️ Enhanced 2D plot saved to: {output_path}")

def save_to_root_file(eta_centers, phi_centers, C, S_N, B_N, pt_min, pt_max, eta_min, eta_max, group_label, output_root):
    """将关联函数数据保存到ROOT文件"""
    if not ROOT_AVAILABLE:
        print("⚠️ ROOT库不可用，跳过ROOT文件生成")
        return
    
    if eta_centers is None or phi_centers is None or C is None:
        print("⚠️ 没有数据可以保存到ROOT文件")
        return
    
    try:
        print(f"💾 保存数据到ROOT文件: {output_root}")
        
        # 创建ROOT文件
        root_file = ROOT.TFile(output_root, "RECREATE")
        
        # 创建2D直方图
        eta_bins = len(eta_centers)
        phi_bins = len(phi_centers)
        
        # 信号分布直方图
        h_signal = ROOT.TH2D(
            f"h_signal_{group_label}", 
            f"Signal Distribution {group_label};#Delta#eta;#Delta#phi (rad);Counts",
            eta_bins, -2.2, 2.2,
            phi_bins, -np.pi/2, 3*np.pi/2
        )
        
        # 背景分布直方图
        h_background = ROOT.TH2D(
            f"h_background_{group_label}", 
            f"Background Distribution {group_label};#Delta#eta;#Delta#phi (rad);Counts",
            eta_bins, -2.2, 2.2,
            phi_bins, -np.pi/2, 3*np.pi/2
        )
        
        # 关联函数直方图
        h_correlation = ROOT.TH2D(
            f"h_correlation_{group_label}", 
            f"Correlation Function {group_label};#Delta#eta;#Delta#phi (rad);C(#Delta#eta,#Delta#phi)",
            eta_bins, -2.2, 2.2,
            phi_bins, -np.pi/2, 3*np.pi/2
        )
        
        # 填充直方图数据
        for i in range(eta_bins):
            for j in range(phi_bins):
                h_signal.SetBinContent(i+1, j+1, S_N[i, j])
                h_background.SetBinContent(i+1, j+1, B_N[i, j])
                h_correlation.SetBinContent(i+1, j+1, C[i, j])
        
        # 创建1D投影直方图
        h_eta_projection = ROOT.TH1D(
            f"h_eta_projection_{group_label}",
            f"#Delta#eta Projection {group_label};#Delta#eta;C(#Delta#eta)",
            eta_bins, -2.2, 2.2
        )
        
        h_phi_projection = ROOT.TH1D(
            f"h_phi_projection_{group_label}",
            f"#Delta#phi Projection {group_label};#Delta#phi (rad);C(#Delta#phi)",
            phi_bins, -np.pi/2, 3*np.pi/2
        )
        
        # 计算投影（沿phi和eta方向积分）
        for i in range(eta_bins):
            eta_sum = 0
            for j in range(phi_bins):
                eta_sum += C[i, j]
            h_eta_projection.SetBinContent(i+1, eta_sum / phi_bins)
        
        for j in range(phi_bins):
            phi_sum = 0
            for i in range(eta_bins):
                phi_sum += C[i, j]
            h_phi_projection.SetBinContent(j+1, phi_sum / eta_bins)
        
        # 创建TTree存储原始数据点
        tree = ROOT.TTree(f"correlation_data_{group_label}", f"Correlation Data {group_label}")
        
        # 定义分支变量
        delta_eta = np.array([0.0], dtype=np.float32)
        delta_phi = np.array([0.0], dtype=np.float32)
        correlation_value = np.array([0.0], dtype=np.float32)
        signal_count = np.array([0.0], dtype=np.float32)
        background_count = np.array([0.0], dtype=np.float32)
        
        # 创建分支
        tree.Branch("delta_eta", delta_eta, "delta_eta/F")
        tree.Branch("delta_phi", delta_phi, "delta_phi/F")
        tree.Branch("correlation_value", correlation_value, "correlation_value/F")
        tree.Branch("signal_count", signal_count, "signal_count/F")
        tree.Branch("background_count", background_count, "background_count/F")
        
        # 填充树
        for i in range(eta_bins):
            for j in range(phi_bins):
                delta_eta[0] = eta_centers[i]
                delta_phi[0] = phi_centers[j]
                correlation_value[0] = C[i, j]
                signal_count[0] = S_N[i, j]
                background_count[0] = B_N[i, j]
                tree.Fill()
        
        # 创建元数据直方图
        h_metadata = ROOT.TH1D(f"h_metadata_{group_label}", "Analysis Parameters", 10, 0, 10)
        h_metadata.SetBinContent(1, pt_min)
        h_metadata.SetBinContent(2, pt_max)
        h_metadata.SetBinContent(3, eta_min)
        h_metadata.SetBinContent(4, eta_max)
        h_metadata.SetBinContent(5, eta_bins)
        h_metadata.SetBinContent(6, phi_bins)
        
        # 设置元数据标签
        h_metadata.GetXaxis().SetBinLabel(1, "pt_min")
        h_metadata.GetXaxis().SetBinLabel(2, "pt_max")
        h_metadata.GetXaxis().SetBinLabel(3, "eta_min")
        h_metadata.GetXaxis().SetBinLabel(4, "eta_max")
        h_metadata.GetXaxis().SetBinLabel(5, "eta_bins")
        h_metadata.GetXaxis().SetBinLabel(6, "phi_bins")
        
        # 写入文件
        h_signal.Write()
        h_background.Write()
        h_correlation.Write()
        h_eta_projection.Write()
        h_phi_projection.Write()
        h_metadata.Write()
        tree.Write()
        
        # 关闭文件
        root_file.Close()
        
        print(f"✅ ROOT文件保存成功: {output_root}")
        print(f"📊 包含内容:")
        print(f"   - 信号分布直方图: h_signal_{group_label}")
        print(f"   - 背景分布直方图: h_background_{group_label}")
        print(f"   - 关联函数直方图: h_correlation_{group_label}")
        print(f"   - Δη投影: h_eta_projection_{group_label}")
        print(f"   - Δφ投影: h_phi_projection_{group_label}")
        print(f"   - 元数据: h_metadata_{group_label}")
        print(f"   - 原始数据树: correlation_data_{group_label}")
        
    except Exception as e:
        print(f"❌ 保存ROOT文件时出错: {e}")
        import traceback
        traceback.print_exc()

def process_all_multiplicity_groups():
    """处理所有5个多重度区间的文件"""
    print("="*80)
    print("🚀 批量处理所有多重度区间文件")
    print("="*80)
    
    # 获取用户输入的pT范围
    print("📝 请输入pT范围...")
    while True:
        try:
            pt_min = float(input("pT_min (GeV): "))
            pt_max = float(input("pT_max (GeV): "))
            if pt_min >= pt_max:
                print("❌ pT最小值必须小于pT最大值！请重新输入")
                continue
            if pt_min < 0 or pt_max < 0:
                print("❌ pT值必须为正数！请重新输入")
                continue
            break
        except ValueError:
            print("❌ 请输入有效的数字！请重新输入")
    
    print(f"✅ 用户设置的pT范围: [{pt_min}, {pt_max}] GeV")
    
    # 只处理多重度80%-100%的文件
    multiplicity_files = [
        "DDDAA/multiplicity_group_80-100_percent.txt"
    ]
    
    # 检查文件是否存在
    existing_files = []
    for file_path in multiplicity_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ 找到文件: {file_path}")
        else:
            print(f"❌ 文件不存在: {file_path}")
    
    if not existing_files:
        print("❌ 没有找到任何多重度分组文件！")
        print("请先运行 extract_high_multiplicity.py 生成多重度分组文件")
        return
    
    print(f"\n📊 将处理 {len(existing_files)} 个多重度区间文件")
    
    # 使用用户输入的pT范围
    eta_min = -1.1
    eta_max = 1.1
    max_pairs = 10000000
    
    total_start_time = time.time()
    
    for i, data_file in enumerate(existing_files):
        print(f"\n{'='*60}")
        print(f"📁 处理文件 {i+1}/{len(existing_files)}: {os.path.basename(data_file)}")
        print(f"{'='*60}")
        
        try:
            # 加载数据
            event_data = load_data_from_txt(data_file, pt_min, pt_max, eta_min, eta_max)
            if not event_data:
                print("⚠️ 没有找到有效事件，跳过此文件")
                continue
            
            # 分析多重度分布
            analyze_multiplicity_distribution(event_data)
            
            # 计算关联函数和直方图
            eta, phi, C, S_N, B_N = calculate_correlation_with_histograms(
                event_data, eta_bins=22, phi_bins=22, max_pairs=max_pairs
            )
            
            if eta is not None:
                # 从文件名提取多重度区间标签
                base = os.path.basename(data_file)
                group_label = base.split("multiplicity_group_")[1].split("_percent")[0]
                
                # 生成输出文件名 - 使用变量Pt范围
                prefix = f"multiplicity_{group_label}_pt{pt_min}-{pt_max}_eta-{eta_min}-{eta_max}_deltaEta2.2_bins22"
                output_3d = f"{prefix}_3D.png"
                output_2d = f"{prefix}_2D.png"
                output_signal = f"signal_{prefix}.png"
                output_background = f"background_{prefix}.png"
                output_root = f"{prefix}.root"
                
                # 绘制信号和背景直方图
                plot_signal_histogram(eta, phi, S_N, pt_min, pt_max, eta_min, eta_max, output_signal)
                plot_background_histogram(eta, phi, B_N, pt_min, pt_max, eta_min, eta_max, output_background)
                
                # 绘制关联函数
                plot_3d_correlation_enhanced(eta, phi, C, pt_min, pt_max, eta_min, eta_max, output_3d)
                plot_2d_correlation_enhanced(eta, phi, C, pt_min, pt_max, eta_min, eta_max, output_2d)
                
                # 保存ROOT文件
                save_to_root_file(eta, phi, C, S_N, B_N, pt_min, pt_max, eta_min, eta_max, group_label, output_root)
                
                print(f"✅ 文件 {group_label} 处理完成")
                print(f"📊 生成图片: {output_3d}, {output_2d}, {output_signal}, {output_background}")
                print(f"💾 生成ROOT文件: {output_root}")
            
        except Exception as e:
            print(f"❌ 处理文件 {data_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("✅ 所有多重度区间文件处理完成！")
    print(f"{'='*60}")
    print(f"⏱️ 总处理时间: {total_elapsed/60:.1f} 分钟")
    print(f"📊 处理了 {len(existing_files)} 个文件")
    if ROOT_AVAILABLE:
        print(f"💾 生成了ROOT文件，可在ROOT中进一步分析")

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="Enhanced particle correlation analysis")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["DDDAA/multiplicity_group_0-20_percent.txt"],
        help="One or more data files to process"
    )
    parser.add_argument(
        "--format",
        choices=["auto", "txt", "csv"],
        default="auto",
        help="Input file format: auto-detect, txt (space-separated), or csv (comma-separated)"
    )
    parser.add_argument(
        "--all-multiplicity",
        action="store_true",
        help="Process all multiplicity group files automatically"
    )
    args = parser.parse_args()

    # 如果指定了--all-multiplicity，则批量处理所有多重度文件
    if args.all_multiplicity:
        process_all_multiplicity_groups()
        return

    # 获取用户输入的pT范围
    print("📝 请输入pT范围...")
    while True:
        try:
            pt_min = float(input("pT_min (GeV): "))
            pt_max = float(input("pT_max (GeV): "))
            if pt_min >= pt_max:
                print("❌ pT最小值必须小于pT最大值！请重新输入")
                continue
            if pt_min < 0 or pt_max < 0:
                print("❌ pT值必须为正数！请重新输入")
                continue
            break
        except ValueError:
            print("❌ 请输入有效的数字！请重新输入")
    
    print(f"✅ 用户设置的pT范围: [{pt_min}, {pt_max}] GeV")
    
    # 使用用户输入的pT范围
    eta_min = -1.1
    eta_max = 1.1
    max_pairs = 10000000

    if MEMORY_MONITOR:
        print("📊 Memory monitoring enabled")

    for data_file in args.data:
        if not os.path.exists(data_file):
            print(f"❌ Error: File not found - {data_file}")
            continue

        print("="*80)
        print("🚀 ENHANCED PARTICLE CORRELATION ANALYSIS WITH HISTOGRAMS")
        print("="*80)
        print(f"📁 File: {data_file}")
        print(f"🎯 pT range: [{pt_min}, {pt_max}] GeV")
        print(f"🎯 |η| range: [{eta_min}, {eta_max}]")
        print(f"🔢 Max pairs: {max_pairs:,}")

        total_start_time = time.time()

        try:
            # 第一阶段：加载数据（格式自适应/指定）
            input_format = args.format
            if input_format == "auto":
                # 简单自动判断：若文件名以 .csv 结尾或文件前几行包含逗号，则认为 csv
                is_csv = data_file.lower().endswith('.csv')
                if not is_csv:
                    with open(data_file, 'r') as _f:
                        for _ in range(10):
                            l = _f.readline()
                            if not l:
                                break
                            if ',' in l:
                                is_csv = True
                                break
                input_format = 'csv' if is_csv else 'txt'

            if input_format == 'csv':
                event_data = load_data_from_pbpb_csv(data_file, pt_min, pt_max, eta_min, eta_max)
            else:
                event_data = load_data_from_txt(data_file, pt_min, pt_max, eta_min, eta_max)
            if not event_data:
                print("⚠️ No valid events found!")
                continue

            # 分析多重度分布
            analyze_multiplicity_distribution(event_data)

            # 第二阶段：计算关联函数和直方图
            eta, phi, C, S_N, B_N = calculate_correlation_with_histograms(
                event_data, eta_bins=22, phi_bins=22, max_pairs=max_pairs
            )

            if eta is not None:
                # 推断输出前缀中的多重度区间
                base = os.path.basename(data_file)
                group_label = ""
                if "multiplicity_group_" in base and "_percent" in base:
                    group_label = base.split("multiplicity_group_")[1].split("_percent")[0]
                else:
                    # 针对 PbPb_20k.txt 自定义前缀
                    group_label = "custom"
                    if "pbpb" in base.lower():
                        group_label = "PbPb20k"

                # 更新输出文件名前缀 - 使用变量Pt范围
                prefix = f"multiplicity_{group_label}_pt{pt_min}-{pt_max}_eta-{eta_min}-{eta_max}_deltaEta2.2_bins22"
                output_3d = f"{prefix}_3D.png"
                output_2d = f"{prefix}_2D.png"
                output_signal = f"signal_{prefix}.png"
                output_background = f"background_{prefix}.png"
                output_root = f"{prefix}.root"

                # 绘制信号和背景直方图
                plot_signal_histogram(eta, phi, S_N, pt_min, pt_max, eta_min, eta_max, output_signal)
                plot_background_histogram(eta, phi, B_N, pt_min, pt_max, eta_min, eta_max, output_background)

                # 绘制关联函数
                plot_3d_correlation_enhanced(eta, phi, C, pt_min, pt_max, eta_min, eta_max, output_3d)
                plot_2d_correlation_enhanced(eta, phi, C, pt_min, pt_max, eta_min, eta_max, output_2d)
                
                # 保存ROOT文件
                save_to_root_file(eta, phi, C, S_N, B_N, pt_min, pt_max, eta_min, eta_max, group_label, output_root)

            total_elapsed = time.time() - total_start_time
            print("\n" + "="*60)
            print("✅ ENHANCED ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"⏱️ Total processing time: {total_elapsed/60:.1f} minutes")
            print(f"🎯 Processed {len(event_data):,} events")
            print(f"📊 Generated signal and background histograms")
            if ROOT_AVAILABLE:
                print(f"💾 Generated ROOT file for further analysis")

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()