#!/usr/bin/env python3
"""
更新版本的Per-trigger-particle yields分析脚本
1. 粒子筛选条件：pT ∈ [0.5, 5.0] GeV，|η| < 1.1
2. 数据格式：event_id particle_id particle_type pt phi eta
3. 使用DDDAA文件夹下的已分类数据
4. 支持多种Δη cut: |Δη| < 1.1, < 1.5, < 2.0
5. 支持保存和加载计算结果，避免重复计算

使用方法：
1. 首次运行：python Y_analysis_updated.py
   - 会分析3种不同的delta eta cut
   - 计算结果会自动保存为.pkl文件
   
2. 查看已保存的结果：
   - 在Python中运行: list_saved_results()
   
3. 从保存的数据生成图表（无需重新计算）：
   - 在Python中运行: plot_from_saved_data('path/to/calculation_results.pkl')
   
4. 重新计算（如果数据文件损坏或需要更新）：
   - 删除对应的.pkl文件，或
   - 在运行时选择'n'来重新计算
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.auto import tqdm
import os
from collections import defaultdict
import re
import psutil
import time
import pickle
from scipy.optimize import minimize

# Enable memory monitoring
try:
    import psutil
    MEMORY_MONITOR = True
except ImportError:
    MEMORY_MONITOR = False

def sanitize_filename(filename):
    """Replace illegal characters in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def save_calculation_results(results_dict, Y_periph_data, general_params, delta_eta_cut, output_dir):
    """
    保存计算结果到pickle文件，包括所有分析结果和参数
    """
    # 创建数据文件名
    data_filename = f"calculation_results_pT{general_params['pt_min']}-{general_params['pt_max']}_eta{general_params['eta_max']}_deltaEta{delta_eta_cut}.pkl"
    data_path = os.path.join(output_dir, data_filename)
    
    # 准备保存的数据
    save_data = {
        'results_dict': results_dict,
        'Y_periph_data': Y_periph_data,
        'general_params': general_params,
        'delta_eta_cut': delta_eta_cut,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0'
    }
    
    # 保存到pickle文件
    with open(data_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Calculation results saved to {data_path}")
    return data_path

def load_calculation_results(data_path):
    """
    从pickle文件加载计算结果
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None, None, None, None
    
    try:
        with open(data_path, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"✅ Calculation results loaded from {data_path}")
        print(f"   Timestamp: {save_data['timestamp']}")
        print(f"   Version: {save_data['version']}")
        print(f"   Delta eta cut: {save_data['delta_eta_cut']}")
        
        return (save_data['results_dict'], 
                save_data['Y_periph_data'], 
                save_data['general_params'], 
                save_data['delta_eta_cut'])
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def check_existing_results(output_dir, general_params, delta_eta_cut):
    """
    检查是否已经存在计算结果文件
    """
    data_filename = f"calculation_results_pT{general_params['pt_min']}-{general_params['pt_max']}_eta{general_params['eta_max']}_deltaEta{delta_eta_cut}.pkl"
    data_path = os.path.join(output_dir, data_filename)
    
    if os.path.exists(data_path):
        # 检查文件是否完整
        try:
            with open(data_path, 'rb') as f:
                save_data = pickle.load(f)
            return True, data_path
        except:
            return False, data_path
    return False, data_path

def load_events_from_classified_data(data_path, pt_min=0.5, pt_max=5.0, eta_max=1.1, chunk_size=2000000):
    """
    从已分类的数据文件中加载事件，应用粒子筛选条件
    数据格式: event_id particle_id particle_type pt phi eta
    """
    print(f"Loading data: {data_path}")
    print(f"Particle selection: pT ∈ [{pt_min}, {pt_max}] GeV, |η| < {eta_max}")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None, None, None, None

    file_size = os.path.getsize(data_path)
    file_size_gb = file_size / (1024**3)
    print(f"📁 File size: {file_size_gb:.2f} GB")

    # 存储筛选后的粒子数据
    event_particles = defaultdict(list)
    total_particles = 0
    filtered_particles = 0
    
    print("\n🔍 Processing data with particle filtering...")
    start_time = time.time()
    
    chunk_iter = pd.read_csv(
        data_path, 
        sep=r'\s+',
        chunksize=chunk_size,
        header=None,
        names=['event_id', 'particle_id', 'particle_type', 'pt', 'phi', 'eta'],
        dtype={
            'event_id': 'int32',
            'particle_id': 'int32', 
            'particle_type': 'int8',
            'pt': 'float32',
            'phi': 'float32',
            'eta': 'float32'
        },
        engine='c',
        low_memory=False
    )
    
    total_lines = 0
    chunk_count = 0
    
    with tqdm(desc="Processing chunks", unit='chunk') as pbar:
        for chunk in chunk_iter:
            chunk_count += 1
            total_lines += len(chunk)
            
            # 应用粒子筛选条件
            mask = (
                (chunk['pt'] >= pt_min) & 
                (chunk['pt'] <= pt_max) & 
                (np.abs(chunk['eta']) < eta_max)
            )
            filtered_chunk = chunk.loc[mask]
            
            total_particles += len(chunk)
            filtered_particles += len(filtered_chunk)
            
            if not filtered_chunk.empty:
                # 按事件分组处理筛选后的粒子
                for event_id, group in filtered_chunk.groupby('event_id'):
                    eta_phi_data = group[['eta', 'phi']].values
                    event_particles[event_id].extend(list(map(tuple, eta_phi_data)))
            
            # 更新进度
            elapsed = time.time() - start_time
            lines_per_sec = total_lines / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'Events': len(event_particles),
                'Chunks': chunk_count,
                'Lines': f'{total_lines:,}',
                'Speed': f'{lines_per_sec/1e6:.1f} M lines/s',
                'Mem': f'{psutil.virtual_memory().percent}%' if MEMORY_MONITOR else 'N/A'
            })
            pbar.update(1)

    elapsed_total = time.time() - start_time
    print(f"\n✅ Data processing completed in {elapsed_total:.1f}s")
    print(f"📝 Total particles: {total_particles:,}")
    print(f"📝 Filtered particles: {filtered_particles:,}")
    print(f"📝 Filtering efficiency: {filtered_particles/total_particles*100:.1f}%")
    print(f"📝 Events with filtered particles: {len(event_particles):,}")
    
    # 筛选至少有2个粒子的事件
    final_valid_events = {
        eid: particles for eid, particles in event_particles.items()
        if len(particles) >= 2
    }
    
    print(f"📝 Final valid events (≥2 particles): {len(final_valid_events):,}")
    
    if final_valid_events:
        avg_particles = np.mean([len(p) for p in final_valid_events.values()])
        print(f"📝 Average particles per event: {avg_particles:.2f}")
    
    return final_valid_events, pt_min, pt_max, eta_max

def calculate_correlation_with_yields_atlas_cut(event_group, eta_bins=30, phi_bins=36, delta_eta_cut=1.0):
    """
    Calculate the correlation function C(Δη, Δφ) and the associated yield Y(Δφ),
    using a configurable Δη cut for the yield calculation.
    
    Parameters:
    - event_group: Dictionary of events with particle data
    - eta_bins: Number of η bins
    - phi_bins: Number of φ bins  
    - delta_eta_cut: Δη cut value (|Δη| < delta_eta_cut)
    """
    eta_range = (-5, 5)
    phi_range = (-np.pi / 2, 3 * np.pi / 2)
    
    S_N = np.zeros((eta_bins, phi_bins), dtype=np.float64)
    B_N = np.zeros((eta_bins, phi_bins), dtype=np.float64)

    event_ids = sorted(event_group.keys())
    event_list = [event_group[eid] for eid in event_ids]
    n_events = len(event_list)
    
    if n_events == 0:
        print("Warning: No events in this group to process.")
        return None, None, None, None

    # Calculate signal distribution S_N (same-event pairs)
    print(f"\nCalculating signal distribution S_N with |Δη| < {delta_eta_cut}...")
    for particles in tqdm(event_list, desc="Processing signal events", leave=False):
        n = len(particles)
        if n < 2:
            continue

        eta = np.array([p[0] for p in particles])
        phi = np.array([p[1] for p in particles])

        # Create all pairs
        idx = np.arange(n)
        i, j = np.meshgrid(idx, idx)
        mask = i != j
        i, j = i[mask], j[mask]
        
        delta_eta = eta[i] - eta[j]
        delta_phi = phi[i] - phi[j]
        
        # Adjust delta_phi to be in [-pi/2, 3pi/2)
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
        delta_phi[delta_phi < -np.pi / 2] += 2 * np.pi

        hist, _, _ = np.histogram2d(
            delta_eta, delta_phi,
            bins=[eta_bins, phi_bins], range=[eta_range, phi_range]
        )
        S_N += hist / (n * (n - 1))

    S_N /= n_events

    # Calculate background distribution B_N (mixed-event pairs)
    print("Calculating background distribution B_N...")
    max_mixed_pairs = 2000000 
    n_mixed_attempts = min(max_mixed_pairs, n_events * (n_events - 1))

    np.random.seed(42)
    idx1 = np.random.randint(0, n_events, size=n_mixed_attempts)
    idx2 = np.random.randint(0, n_events, size=n_mixed_attempts)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    n_mixed_eff = len(idx1)
    
    if n_mixed_eff == 0 and n_events > 1:
        print("Warning: Failed to create mixed event pairs. Background will be zero.")
    elif n_mixed_eff > 0:
        for i in tqdm(range(n_mixed_eff), desc="Processing mixed pairs", leave=False):
            particles1 = event_list[idx1[i]]
            particles2 = event_list[idx2[i]]
            n1, n2 = len(particles1), len(particles2)
            if n1 == 0 or n2 == 0: continue

            eta1, phi1 = np.array([p[0] for p in particles1]).reshape(-1, 1), np.array([p[1] for p in particles1]).reshape(-1, 1)
            eta2, phi2 = np.array([p[0] for p in particles2]), np.array([p[1] for p in particles2])

            delta_eta = eta1 - eta2
            delta_phi = phi1 - phi2
            
            delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
            delta_phi[delta_phi < -np.pi / 2] += 2 * np.pi
            
            hist, _, _ = np.histogram2d(
                delta_eta.ravel(), delta_phi.ravel(),
                bins=[eta_bins, phi_bins], range=[eta_range, phi_range]
            )
            B_N += hist / (n1 * n2)

        B_N /= n_mixed_eff

    # Calculate C(Δη, Δφ)
    C = np.divide(S_N, B_N, out=np.zeros_like(S_N), where=B_N != 0)

    eta_bin_width = (eta_range[1] - eta_range[0]) / eta_bins
    eta_centers = np.linspace(eta_range[0], eta_range[1], eta_bins, endpoint=False) + eta_bin_width / 2
    
    phi_bin_width = (phi_range[1] - phi_range[0]) / phi_bins
    phi_centers = np.linspace(phi_range[0], phi_range[1], phi_bins, endpoint=False) + phi_bin_width / 2

    # Calculate Y(Δφ) using configurable Δη cut |Δη| < delta_eta_cut
    print(f"Calculating Y(Δφ) with |Δη| < {delta_eta_cut} cut...")
    
    # Create |Δη| mask
    eta_mask = np.abs(eta_centers) < delta_eta_cut
    
    B_N_filtered = B_N[eta_mask, :]
    C_filtered = C[eta_mask, :]
    
    # Integrate over Δη in the selected region
    # The integral of B_N(Δη, Δφ) over Δη gives b(Δφ)
    b_delta_phi = np.sum(B_N_filtered, axis=0) * eta_bin_width
    
    # Calculate α, the normalization factor
    alpha = np.sum(b_delta_phi) * phi_bin_width
    if alpha == 0:
        print("Warning: Background normalization 'alpha' is zero. Y(Δφ) will be zero.")
        Y = np.zeros_like(phi_centers)
    else:
        # Normalize b(Δφ) to get the pair acceptance a(Δφ)
        a_delta_phi = b_delta_phi / alpha
        
        # The yield is defined as S_N(Δφ) / a(Δφ)
        S_N_phi = np.sum(S_N, axis=0) * eta_bin_width
        Y = np.divide(S_N_phi, a_delta_phi, out=np.zeros_like(S_N_phi), where=a_delta_phi != 0)
        
    return phi_centers, Y, C, eta_centers

def fit_template_model_with_v3(phi_centers, Y_data, phi_centers_periph, Y_periph):
    """
    Fits the template model: Y_templ(dphi) = F * Y_periph(dphi) + Y_ridge(dphi)
    where Y_ridge(dphi) = G * (1 + 2 * v2_2 * cos(2 * dphi) + 2 * v3_3 * cos(3 * dphi))
    with the constraint: integral(Y_templ, 0, pi) = integral(Y_data, 0, pi)
    
    Modified version with v3_3 component.
    """
    # First, interpolate Y_periph to match phi_centers
    if not np.array_equal(phi_centers, phi_centers_periph):
        print("Interpolating peripheral template to match current phi grid...")
        Y_periph_interp = np.interp(phi_centers, phi_centers_periph, Y_periph)
    else:
        Y_periph_interp = Y_periph.copy()
    
    def template_model(params, phi):
        F, G, v2_2, v3_3 = params
        Y_ridge = G * (1 + 2 * v2_2 * np.cos(2 * phi) + 2 * v3_3 * np.cos(3 * phi))
        Y_templ = F * Y_periph_interp + Y_ridge
        return Y_templ

    # Prepare data for integration (0 to pi)
    integral_mask = (phi_centers >= 0) & (phi_centers <= np.pi)
    phi_integral_range = phi_centers[integral_mask]
    Y_data_integral_range = Y_data[integral_mask]
    Y_periph_integral_range = Y_periph_interp[integral_mask]

    # Calculate target integral
    target_integral = np.trapz(Y_data_integral_range, phi_integral_range)
    print(f"Target integral (0 to π): {target_integral:.6f}")

    # Objective function to minimize
    def objective(params):
        F, G, v2_2, v3_3 = params
        
        # 1. Chi-squared term (goodness of fit over full range)
        Y_model = template_model(params, phi_centers)
        
        # Weight the fit more heavily around the peak region
        weights = np.ones_like(phi_centers)
        peak_mask = (phi_centers >= -0.5) & (phi_centers <= 0.5)
        weights[peak_mask] *= 2.0  # Give more weight to the peak region
        
        residual = np.sum(weights * (Y_model - Y_data)**2)

        # 2. Integral constraint penalty
        Y_ridge_integral = G * (1 + 2 * v2_2 * np.cos(2 * phi_integral_range) + 
                               2 * v3_3 * np.cos(3 * phi_integral_range))
        Y_templ_integral_values = F * Y_periph_integral_range + Y_ridge_integral
        model_integral = np.trapz(Y_templ_integral_values, phi_integral_range)
        
        # Heavy penalty for violating the integral constraint
        integral_penalty = 10000 * (model_integral - target_integral)**2
        
        # 3. Additional constraint: Y_templ should approximately match Y_data at φ=0
        phi_zero_idx = np.argmin(np.abs(phi_centers))
        Y_model_at_zero = F * Y_periph_interp[phi_zero_idx] + G * (1 + 2 * v2_2 + 2 * v3_3)
        zero_penalty = 100 * (Y_model_at_zero - Y_data[phi_zero_idx])**2
        
        return residual + integral_penalty + zero_penalty

    # Better initial guess based on data characteristics
    print("Estimating initial parameters from data...")
    
    # Basic statistics
    baseline = np.mean(Y_data)
    max_Y_data = np.max(Y_data)
    max_Y_periph = np.max(Y_periph_interp)
    mean_Y_periph = np.mean(Y_periph_interp)
    
    # Estimate flow coefficients using simple Fourier decomposition
    cos2 = np.cos(2 * phi_centers)
    cos3 = np.cos(3 * phi_centers)
    
    # Simple least squares estimation for flow components
    Y_fluctuation = Y_data - baseline
    A2 = np.sum(Y_fluctuation * cos2) / np.sum(cos2**2) if np.sum(cos2**2) > 0 else 0
    A3 = np.sum(Y_fluctuation * cos3) / np.sum(cos3**2) if np.sum(cos3**2) > 0 else 0
    
    # Convert to v2_2 and v3_3 estimates (A = 2*G*v*v for v_n^n)
    v2_2_guess = A2 / (2 * baseline) if baseline > 0 else 0.01
    v3_3_guess = A3 / (2 * baseline) if baseline > 0 else 0.005
    
    # Estimate F from peak ratio
    F_guess = max_Y_data / max_Y_periph if max_Y_periph > 0 else 0.3
    
    # Estimate G from baseline difference
    G_guess = baseline - F_guess * mean_Y_periph
    if G_guess < 0:  # Ensure G is positive
        G_guess = baseline * 0.1
    
    # Sanity checks and adjustments
    F_guess = np.clip(F_guess, 0.1, 2.0)  # Reasonable range for F
    G_guess = np.clip(G_guess, baseline * 0.05, baseline * 0.5)  # G should be reasonable fraction of baseline
    v2_2_guess = np.clip(v2_2_guess, -0.3, 0.3)  # Typical v2^2 range
    v3_3_guess = np.clip(v3_3_guess, -0.2, 0.2)  # Typical v3^3 range
    
    initial_guess = [F_guess, G_guess, v2_2_guess, v3_3_guess]
    print(f"Data-driven initial guess:")
    print(f"  F={F_guess:.4f} (peak ratio: {max_Y_data:.4f}/{max_Y_periph:.4f})")
    print(f"  G={G_guess:.4f} (baseline: {baseline:.4f}, mean_periph: {mean_Y_periph:.4f})")
    print(f"  v2_2={v2_2_guess:.4f} (A2={A2:.4f})")
    print(f"  v3_3={v3_3_guess:.4f} (A3={A3:.4f})")
    
    # Tighter parameter bounds based on physical expectations
    bounds = [(0.05, 3.0),   # F: reasonable scaling factor
              (0.0, baseline),  # G: cannot exceed baseline
              (-0.5, 0.5),    # v2_2: typical range for v2^2
              (-0.3, 0.3)]    # v3_3: typical range for v3^3

    # Perform the optimization with multiple attempts
    best_result = None
    best_objective = float('inf')
    
    for attempt in range(3):
        if attempt > 0:
            # Add smarter noise to initial guess for subsequent attempts
            # Use relative perturbation based on parameter magnitude
            noise_factors = [0.15, 0.15, 0.20, 0.20]  # Different noise levels for different parameters
            noisy_guess = [
                initial_guess[0] * (1 + noise_factors[0] * np.random.randn()),
                initial_guess[1] * (1 + noise_factors[1] * np.random.randn()), 
                initial_guess[2] + noise_factors[2] * 0.05 * np.random.randn(),  # Additive for small flow coefficients
                initial_guess[3] + noise_factors[3] * 0.03 * np.random.randn()   # Additive for small flow coefficients
            ]
            # Ensure bounds are respected
            for i in range(4):
                noisy_guess[i] = np.clip(noisy_guess[i], bounds[i][0], bounds[i][1])
            print(f"  Attempt {attempt+1}: F={noisy_guess[0]:.4f}, G={noisy_guess[1]:.4f}, v2_2={noisy_guess[2]:.4f}, v3_3={noisy_guess[3]:.4f}")
        else:
            noisy_guess = initial_guess
            print(f"  Attempt {attempt+1}: Using data-driven initial guess")
            
        result = minimize(objective, noisy_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success and result.fun < best_objective:
            best_result = result
            best_objective = result.fun
    
    if best_result and best_result.success:
        F_opt, G_opt, v2_2_opt, v3_3_opt = best_result.x
        print(f"Fit successful: F={F_opt:.4f}, G={G_opt:.4f}, v2_2={v2_2_opt:.4f}, v3_3={v3_3_opt:.4f}")

        # Verify constraints
        Y_ridge_opt_integral = G_opt * (1 + 2 * v2_2_opt * np.cos(2 * phi_integral_range) + 2 * v3_3_opt * np.cos(3 * phi_integral_range))
        Y_templ_opt_integral = F_opt * Y_periph_integral_range + Y_ridge_opt_integral
        integral_check = np.trapz(Y_templ_opt_integral, phi_integral_range)
        print(f"  Integral Check: Target={target_integral:.6f}, Fit={integral_check:.6f}, Diff={abs(integral_check - target_integral):.6f}")
        
        # Check fit at φ=0
        phi_zero_idx = np.argmin(np.abs(phi_centers))
        Y_model_at_zero = F_opt * Y_periph_interp[phi_zero_idx] + G_opt * (1 + 2 * v2_2_opt + 2 * v3_3_opt)
        print(f"  φ=0 Check: Data={Y_data[phi_zero_idx]:.6f}, Model={Y_model_at_zero:.6f}, Diff={abs(Y_model_at_zero - Y_data[phi_zero_idx]):.6f}")

        return F_opt, G_opt, v2_2_opt, v3_3_opt, Y_periph_interp
    else:
        print(f"Fit failed: {best_result.message if best_result else 'All attempts failed'}")
        return None, None, None, None, None

def plot_template_analysis_comparison(results_dict, Y_periph_data, general_params, output_path, delta_eta_cut):
    """
    Fixed version of the plotting function with proper interpolation handling.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 20), tight_layout=True)
    axes = axes.flatten()
    
    plot_configs = [
        ("0-20%", 0),
        ("20-40%", 1), 
        ("40-60%", 2),
        ("60-80%", 3),
        ("80-100%", 4),
        ("Summary", 5)
    ]
    
    phi_centers_periph, Y_periph = Y_periph_data
    
    # 首先计算所有Y数据的范围，用于统一Y轴
    all_y_values = []
    for range_label in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
        if range_label in results_dict and results_dict[range_label] is not None:
            all_y_values.extend(results_dict[range_label]['Y_data'])
    
    if all_y_values:
        y_min = min(all_y_values) * 0.95  # 留5%的边距
        y_max = max(all_y_values) * 1.05  # 留5%的边距
    else:
        y_min, y_max = 0, 1
    
    for i, (range_label, ax_idx) in enumerate(plot_configs):
        ax = axes[ax_idx]
        
        if range_label == "Summary":
            # Create summary plot showing all Y(Δφ) curves
            ax.set_title("Summary: Y(Δφ) for all multiplicity ranges", fontsize=12, pad=10)
            
            colors = ['blue', 'green', 'orange', 'red', 'purple']
            for j, (label, color) in enumerate(zip(["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], colors)):
                if label in results_dict and results_dict[label] is not None:
                    plot_data = results_dict[label]
                    phi_centers = plot_data['phi_centers']
                    Y_data = plot_data['Y_data']
                    ax.plot(phi_centers, Y_data, color=color, marker='o', markersize=3, 
                           label=f'{label} ({plot_data["n_events"]:,} events)', linewidth=1)
            
            # 设置统一的Y轴范围
            ax.set_ylim(y_min, y_max)
            
            # 图例移到右上角，不显示在图中
            ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
            
        elif range_label not in results_dict or results_dict[range_label] is None:
            ax.text(0.5, 0.5, 'No data for this range', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f'{range_label}\n(No events)', fontsize=12, pad=10)
            ax.set_ylim(y_min, y_max)
        else:
            plot_data = results_dict[range_label]

            if range_label == "0-20%":
                # Task 1: Plot peripheral template Y_periph(dphi)
                ax.plot(phi_centers_periph, Y_periph, 'bo', markersize=5, label=r'$Y_{periph}(\Delta\phi)$')
                title = (f'{range_label} (Peripheral Template)\n'
                         f'{plot_data["n_events"]:,} events, '
                         f'⟨N_rec⟩ = {plot_data["avg_n"]:.1f}')
                ax.set_title(title, fontsize=12, pad=10)
                ax.set_ylim(y_min, y_max)

            else:
                # Task 2: Plot data and template fit components
                phi_centers = plot_data['phi_centers']
                Y_data = plot_data['Y_data']
                F, G, v2_2, v3_3 = plot_data['F'], plot_data['G'], plot_data['v2_2'], plot_data['v3_3']
                Y_periph_interp = plot_data.get('Y_periph_interp')

                if F is None: # Handle fit failure
                    ax.plot(phi_centers, Y_data, 'ko', markersize=4, label=r'$Y(\Delta\phi)$ data')
                    ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, ha='center', va='center', fontsize=14, color='orange')
                    ax.set_title(f'{range_label}\n(Fit Failed)', fontsize=12, pad=10)
                    ax.set_ylim(y_min, y_max)
                else:
                    # Use interpolated Y_periph if available, otherwise interpolate here
                    if Y_periph_interp is None:
                        if not np.array_equal(phi_centers, phi_centers_periph):
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(phi_centers_periph, Y_periph, kind='linear', 
                                                  bounds_error=False, fill_value='extrapolate')
                            Y_periph_interp = interp_func(phi_centers)
                        else:
                            Y_periph_interp = Y_periph
                    
                    # Calculate model components for plotting
                    Y_ridge = G * (1 + 2 * v2_2 * np.cos(2 * phi_centers) + 2 * v3_3 * np.cos(3 * phi_centers))
                    Y_templ = F * Y_periph_interp + Y_ridge
                    F_Y_periph_plus_G = F * Y_periph_interp + G
                    
                    # For the baseline, use Y_periph value at φ=0
                    phi_zero_idx = np.argmin(np.abs(phi_centers))  
                    Y_periph_at_zero = Y_periph_interp[phi_zero_idx]
                    Y_ridge_plus_F_Y_periph_0 = Y_ridge + F * Y_periph_at_zero

                    # 1. Black solid circles: Y(dphi) data
                    ax.plot(phi_centers, Y_data, 'ko', markersize=4, label=r'$Y(\Delta\phi)$ data')
                    
                    # 2. Red step line: Y_templ(dphi) - this should match the data
                    ax.step(phi_centers, Y_templ, 'r-', linewidth=2, where='mid', label=r'$Y_{templ}(\Delta\phi)$')
                    
                    # 3. Hollow circles: F*Y_periph(dphi)+G
                    ax.plot(phi_centers, F_Y_periph_plus_G, 'o', markersize=5, 
                            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
                            label=rf'$F \cdot Y_{{periph}} + G$')
                    
                    # 4. Blue dashed line: Y_ridge(dphi) + F*Y_periph(0)
                    ax.plot(phi_centers, Y_ridge_plus_F_Y_periph_0, 'b--', linewidth=2, 
                            label=rf'$Y_{{ridge}} + F \cdot Y_{{periph}}(0)$')
                    
                    title = (f'{range_label} Template Fit\n'
                             f'{plot_data["n_events"]:,} events, ⟨N_rec⟩ = {plot_data["avg_n"]:.1f}\n'
                             f'F={F:.3f}, G={G:.3f}, v2_2={v2_2:.3f}, v3_3={v3_3:.3f}')
                    ax.set_title(title, fontsize=11, pad=10)
                    ax.set_ylim(y_min, y_max)

        # Common settings for all subplots
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_xlim(-np.pi / 2, 3 * np.pi / 2)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        if ax_idx % 2 == 0:
            ax.set_ylabel(r'$Y(\Delta\phi)$', fontsize=12)
        if ax_idx >= 4:
            ax.set_xlabel(r'$\Delta\phi$', fontsize=12)
            
        # Add legend only if there are items to show
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # 图例移到右上角，不显示在图中
            ax.legend(handles, labels, fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')

    # Overall figure title
    param_str = (f'Particle selection: pT ∈ [{general_params["pt_min"]}, {general_params["pt_max"]}] GeV, '
                 f'|η| < {general_params["eta_max"]}, |Δη| < {delta_eta_cut}')
    fig.suptitle(r'Per-trigger-particle yields: $Y(\Delta\phi)$ Analysis' + '\n' +
                 r'Template fit: $Y_{templ}(\Delta\phi) = F \cdot Y_{periph}(\Delta\phi) + Y_{ridge}(\Delta\phi)$' + '\n' +
                 param_str, fontsize=14)
    
    # 调整布局，为图例留出空间
    fig.tight_layout(rect=[0, 0.03, 0.85, 0.95])

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {output_path}")
    plt.show()

def main():
    """
    Main function to run the entire analysis workflow with multiple delta eta cuts.
    """
    # --- User-configurable parameters ---
    data_dir = 'DDDAA'
    output_dir = "Y_analysis_results"

    # Physics cuts
    PT_MIN = 0.5         # Min particle pT
    PT_MAX = 5.0         # Max particle pT
    ETA_MAX = 1.1        # Max |η|

    # Delta eta cuts to analyze
    DELTA_ETA_CUTS = [1.0, 1.5, 2.0]

    # Data files for different multiplicity ranges
    data_files = {
        "0-20%": "multiplicity_group_0-20_percent.txt",
        "20-40%": "multiplicity_group_20-40_percent.txt", 
        "40-60%": "multiplicity_group_40-60_percent.txt",
        "60-80%": "multiplicity_group_60-80_percent.txt",
        "80-100%": "multiplicity_group_80-100_percent.txt"
    }

    # --- Analysis execution for each delta eta cut ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for delta_eta_cut in DELTA_ETA_CUTS:
        print(f"\n{'='*80}")
        print(f"ANALYZING WITH |Δη| > {delta_eta_cut}")
        print(f"{'='*80}")
        
        # 检查是否已经存在计算结果
        existing_results, data_path = check_existing_results(output_dir, {
            "pt_min": PT_MIN,
            "pt_max": PT_MAX,
            "eta_max": ETA_MAX
        }, delta_eta_cut)
        
        if existing_results:
            print(f"📁 Found existing results for |Δη| < {delta_eta_cut}")
            user_input = input("Do you want to use existing results? (y/n, default: y): ").strip().lower()
            if user_input != 'n':
                print("Loading existing results...")
                results_dict, Y_periph_data, general_params, loaded_delta_eta_cut = load_calculation_results(data_path)
                
                if results_dict is not None:
                    # 生成图表和保存数值结果
                    output_filename = sanitize_filename(
                        f"Y_analysis_pT{PT_MIN}-{PT_MAX}_eta{ETA_MAX}_deltaEta{delta_eta_cut}.png"
                    )
                    output_path = os.path.join(output_dir, output_filename)
                    
                    general_params = {
                        "pt_min": PT_MIN,
                        "pt_max": PT_MAX,
                        "eta_max": ETA_MAX
                    }
                    
                    plot_template_analysis_comparison(results_dict, Y_periph_data, general_params, output_path, delta_eta_cut)
                    
                    # 保存数值结果
                    results_filename = sanitize_filename(
                        f"Y_analysis_results_pT{PT_MIN}-{PT_MAX}_eta{ETA_MAX}_deltaEta{delta_eta_cut}.txt"
                    )
                    results_path = os.path.join(output_dir, results_filename)
                    
                    with open(results_path, 'w') as f:
                        f.write("Per-trigger-particle yields Y(Δφ) Analysis Results\n")
                        f.write("="*50 + "\n")
                        f.write(f"Particle selection: pT ∈ [{PT_MIN}, {PT_MAX}] GeV, |η| < {ETA_MAX}, |Δη| < {delta_eta_cut}\n\n")
                        
                        for range_label, data in results_dict.items():
                            if data is None:
                                f.write(f"{range_label}: No data\n\n")
                                continue
                                
                            f.write(f"{range_label}:\n")
                            f.write(f"  Events: {data['n_events']:,}\n")
                            f.write(f"  Average particles per event: {data['avg_n']:.2f}\n")
                            
                            if 'F' in data and data['F'] is not None:
                                f.write(f"  Template fit parameters:\n")
                                f.write(f"    F = {data['F']:.6f}\n")
                                f.write(f"    G = {data['G']:.6f}\n")
                                f.write(f"    v2_2 = {data['v2_2']:.6f}\n")
                                f.write(f"    v3_3 = {data['v3_3']:.6f}\n")
                            
                            f.write("\n")
                    
                    print(f"\n✅ Numerical results saved to {results_path}")
                    continue
        
        # 如果没有现有结果或用户选择重新计算，则进行计算
        print(f"🔄 Starting new calculation for |Δη| < {delta_eta_cut}")
        results_dict = {}
        Y_periph_data = (None, None)

        # Process each multiplicity range
        for range_label, filename in data_files.items():
            data_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(data_path):
                print(f"Warning: Data file {data_path} not found. Skipping...")
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing {range_label}: {filename}")
            print(f"{'='*60}")
            
            # Load and filter data
            valid_events, pt_min, pt_max, eta_max = load_events_from_classified_data(
                data_path, 
                pt_min=PT_MIN, pt_max=PT_MAX, eta_max=ETA_MAX
            )

            if not valid_events:
                print(f"No valid events found for {range_label}. Skipping...")
                results_dict[range_label] = None
                continue

            # Calculate correlation and yields with current delta eta cut
            phi_centers, Y_data, _, _ = calculate_correlation_with_yields_atlas_cut(
                valid_events, delta_eta_cut=delta_eta_cut
            )
            
            if Y_data is None:
                print(f"Failed to calculate Y(Δφ) for {range_label}. Skipping...")
                results_dict[range_label] = None
                continue

            # Store results
            avg_particles = np.mean([len(p) for p in valid_events.values()])
            results_dict[range_label] = {
                "phi_centers": phi_centers,
                "Y_data": Y_data,
                "n_events": len(valid_events),
                "avg_n": avg_particles
            }

            # Use 0-20% as peripheral template (lowest multiplicity)
            if range_label == "0-20%" and Y_periph_data[0] is None:
                Y_periph_data = (phi_centers, Y_data)
                print(f"Using {range_label} as peripheral template")

        # Perform template fits for non-peripheral ranges
        if Y_periph_data[0] is not None:
            print(f"\n{'='*60}")
            print("Performing template fits...")
            print(f"{'='*60}")
            
            for range_label in results_dict:
                if range_label == "0-20%" or results_dict[range_label] is None:
                    continue
                    
                print(f"\nFitting template for {range_label}...")
                plot_data = results_dict[range_label]
                
                F, G, v2_2, v3_3, Y_periph_interp = fit_template_model_with_v3(
                    plot_data['phi_centers'], plot_data['Y_data'], 
                    Y_periph_data[0], Y_periph_data[1]
                )
                
                # Update results with fit parameters
                plot_data.update({
                    "F": F,
                    "G": G, 
                    "v2_2": v2_2,
                    "v3_3": v3_3,
                    "Y_periph_interp": Y_periph_interp
                })

        # Generate the final comparison plot for this delta eta cut
        output_filename = sanitize_filename(
            f"Y_analysis_pT{PT_MIN}-{PT_MAX}_eta{ETA_MAX}_deltaEta{delta_eta_cut}.png"
        )
        output_path = os.path.join(output_dir, output_filename)
        
        general_params = {
            "pt_min": PT_MIN,
            "pt_max": PT_MAX,
            "eta_max": ETA_MAX
        }

        plot_template_analysis_comparison(results_dict, Y_periph_data, general_params, output_path, delta_eta_cut)

        # Save numerical results for this delta eta cut
        results_filename = sanitize_filename(
            f"Y_analysis_results_pT{PT_MIN}-{PT_MAX}_eta{ETA_MAX}_deltaEta{delta_eta_cut}.txt"
        )
        results_path = os.path.join(output_dir, results_filename)
        
        with open(results_path, 'w') as f:
            f.write("Per-trigger-particle yields Y(Δφ) Analysis Results\n")
            f.write("="*50 + "\n")
            f.write(f"Particle selection: pT ∈ [{PT_MIN}, {PT_MAX}] GeV, |η| < {ETA_MAX}, |Δη| < {delta_eta_cut}\n\n")
            
            for range_label, data in results_dict.items():
                if data is None:
                    f.write(f"{range_label}: No data\n\n")
                    continue
                    
                f.write(f"{range_label}:\n")
                f.write(f"  Events: {data['n_events']:,}\n")
                f.write(f"  Average particles per event: {data['avg_n']:.2f}\n")
                
                if 'F' in data and data['F'] is not None:
                    f.write(f"  Template fit parameters:\n")
                    f.write(f"    F = {data['F']:.6f}\n")
                    f.write(f"    G = {data['G']:.6f}\n")
                    f.write(f"    v2_2 = {data['v2_2']:.6f}\n")
                    f.write(f"    v3_3 = {data['v3_3']:.6f}\n")
                
                f.write("\n")
        
        print(f"\n✅ Numerical results saved to {results_path}")
        
        # 保存计算结果到pickle文件
        general_params = {
            "pt_min": PT_MIN,
            "pt_max": PT_MAX,
            "eta_max": ETA_MAX
        }
        save_calculation_results(results_dict, Y_periph_data, general_params, delta_eta_cut, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED FOR ALL DELTA ETA CUTS")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

def plot_from_saved_data(data_path, output_dir=None):
    """
    从保存的计算结果文件直接生成图表，无需重新计算
    
    Parameters:
    - data_path: 保存的计算结果文件路径
    - output_dir: 输出目录，如果为None则使用数据文件所在目录
    """
    print(f"📊 Loading data from {data_path}")
    
    # 加载计算结果
    results_dict, Y_periph_data, general_params, delta_eta_cut = load_calculation_results(data_path)
    
    if results_dict is None:
        print("❌ Failed to load data")
        return
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(data_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成图表
    output_filename = sanitize_filename(
        f"Y_analysis_pT{general_params['pt_min']}-{general_params['pt_max']}_eta{general_params['eta_max']}_deltaEta{delta_eta_cut}.png"
    )
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"🎨 Generating plot: {output_filename}")
    plot_template_analysis_comparison(results_dict, Y_periph_data, general_params, output_path, delta_eta_cut)
    
    # 保存数值结果
    results_filename = sanitize_filename(
        f"Y_analysis_results_pT{general_params['pt_min']}-{general_params['pt_max']}_eta{general_params['eta_max']}_deltaEta{delta_eta_cut}.txt"
    )
    results_path = os.path.join(output_dir, results_filename)
    
    with open(results_path, 'w') as f:
        f.write("Per-trigger-particle yields Y(Δφ) Analysis Results\n")
        f.write("="*50 + "\n")
        f.write(f"Particle selection: pT ∈ [{general_params['pt_min']}, {general_params['pt_max']}] GeV, |η| < {general_params['eta_max']}, |Δη| < {delta_eta_cut}\n\n")
        
        for range_label, data in results_dict.items():
            if data is None:
                f.write(f"{range_label}: No data\n\n")
                continue
                
            f.write(f"{range_label}:\n")
            f.write(f"  Events: {data['n_events']:,}\n")
            f.write(f"  Average particles per event: {data['avg_n']:.2f}\n")
            
            if 'F' in data and data['F'] is not None:
                f.write(f"  Template fit parameters:\n")
                f.write(f"    F = {data['F']:.6f}\n")
                f.write(f"    G = {data['G']:.6f}\n")
                f.write(f"    v2_2 = {data['v2_2']:.6f}\n")
                f.write(f"    v3_3 = {data['v3_3']:.6f}\n")
            
            f.write("\n")
    
    print(f"\n✅ Numerical results saved to {results_path}")
    print(f"✅ Plot saved to {output_path}")

def list_saved_results(output_dir="Y_analysis_results"):
    """
    列出所有保存的计算结果文件
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    print(f"📁 Saved calculation results in {output_dir}:")
    print("="*60)
    
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.pkl') and filename.startswith('calculation_results_'):
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    save_data = pickle.load(f)
                
                print(f"📊 {filename}")
                print(f"   Timestamp: {save_data['timestamp']}")
                print(f"   Delta eta cut: |Δη| < {save_data['delta_eta_cut']}")
                print(f"   Parameters: pT ∈ [{save_data['general_params']['pt_min']}, {save_data['general_params']['pt_max']}] GeV, |η| < {save_data['general_params']['eta_max']}")
                print(f"   File size: {os.path.getsize(file_path) / 1024:.1f} KB")
                print()
                
            except Exception as e:
                print(f"❌ {filename} - Error reading: {e}")
                print()
    
    print("="*60)
    print("💡 Use plot_from_saved_data() function to generate plots from saved results")
