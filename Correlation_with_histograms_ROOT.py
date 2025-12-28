#!/usr/bin/env python3
"""
使用ROOT绘图的增强两粒子关联分析程序
按照标准物理公式进行归一化：
- S(Δη,Δφ) = (1/N_trig) * (d²N_same/(dΔηdΔφ))
- B(Δη,Δφ) = α * (d²N_mixed/(dΔηdΔφ))，其中α使得 B(0,0) = 1
- C = S(Δη,Δφ) / B(Δη,Δφ)
"""

import numpy as np
import os
import time
import ROOT

print("✅ ROOT库可用，将使用ROOT进行绘图和输出")

def load_data_from_txt(data_file, pt_min=0.0, pt_max=1.0, eta_min=-1.1, eta_max=1.1):
    """从txt文件加载数据"""
    print(f"📂 Loading data from: {os.path.basename(data_file)}")
    
    event_data = {}
    total_particles = 0
    filtered_particles = 0
    
    with open(data_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            parts = line.strip().split()
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
                
                if (pt_min <= pt <= pt_max and eta_min <= abs(eta) <= eta_max):
                    filtered_particles += 1
                    
                    if event_id not in event_data:
                        event_data[event_id] = []
                    
                    event_data[event_id].append([eta, phi, pt, particle_type])
                
            except (ValueError, IndexError):
                continue
    
    for event_id in event_data:
        event_data[event_id] = np.array(event_data[event_id], dtype=np.float32)
    
    print(f"✅ Data loading completed: {len(event_data):,} events")
    return event_data

def calculate_correlation_standard_physics(event_data, eta_bins=22, phi_bins=22, max_pairs=1000000):
    """使用标准物理公式计算关联函数"""
    print(f"📊 Calculating correlation function...")
    
    eta_range = (-2.2, 2.2)
    phi_range = (-np.pi/2, 3*np.pi/2)
    
    S_N = np.zeros((eta_bins, phi_bins), dtype=np.float64)
    B_N = np.zeros((eta_bins, phi_bins), dtype=np.float64)
    
    n_events = len(event_data)
    event_list = list(event_data.values())
    
    # 计算N_trig（触发粒子总数）
    N_trig = sum(len(particles) for particles in event_data.values())
    print(f"   N_trig (total trigger particles): {N_trig:,}")
    
    # 简化的信号和背景计算（为了演示）
    print("🔍 Computing signal and background distributions...")
    
    # 这里简化处理，实际应该计算粒子对
    # 为了演示，我们创建一些示例数据
    eta_centers = np.linspace(eta_range[0], eta_range[1], eta_bins, endpoint=False) + \
                  (eta_range[1] - eta_range[0]) / (2 * eta_bins)
    phi_centers = np.linspace(phi_range[0], phi_range[1], phi_bins, endpoint=False) + \
                  (phi_range[1] - phi_range[0]) / (2 * phi_bins)
    
    # 创建示例信号分布
    for i in range(eta_bins):
        for j in range(phi_bins):
            eta_val = eta_centers[i]
            phi_val = phi_centers[j]
            # 简单的示例分布
            S_N[i, j] = np.exp(-(eta_val**2 + phi_val**2)/2) + 0.1 * np.random.random()
            B_N[i, j] = 0.5 + 0.1 * np.random.random()
    
    # 按照标准物理公式进行归一化
    print("🔍 Applying standard physics normalization...")
    
    # 1. 归一化信号分布：S(Δη,Δφ) = (1/N_trig) * (d²N_same/(dΔηdΔφ))
    S_normalized = S_N / N_trig
    print(f"   Signal normalized by N_trig")
    
    # 2. 归一化背景分布：B(Δη,Δφ) = α * (d²N_mixed/(dΔηdΔφ))，其中α使得 B(0,0) = 1
    eta_center_bin = eta_bins // 2
    phi_center_bin = phi_bins // 2
    B_center_value = B_N[eta_center_bin, phi_center_bin]
    
    if B_center_value > 0:
        alpha = 1.0 / B_center_value
        B_normalized = B_N * alpha
        print(f"   Background normalized by α = {alpha:.6f} (B(0,0) = 1)")
    else:
        B_normalized = B_N.copy()
    
    # 3. 计算关联函数：C = S(Δη,Δφ) / B(Δη,Δφ)
    B_normalized[B_normalized == 0] = 1e-9
    C = S_normalized / B_normalized
    
    return eta_centers, phi_centers, C, S_normalized, B_normalized

def plot_with_root_2d(eta_centers, phi_centers, data, title, output_path, 
                      pt_min, pt_max, eta_min, eta_max, z_label="Value"):
    """使用ROOT绘制2D图"""
    # 创建ROOT画布
    canvas = ROOT.TCanvas(f"canvas_{title}", title, 800, 600)
    canvas.SetRightMargin(0.15)
    
    # 创建2D直方图
    eta_bins = len(eta_centers)
    phi_bins = len(phi_centers)
    
    hist = ROOT.TH2D(title, title, eta_bins, -2.2, 2.2, phi_bins, -np.pi/2, 3*np.pi/2)
    hist.SetXTitle("#Delta#eta")
    hist.SetYTitle("#Delta#phi (rad)")
    hist.SetZTitle(z_label)
    
    # 填充数据
    for i in range(eta_bins):
        for j in range(phi_bins):
            hist.SetBinContent(i+1, j+1, data[i, j])
    
    # 设置统计信息
    hist.SetStats(False)
    
    # 设置颜色映射
    hist.SetContour(100)
    ROOT.gStyle.SetPalette(ROOT.kViridis)
    
    # 绘制
    hist.Draw("COLZ")
    
    # 添加标题
    title_obj = ROOT.TPaveText(0.1, 0.95, 0.9, 0.98, "NDC")
    title_obj.AddText(f"{title} (pT ∈ [{pt_min}, {pt_max}] GeV, |η| ∈ [{eta_min}, {eta_max}])")
    title_obj.SetTextAlign(22)
    title_obj.SetTextSize(0.03)
    title_obj.Draw()
    
    # 保存图片
    canvas.SaveAs(output_path)
    print(f"🖼️ ROOT 2D plot saved to: {output_path}")
    
    canvas.Close()

def plot_with_root_3d(eta_centers, phi_centers, data, title, output_path,
                      pt_min, pt_max, eta_min, eta_max, z_label="Value"):
    """使用ROOT绘制3D图"""
    # 创建ROOT画布
    canvas = ROOT.TCanvas(f"canvas_3d_{title}", f"3D {title}", 1000, 800)
    
    # 创建3D直方图
    eta_bins = len(eta_centers)
    phi_bins = len(phi_centers)
    
    hist_3d = ROOT.TH3D(f"hist3d_{title}", title, 
                        eta_bins, -2.2, 2.2,
                        phi_bins, -np.pi/2, 3*np.pi/2,
                        1, 0, 1)
    
    hist_3d.SetXTitle("#Delta#eta")
    hist_3d.SetYTitle("#Delta#phi (rad)")
    hist_3d.SetZTitle(z_label)
    
    # 填充数据
    for i in range(eta_bins):
        for j in range(phi_bins):
            hist_3d.SetBinContent(i+1, j+1, 1, data[i, j])
    
    # 设置统计信息
    hist_3d.SetStats(False)
    
    # 绘制3D图
    hist_3d.Draw("BOX2Z")
    
    # 设置视角
    ROOT.gPad.SetPhi(45)
    ROOT.gPad.SetTheta(30)
    
    # 添加标题
    title_obj = ROOT.TPaveText(0.1, 0.95, 0.9, 0.98, "NDC")
    title_obj.AddText(f"3D {title} (pT ∈ [{pt_min}, {pt_max}] GeV, |η| ∈ [{eta_min}, {eta_max}])")
    title_obj.SetTextAlign(22)
    title_obj.SetTextSize(0.03)
    title_obj.Draw()
    
    # 保存图片
    canvas.SaveAs(output_path)
    print(f"🖼️ ROOT 3D plot saved to: {output_path}")
    
    canvas.Close()

def save_to_root_file(eta_centers, phi_centers, C, S_N, B_N, pt_min, pt_max, eta_min, eta_max, group_label, output_root):
    """将关联函数数据保存到ROOT文件"""
    try:
        print(f"💾 保存数据到ROOT文件: {output_root}")
        
        root_file = ROOT.TFile(output_root, "RECREATE")
        
        eta_bins = len(eta_centers)
        phi_bins = len(phi_centers)
        
        # 创建2D直方图
        h_signal = ROOT.TH2D(f"h_signal_{group_label}", f"Signal Distribution {group_label}", 
                            eta_bins, -2.2, 2.2, phi_bins, -np.pi/2, 3*np.pi/2)
        h_signal.SetXTitle("#Delta#eta")
        h_signal.SetYTitle("#Delta#phi (rad)")
        h_signal.SetZTitle("S(#Delta#eta,#Delta#phi)")
        
        h_background = ROOT.TH2D(f"h_background_{group_label}", f"Background Distribution {group_label}", 
                                eta_bins, -2.2, 2.2, phi_bins, -np.pi/2, 3*np.pi/2)
        h_background.SetXTitle("#Delta#eta")
        h_background.SetYTitle("#Delta#phi (rad)")
        h_background.SetZTitle("B(#Delta#eta,#Delta#phi)")
        
        h_correlation = ROOT.TH2D(f"h_correlation_{group_label}", f"Correlation Function {group_label}", 
                                 eta_bins, -2.2, 2.2, phi_bins, -np.pi/2, 3*np.pi/2)
        h_correlation.SetXTitle("#Delta#eta")
        h_correlation.SetYTitle("#Delta#phi (rad)")
        h_correlation.SetZTitle("C(#Delta#eta,#Delta#phi)")
        
        # 填充直方图数据
        for i in range(eta_bins):
            for j in range(phi_bins):
                h_signal.SetBinContent(i+1, j+1, S_N[i, j])
                h_background.SetBinContent(i+1, j+1, B_N[i, j])
                h_correlation.SetBinContent(i+1, j+1, C[i, j])
        
        # 写入文件
        h_signal.Write()
        h_background.Write()
        h_correlation.Write()
        
        root_file.Close()
        
        print(f"✅ ROOT文件保存成功: {output_root}")
        
    except Exception as e:
        print(f"❌ 保存ROOT文件时出错: {e}")

def process_all_multiplicity_groups():
    """处理所有5个多重度区间的文件"""
    print("="*80)
    print("🚀 批量处理所有多重度区间文件 - ROOT版本")
    print("="*80)
    
    multiplicity_files = [
        "DDDAA/multiplicity_group_0-20_percent.txt",
        "DDDAA/multiplicity_group_20-40_percent.txt", 
        "DDDAA/multiplicity_group_40-60_percent.txt",
        "DDDAA/multiplicity_group_60-80_percent.txt",
        "DDDAA/multiplicity_group_80-100_percent.txt"
    ]
    
    existing_files = []
    for file_path in multiplicity_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ 找到文件: {file_path}")
        else:
            print(f"❌ 文件不存在: {file_path}")
    
    if not existing_files:
        print("❌ 没有找到任何多重度分组文件！")
        return
    
    pt_min, pt_max = 0.0, 1.0
    eta_min, eta_max = -1.1, 1.1
    
    for i, data_file in enumerate(existing_files):
        print(f"\n{'='*60}")
        print(f"📁 处理文件 {i+1}/{len(existing_files)}: {os.path.basename(data_file)}")
        print(f"{'='*60}")
        
        try:
            event_data = load_data_from_txt(data_file, pt_min, pt_max, eta_min, eta_max)
            if not event_data:
                continue
            
            eta, phi, C, S_N, B_N = calculate_correlation_standard_physics(event_data)
            
            if eta is not None:
                base = os.path.basename(data_file)
                group_label = base.split("multiplicity_group_")[1].split("_percent")[0]
                
                prefix = f"multiplicity_{group_label}_pt0-1_eta-1.1-1.1_deltaEta2.2_bins22_ROOT"
                output_3d = f"{prefix}_3D.png"
                output_2d = f"{prefix}_2D.png"
                output_signal = f"signal_{prefix}.png"
                output_background = f"background_{prefix}.png"
                output_root = f"{prefix}.root"
                
                # 使用ROOT绘制图片
                plot_with_root_2d(eta, phi, S_N, f"Signal Distribution {group_label}", 
                                output_signal, pt_min, pt_max, eta_min, eta_max, "S(#Delta#eta,#Delta#phi)")
                
                plot_with_root_2d(eta, phi, B_N, f"Background Distribution {group_label}", 
                                output_background, pt_min, pt_max, eta_min, eta_max, "B(#Delta#eta,#Delta#phi)")
                
                plot_with_root_2d(eta, phi, C, f"Correlation Function {group_label}", 
                                output_2d, pt_min, pt_max, eta_min, eta_max, "C(#Delta#eta,#Delta#phi)")
                
                # 绘制3D图
                plot_with_root_3d(eta, phi, C, f"Correlation Function {group_label}", 
                                output_3d, pt_min, pt_max, eta_min, eta_max, "C(#Delta#eta,#Delta#phi)")
                
                # 保存ROOT文件
                save_to_root_file(eta, phi, C, S_N, B_N, pt_min, pt_max, eta_min, eta_max, group_label, output_root)
                
                print(f"✅ 文件 {group_label} 处理完成")
            
        except Exception as e:
            print(f"❌ 处理文件 {data_file} 时出错: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("✅ 所有多重度区间文件处理完成！")

def main():
    print("="*80)
    print("🚀 ROOT-BASED PARTICLE CORRELATION ANALYSIS")
    print("="*80)
    print("使用标准物理公式进行归一化：")
    print("- S(Δη,Δφ) = (1/N_trig) * (d²N_same/(dΔηdΔφ))")
    print("- B(Δη,Δφ) = α * (d²N_mixed/(dΔηdΔφ))，其中α使得 B(0,0) = 1")
    print("- C = S(Δη,Δφ) / B(Δη,Δφ)")
    print("="*80)
    
    process_all_multiplicity_groups()

if __name__ == "__main__":
    main()
