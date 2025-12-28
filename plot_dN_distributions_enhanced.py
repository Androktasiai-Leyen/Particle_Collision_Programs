#!/usr/bin/env python3
"""
Enhanced script to plot dN/dpT, dN/dη and dN/dφ distributions
Using data from all 20 events files in DDDAA folder
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a font that supports all characters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def load_events_data(events_dir="DDDAA", pt_min=0.5, pt_max=5.0, eta_min=-1.1, eta_max=1.1):
    """
    Load data from all events files
    Data format: event_id particle_id particle_type pt phi eta
    """
    print("📂 Loading data from all events files...")
    
    # Find all events files
    events_files = glob.glob(os.path.join(events_dir, "events_*_converted.txt"))
    events_files.sort()
    
    if not events_files:
        print("❌ No events files found!")
        return None
    
    print(f"📊 Found {len(events_files)} events files")
    
    # Store all data
    all_pt = []
    all_eta = []
    all_phi = []
    all_events = []
    all_particle_types = []
    
    total_particles = 0
    filtered_particles = 0
    
    for file_path in tqdm(events_files, desc="Loading files"):
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
                        
                        # Apply selection criteria
                        if (pt_min <= pt <= pt_max and eta_min <= eta <= eta_max):
                            filtered_particles += 1
                            
                            all_pt.append(pt)
                            all_eta.append(eta)
                            all_phi.append(phi)
                            all_events.append(event_id)
                            all_particle_types.append(particle_type)
                            
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"⚠️ Error reading file {file_path}: {e}")
            continue
    
    print(f"\n✅ Data loading completed:")
    print(f"📈 Total particles: {total_particles:,}")
    print(f"📊 Filtered particles: {filtered_particles:,}")
    print(f"📊 Valid events: {len(set(all_events)):,}")
    
    return (np.array(all_pt), np.array(all_eta), np.array(all_phi), 
            all_events, np.array(all_particle_types))

def plot_dN_dpt(pt_data, pt_min=0.5, pt_max=5.0, output_path="dN_dpt_results.png"):
    """Plot dN/dpT distribution"""
    print("📊 Plotting dN/dpT distribution...")
    
    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    counts, bin_edges = np.histogram(pt_data, bins=50, range=(pt_min, pt_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Calculate dN/dpT (normalized to bin width)
    dN_dpt = counts / bin_widths
    
    # Linear plot
    ax1.errorbar(bin_centers, dN_dpt, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1, 
                label=f'Total particles: {len(pt_data):,}')
    ax1.set_xlabel(r'$p_T$ (GeV/c)', fontsize=14)
    ax1.set_ylabel(r'$dN/dp_T$ (GeV/c)$^{-1}$', fontsize=14)
    ax1.set_title(r'$dN/dp_T$ Distribution (Linear Scale)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log scale
    ax2.errorbar(bin_centers, dN_dpt, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1)
    ax2.set_xlabel(r'$p_T$ (GeV/c)', fontsize=14)
    ax2.set_ylabel(r'$dN/dp_T$ (GeV/c)$^{-1}$', fontsize=14)
    ax2.set_title(r'$dN/dp_T$ Distribution (Log Scale)', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ dN/dpT plot saved to: {output_path}")
    
    # Save data to text file
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# pT(GeV/c) dN/dpT error\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.4f} {dN_dpt[i]:.6f} {np.sqrt(counts[i])/bin_widths[i]:.6f}\n")
    
    print(f"💾 Data saved to: {data_file}")
    
    return bin_centers, dN_dpt, np.sqrt(counts)/bin_widths

def plot_dN_deta(eta_data, eta_min=-1.1, eta_max=1.1, output_path="dN_deta_results.png"):
    """Plot dN/dη distribution"""
    print("📊 Plotting dN/dη distribution...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create histogram
    counts, bin_edges = np.histogram(eta_data, bins=44, range=(eta_min, eta_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Calculate dN/dη (normalized to bin width)
    dN_deta = counts / bin_widths
    
    # Linear plot
    ax1.errorbar(bin_centers, dN_deta, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1,
                label=f'Total particles: {len(eta_data):,}')
    ax1.set_xlabel(r'$\eta$', fontsize=14)
    ax1.set_ylabel(r'$dN/d\eta$', fontsize=14)
    ax1.set_title(r'$dN/d\eta$ Distribution (Linear Scale)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log scale
    ax2.errorbar(bin_centers, dN_deta, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1)
    ax2.set_xlabel(r'$\eta$', fontsize=14)
    ax2.set_ylabel(r'$dN/d\eta$', fontsize=14)
    ax2.set_title(r'$dN/d\eta$ Distribution (Log Scale)', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ dN/dη plot saved to: {output_path}")
    
    # Save data to text file
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# eta dN/deta error\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.4f} {dN_deta[i]:.6f} {np.sqrt(counts[i])/bin_widths[i]:.6f}\n")
    
    print(f"💾 Data saved to: {data_file}")
    
    return bin_centers, dN_deta, np.sqrt(counts)/bin_widths

def plot_dN_dphi(phi_data, output_path="dN_dphi_results.png"):
    """Plot dN/dφ distribution"""
    print("📊 Plotting dN/dφ distribution...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create histogram (φ range: -π to π)
    counts, bin_edges = np.histogram(phi_data, bins=44, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Calculate dN/dφ (normalized to bin width)
    dN_dphi = counts / bin_widths
    
    # Linear plot
    ax1.errorbar(bin_centers, dN_dphi, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1,
                label=f'Total particles: {len(phi_data):,}')
    ax1.set_xlabel(r'$\phi$ (rad)', fontsize=14)
    ax1.set_ylabel(r'$dN/d\phi$ (rad)$^{-1}$', fontsize=14)
    ax1.set_title(r'$dN/d\phi$ Distribution (Linear Scale)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log scale
    ax2.errorbar(bin_centers, dN_dphi, yerr=np.sqrt(counts)/bin_widths, 
                fmt='o', markersize=4, capsize=3, capthick=1)
    ax2.set_xlabel(r'$\phi$ (rad)', fontsize=14)
    ax2.set_ylabel(r'$dN/d\phi$ (rad)$^{-1}$', fontsize=14)
    ax2.set_title(r'$dN/d\phi$ Distribution (Log Scale)', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ dN/dφ plot saved to: {output_path}")
    
    # Save data to text file
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# phi(rad) dN/dphi error\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.4f} {dN_dphi[i]:.6f} {np.sqrt(counts[i])/bin_widths[i]:.6f}\n")
    
    print(f"💾 Data saved to: {data_file}")
    
    return bin_centers, dN_dphi, np.sqrt(counts)/bin_widths

def plot_particle_type_distribution(particle_types, output_path="particle_type_distribution.png"):
    """Plot particle type distribution"""
    print("📊 Plotting particle type distribution...")
    
    # Count particle types
    unique_types, counts = np.unique(particle_types, return_counts=True)
    
    # Sort by count
    sort_idx = np.argsort(counts)[::-1]
    unique_types = unique_types[sort_idx]
    counts = counts[sort_idx]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    bars = ax1.bar(range(len(unique_types)), counts, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Particle Type (PDG Code)', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_title('Particle Type Distribution', fontsize=16)
    ax1.set_xticks(range(len(unique_types)))
    ax1.set_xticklabels([str(t) for t in unique_types], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # Pie chart (top 10 types)
    top_n = min(10, len(unique_types))
    top_types = unique_types[:top_n]
    top_counts = counts[:top_n]
    
    # Add "Others" if there are more types
    if len(unique_types) > top_n:
        top_types = np.append(top_types, -999)
        top_counts = np.append(top_counts, np.sum(counts[top_n:]))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_types)))
    wedges, texts, autotexts = ax2.pie(top_counts, labels=[str(t) if t != -999 else 'Others' for t in top_types],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title(f'Top {top_n} Particle Types', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ Particle type distribution saved to: {output_path}")
    
    # Save data to text file
    data_file = output_path.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# particle_type count percentage\n")
        total = np.sum(counts)
        for i, (ptype, count) in enumerate(zip(unique_types, counts)):
            percentage = (count / total) * 100
            f.write(f"{ptype} {count} {percentage:.2f}\n")
    
    print(f"💾 Data saved to: {data_file}")

def plot_combined_distributions(pt_data, eta_data, phi_data, output_path="combined_distributions.png"):
    """Plot combined distributions"""
    print("📊 Plotting combined distributions...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Set parameters
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
    axes[0, 0].set_title(r'$dN/dp_T$ Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].errorbar(bin_centers_pt, dN_dpt, yerr=np.sqrt(counts_pt)/bin_widths_pt, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[1, 0].set_xlabel(r'$p_T$ (GeV/c)')
    axes[1, 0].set_ylabel(r'$dN/dp_T$')
    axes[1, 0].set_title(r'$dN/dp_T$ Distribution (Log)')
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
    axes[0, 1].set_title(r'$dN/d\eta$ Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].errorbar(bin_centers_eta, dN_deta, yerr=np.sqrt(counts_eta)/bin_widths_eta, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[1, 1].set_xlabel(r'$\eta$')
    axes[1, 1].set_ylabel(r'$dN/d\eta$')
    axes[1, 1].set_title(r'$dN/d\eta$ Distribution (Log)')
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
    axes[0, 2].set_title(r'$dN/d\phi$ Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].errorbar(bin_centers_phi, dN_dphi, yerr=np.sqrt(counts_phi)/bin_widths_phi, 
                        fmt='o', markersize=3, capsize=2, capthick=1)
    axes[1, 2].set_xlabel(r'$\phi$ (rad)')
    axes[1, 2].set_ylabel(r'$dN/d\phi$')
    axes[1, 2].set_title(r'$dN/d\phi$ Distribution (Log)')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ Combined distributions saved to: {output_path}")

def main():
    """Main function"""
    print("="*80)
    print("🚀 Enhanced dN/dpT, dN/dη and dN/dφ Distribution Analysis")
    print("="*80)
    
    # Analysis parameters
    pt_min = 0.5
    pt_max = 5.0
    eta_min = -1.1
    eta_max = 1.1
    
    print(f"🎯 Analysis parameters:")
    print(f"   pT range: [{pt_min}, {pt_max}] GeV/c")
    print(f"   η range: [{eta_min}, {eta_max}]")
    
    # Load data
    data = load_events_data("DDDAA", pt_min, pt_max, eta_min, eta_max)
    if data is None:
        print("❌ Data loading failed!")
        return
    
    pt_data, eta_data, phi_data, events, particle_types = data
    
    print(f"\n📊 Data statistics:")
    print(f"   pT: mean={np.mean(pt_data):.3f}, std={np.std(pt_data):.3f}")
    print(f"   η: mean={np.mean(eta_data):.3f}, std={np.std(eta_data):.3f}")
    print(f"   φ: mean={np.mean(phi_data):.3f}, std={np.std(phi_data):.3f}")
    
    # Plot distributions
    print(f"\n🎨 Starting to plot distributions...")
    
    # dN/dpT
    plot_dN_dpt(pt_data, pt_min, pt_max, "dN_dpt_results.png")
    
    # dN/dη
    plot_dN_deta(eta_data, eta_min, eta_max, "dN_deta_results.png")
    
    # dN/dφ
    plot_dN_dphi(phi_data, "dN_dphi_results.png")
    
    # Particle type distribution
    plot_particle_type_distribution(particle_types, "particle_type_distribution.png")
    
    # Combined plot
    plot_combined_distributions(pt_data, eta_data, phi_data, "combined_distributions.png")
    
    print(f"\n✅ All distribution plots completed!")
    print(f"📊 Generated files:")
    print(f"   - dN_dpt_results.png and dN_dpt_results.txt")
    print(f"   - dN_deta_results.png and dN_deta_results.txt")
    print(f"   - dN_dphi_results.png and dN_dphi_results.txt")
    print(f"   - particle_type_distribution.png and particle_type_distribution.txt")
    print(f"   - combined_distributions.png")
    
    # Display statistics
    print(f"\n📈 Statistical summary:")
    print(f"   pT distribution: min={np.min(pt_data):.3f}, max={np.max(pt_data):.3f}")
    print(f"   η distribution: min={np.min(eta_data):.3f}, max={np.max(eta_data):.3f}")
    print(f"   φ distribution: min={np.min(phi_data):.3f}, max={np.max(phi_data):.3f}")
    
    # Additional analysis
    print(f"\n🔍 Additional analysis:")
    print(f"   Unique particle types: {len(np.unique(particle_types))}")
    print(f"   Most common particle type: {np.bincount(particle_types).argmax()}")
    print(f"   Events with particles: {len(set(events)):,}")
    
    # Calculate efficiency
    total_loaded = sum(1 for _ in glob.glob(os.path.join("DDDAA", "events_*_converted.txt")))
    print(f"   Events files processed: {total_loaded}")

if __name__ == "__main__":
    main()
