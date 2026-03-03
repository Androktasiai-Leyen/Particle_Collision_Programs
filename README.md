# Particle Collision Programs

A high-energy particle collision data analysis toolkit for constructing
two-particle correlation functions, histogram distributions, and
anisotropic flow observables from simulated or experimental datasets.

------------------------------------------------------------------------

## 1. Overview

This project provides a modular Python-based framework for:

-   Extracting and filtering particle-level data from collision events
-   Constructing Δη--Δφ two-particle correlation histograms
-   Performing mixed-event background subtraction
-   Applying ZYAM normalization
-   Computing anisotropic flow coefficients (v₂, v₃)
-   Exporting results to ROOT format (optional)
-   Generating enhanced 2D and 3D visualizations

The code is designed for heavy-ion or high-energy particle physics
analysis workflows.

------------------------------------------------------------------------

## 2. Installation

### 2.1 Requirements

Required Python packages:

    numpy
    pandas
    matplotlib
    tqdm
    numba

Install via:

    pip install numpy pandas matplotlib tqdm numba

Optional:

    ROOT (for CERN ROOT file output)

------------------------------------------------------------------------

## 3. Input Data Format

### 3.1 Text Format (space-separated)

Example:

    event_id particle_id particle_type pt phi eta
    1 101 211 0.85 1.52 0.23
    1 102 -211 0.74 -2.01 -0.17
    2 201 211 1.10 0.44 0.78

Columns:

  Column          Description
  --------------- ----------------------------
  event_id        Collision event identifier
  particle_id     Track identifier
  particle_type   PDG particle code
  pt              Transverse momentum
  phi             Azimuthal angle
  eta             Pseudorapidity

### 3.2 CSV Format

Example:

    event,track,pt,eta,phi,charge
    1,101,0.85,0.23,1.52,1

The program can auto-detect or explicitly specify format.

------------------------------------------------------------------------

## 4. Quick Start

### 4.1 Basic Correlation Analysis

    python Correlation_with_histograms.py --data your_data.txt --format auto

The program will:

1.  Load event data
2.  Apply pT and η cuts
3.  Construct same-event signal pairs
4.  Construct mixed-event background pairs
5.  Generate Δη--Δφ histograms
6.  Apply ZYAM normalization
7.  Output plots and optional ROOT file

------------------------------------------------------------------------

## 5. Example Workflow

### Example 1 --- Apply Custom pT Range

Run:

    python Correlation_with_histograms.py --data sample.txt

When prompted:

    Enter minimum pT: 0.5
    Enter maximum pT: 2.0

This restricts the analysis to particles within 0.5 \< pT \< 2.0 GeV.

------------------------------------------------------------------------

### Example 2 --- Multi-Multiplicity Batch Processing

If files exist:

    multiplicity_group_0_10_percent.txt
    multiplicity_group_10_30_percent.txt
    multiplicity_group_30_50_percent.txt

Run:

    python Correlation_with_histograms.py --all-multiplicity

The script will process all centrality bins automatically and generate
independent correlation outputs.

------------------------------------------------------------------------

### Example 3 --- Anisotropic Flow Analysis

    python anisotropic_flow_analysis.py --data sample.txt

Outputs:

-   v₂(pT)
-   v₃(pT)
-   Flow vs centrality plots

------------------------------------------------------------------------

## 6. Output Description

### 6.1 Generated Images

-   `*_2D.png` --- 2D correlation heatmap
-   `*_3D.png` --- 3D surface plot
-   `signal_histogram.png`
-   `background_histogram.png`

### 6.2 ROOT Output (if enabled)

Saved `.root` file contains:

-   TH2D: signal histogram
-   TH2D: background histogram
-   TH2D: normalized correlation
-   TH1D: Δη projection
-   TH1D: Δφ projection
-   TTree: correlation data points

------------------------------------------------------------------------

## 7. Algorithmic Structure

### 7.1 Signal Construction

For each event:

    for particle i:
        for particle j > i:
            Δη = η_i - η_j
            Δφ = φ_i - φ_j
            fill histogram

### 7.2 Mixed-Event Background

Particles from different events are paired to estimate combinatorial
background.

### 7.3 ZYAM Normalization

The minimum yield region is used to define baseline subtraction:

    C(Δφ) → C(Δφ) - C_min

------------------------------------------------------------------------

## 8. Performance Notes

For large datasets:

-   Limit maximum pair count
-   Increase bin width
-   Use event sampling
-   Consider multiprocessing extensions

------------------------------------------------------------------------

## 9. Directory Structure

    Particle_Collision_Programs/
    │
    ├── Correlation_with_histograms.py
    ├── anisotropic_flow_analysis.py
    ├── extract.py
    ├── plot_dN_distributions.py
    ├── convert_HM20_coordinates.py
    └── README.md

------------------------------------------------------------------------

## 10. Suggested Citation Format

If used in academic work, cite as:

    Particle Collision Programs,
    High-Energy Two-Particle Correlation Toolkit,
    GitHub Repository.

------------------------------------------------------------------------

## 11. Contribution Guidelines

1.  Fork repository
2.  Create feature branch
3.  Add documentation
4.  Submit Pull Request with reproducible example

------------------------------------------------------------------------
