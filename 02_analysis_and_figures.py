#!/usr/bin/env python3
"""
Population genetics analysis — MDS, clustering, and figures.

Linta Mahboob, MIPT (MSc thesis, supervisor Prof. Oleg Balanovsky).
Analysis run 2021; scripts tidied up for sharing in 2025.

Takes the 2,434 x 2,434 PLINK IBS distance matrix (PCA.mdist) and sample
metadata (pcAinfo.xlsx) and produces the MDS, clustering, heatmap, and
boxplot figures, plus the summary tables, used in the manuscript.

Run with default paths (PCA.mdist, pcAinfo.xlsx in the current directory):
    python3 02_analysis_and_figures.py

Or with explicit paths:
    python3 02_analysis_and_figures.py --mdist PCA.mdist --meta pcAinfo.xlsx --outdir ./output_figures

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, openpyxl, xlsxwriter
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for server/cluster use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS
warnings.filterwarnings('ignore')

# Matplotlib global settings for publication quality
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.titlesize'   : 15,
    'axes.labelsize'   : 13,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'figure.dpi'       : 100,
    'savefig.dpi'      : 300,
})

# Regional classification
SOUTH_ASIA   = ['Indian', 'Pakistani', 'Afghani', 'Nepalease']
WEST_ASIA    = ['Turkish', 'Iranian', 'Armenian', 'Lebanese', 'Cypriot',
                'Israeli', 'Palestinian', 'Jordanian', 'Iraqi', 'Saudi',
                'Syrian', 'Yemeni', 'Egypt', 'Azeri']
CENTRAL_ASIA = ['Kazakh', 'Kyrgyz', 'Uzbeks', 'Turkmen', 'Tajik', 'Tajik_4M']

# Color palette — consistent across all figures
REGION_COLORS = {
    'South Asian'  : '#E63946',
    'West Asian'   : '#2196F3',
    'Central Asian': '#4CAF50',
}
GROUP_COLORS = {}   # Filled after data load

def get_region(group):
    """Map a population group name to its regional classification."""
    if group in SOUTH_ASIA:    return 'South Asian'
    if group in WEST_ASIA:     return 'West Asian'
    if group in CENTRAL_ASIA:  return 'Central Asian'
    return 'Other'

def get_group_color(group):
    """Return the hex color for a population group based on its region."""
    region = get_region(group)
    return REGION_COLORS.get(region, '#888888')

# Legend patch helper
def region_legend():
    """Return a list of matplotlib patch handles for the region legend."""
    return [
        mpatches.Patch(facecolor='#E63946', label='South Asian',   linewidth=0),
        mpatches.Patch(facecolor='#2196F3', label='West Asian',    linewidth=0),
        mpatches.Patch(facecolor='#4CAF50', label='Central Asian', linewidth=0),
    ]

# Data loading
def load_data(mdist_path, meta_path):
    """
    Load the IBS distance matrix and sample metadata.

    Parameters
    ----------
    mdist_path : str
        Path to PLINK .mdist file (N x N whitespace-delimited distance matrix).
    meta_path  : str
        Path to metadata Excel file with columns:
        FIID, IID, GROUP, LABELS, REFERENCE, N, LATITUDES, LONGITUDES, COUNTRY

    Returns
    -------
    dist : np.ndarray, shape (N, N)
        Symmetric IBS distance matrix. Diagonal = 0.
    meta : pd.DataFrame
        Sample metadata aligned to dist rows/columns.
    """
    print("Loading IBS distance matrix...")
    dist = np.loadtxt(mdist_path)
    print(f"  Matrix shape : {dist.shape}")
    print(f"  Min distance : {dist.min():.6f}")
    print(f"  Max distance : {dist.max():.6f}")
    print(f"  Diagonal (should be 0): {dist.diagonal()[:3]}")

    print("Loading sample metadata...")
    meta = pd.read_excel(meta_path)
    print(f"  Samples : {len(meta)}")
    print(f"  Columns : {list(meta.columns)}")
    print(f"  Groups  : {sorted(meta['GROUP'].unique())}")

    assert dist.shape[0] == len(meta), \
        f"Mismatch: matrix has {dist.shape[0]} rows but metadata has {len(meta)} rows"

    meta['REGION'] = meta['GROUP'].apply(get_region)
    return dist, meta

# Analysis 1: multidimensional scaling
def run_mds(dist, n_components=3, max_iter=200, random_state=42):
    """
    Perform Multidimensional Scaling on the IBS distance matrix.

    MDS projects the high-dimensional pairwise distance structure into a
    low-dimensional coordinate space that preserves pairwise distances as
    faithfully as possible. This is equivalent to classical metric MDS
    (cmdscale in R) when using a precomputed distance matrix.

    Parameters
    ----------
    dist         : np.ndarray (N x N)  Symmetric distance matrix.
    n_components : int                 Number of MDS dimensions to compute.
    max_iter     : int                 Maximum iterations for optimization.
    random_state : int                 Random seed for reproducibility.

    Returns
    -------
    coords : np.ndarray (N x n_components)  MDS coordinates per individual.
    stress : float  Stress value (goodness of fit; lower = better).
    """
    print(f"\nRunning MDS (n_components={n_components}, max_iter={max_iter})...")
    mds = MDS(
        n_components  = n_components,
        dissimilarity = 'precomputed',
        random_state  = random_state,
        max_iter      = max_iter,
        n_init        = 1,
        normalized_stress = 'auto'
    )
    coords = mds.fit_transform(dist)
    print(f"  MDS stress : {mds.stress_:.4f}")
    return coords, mds.stress_

# Analysis 2: population-level pairwise distances
def compute_group_distances(dist, groups_array):
    """
    Compute average pairwise IBS distance between all pairs of population groups.

    For between-group distances: mean of all individual-level cross-group pairs.
    For within-group distances: mean of upper-triangle only (excluding self-pairs).

    Parameters
    ----------
    dist         : np.ndarray (N x N)  Full individual-level distance matrix.
    groups_array : np.ndarray (N,)     Group label per individual (same order as dist).

    Returns
    -------
    group_dist : pd.DataFrame (G x G)  Population-level pairwise distance matrix.
    """
    print("\nComputing population-level pairwise distances...")
    unique_groups = sorted(np.unique(groups_array))
    n_groups = len(unique_groups)
    group_dist = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)

    for g1 in unique_groups:
        idx1 = np.where(groups_array == g1)[0]
        for g2 in unique_groups:
            idx2 = np.where(groups_array == g2)[0]
            sub = dist[np.ix_(idx1, idx2)]
            if g1 == g2:
                upper = sub[np.triu_indices(len(idx1), k=1)]
                group_dist.loc[g1, g2] = float(upper.mean()) if len(upper) > 0 else 0.0
            else:
                group_dist.loc[g1, g2] = float(sub.mean())

    print(f"  Group distance matrix: {n_groups} x {n_groups}")
    return group_dist

# Figure 1: MDS scatter plot (all individuals)
def plot_mds_all(mds_df, out_path):
    """
    Figure 1: MDS scatter plot of all 2,434 individuals, showing both
    MDS1 vs MDS2 and MDS1 vs MDS3 side by side.

    Individual points are colored by regional group. Population centroids
    are marked with stars and labeled with bold population names.
    """
    print("\nGenerating Figure 1: MDS scatter plot (all individuals)...")

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor('white')

    plot_pairs = [('MDS1', 'MDS2'), ('MDS1', 'MDS3')]
    axis_labels = [('MDS Axis 1', 'MDS Axis 2'), ('MDS Axis 1', 'MDS Axis 3')]
    subtitles   = ['(a) MDS Axis 1 vs Axis 2', '(b) MDS Axis 1 vs Axis 3']

    for ax, (x_col, y_col), (xl, yl), subtitle in zip(
            axes, plot_pairs, axis_labels, subtitles):

        ax.set_facecolor('#F8F9FA')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#BBBBBB')

        # Plot individual-level points
        for group in mds_df['GROUP'].unique():
            sub = mds_df[mds_df['GROUP'] == group]
            color = get_group_color(group)
            ax.scatter(sub[x_col], sub[y_col],
                       c=color, s=25, alpha=0.55,
                       linewidths=0, rasterized=True)

        # Overlay population centroids with star markers and labeled halos
        for group in mds_df['GROUP'].unique():
            sub = mds_df[mds_df['GROUP'] == group]
            cx, cy = sub[x_col].mean(), sub[y_col].mean()
            color = get_group_color(group)
            ax.scatter(cx, cy, c=color, s=220, marker='*',
                       edgecolors='white', linewidths=1.0, zorder=5)
            ax.annotate(
                group, (cx, cy),
                fontsize=9, fontweight='bold',
                xytext=(5, 5), textcoords='offset points',
                color='#111111', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.25, ec='none')
            )

        ax.set_xlabel(xl, fontsize=13, fontweight='bold')
        ax.set_ylabel(yl, fontsize=13, fontweight='bold')
        ax.set_title(subtitle, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)

    fig.legend(handles=region_legend(), loc='lower center', ncol=3,
               fontsize=13, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03), edgecolor='#CCCCCC')
    fig.suptitle(
        'Figure 1. Multidimensional Scaling of IBS Genetic Distances\n'
        'South, West and Central Asian Populations (n = 2,434; 24 population groups)',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# Figure 2: pairwise distance heatmap
def plot_heatmap(group_dist, out_path):
    """
    Figure 2: Clustered heatmap of pairwise population-level IBS distances.

    Rows and columns are reordered by hierarchical clustering (average linkage).
    Color strips on margins encode regional group membership.
    """
    print("\nGenerating Figure 2: Pairwise distance heatmap...")

    gdist = group_dist.astype(float)
    row_colors = pd.Series(
        {g: get_group_color(g) for g in gdist.index}, name='Region'
    )
    link = linkage(gdist.values, method='average')

    g = sns.clustermap(
        gdist,
        row_linkage=link, col_linkage=link,
        row_colors=row_colors, col_colors=row_colors,
        cmap='RdYlBu_r',
        figsize=(16, 14),
        xticklabels=True, yticklabels=True,
        linewidths=0.4, linecolor='white',
        cbar_kws={'label': 'IBS Distance', 'shrink': 0.45}
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        fontsize=11, fontweight='bold', rotation=45, ha='right'
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=11, fontweight='bold', rotation=0
    )
    g.cax.tick_params(labelsize=10)
    g.cax.set_ylabel('IBS Distance', fontsize=11, fontweight='bold')

    g.ax_col_dendrogram.legend(
        handles=region_legend(), loc='center left',
        ncol=1, fontsize=11, frameon=True,
        bbox_to_anchor=(0.75, 0.5)
    )
    g.fig.suptitle(
        'Figure 2. Pairwise Genetic Distance Heatmap Between Population Groups\n'
        '(Average IBS Distance, Hierarchically Clustered — Average Linkage)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# Figure 3: hierarchical clustering dendrogram
def plot_dendrogram(group_dist, out_path):
    """
    Figure 3: Hierarchical clustering dendrogram of 24 population groups.

    Branch height = average IBS distance at which groups merge.
    Leaf labels are color-coded by regional group.
    """
    print("\nGenerating Figure 3: Hierarchical clustering dendrogram...")

    gdist  = group_dist.astype(float)
    link   = linkage(gdist.values, method='average')
    labels = list(gdist.index)

    region_col = {v: k for k, v in {
        'South Asian': '#E63946', 'West Asian': '#2196F3', 'Central Asian': '#4CAF50'
    }.items()}

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')

    dendrogram(
        link, labels=labels, ax=ax,
        orientation='top', leaf_rotation=40,
        color_threshold=0,
        above_threshold_color='#444444',
        leaf_font_size=12
    )

    # Color-code leaf tick labels by region
    for lbl in ax.get_xticklabels():
        lbl.set_color(get_group_color(lbl.get_text()))
        lbl.set_fontweight('bold')
        lbl.set_fontsize(12)

    ax.set_ylabel('Average IBS Genetic Distance', fontsize=13, fontweight='bold')
    ax.set_title(
        'Figure 3. Hierarchical Clustering Dendrogram of Asian Population Groups\n'
        'Based on Pairwise IBS Genetic Distances (n = 2,434; 24 groups)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(handles=region_legend(), loc='upper right', fontsize=11,
              frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# Figure 4: focus panels — south asian groups in regional context
def plot_focus_panels(mds_df, out_path):
    """
    Figure 4: Two-panel MDS focus plots highlighting Pakistani and Indian
    group positions relative to their closest neighbors.

    Background grey points show the full dataset for spatial reference.
    """
    print("\nGenerating Figure 4: South Asian focus MDS panels...")

    focus_sets = [
        (
            ['Pakistani', 'Afghani', 'Indian', 'Turkish',
             'Tajik', 'Tajik_4M', 'Kazakh', 'Kyrgyz', 'Uzbeks'],
            '(a) Pakistani Groups in Regional Context'
        ),
        (
            ['Indian', 'Pakistani', 'Nepalease',
             'Kazakh', 'Kyrgyz', 'Uzbeks', 'Tajik', 'Tajik_4M'],
            '(b) Indian & Himalayan Groups in Regional Context'
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor('white')

    for ax, (focus, title) in zip(axes, focus_sets):
        ax.set_facecolor('#F8F9FA')
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('#CCCCCC')

        # All individuals as grey background
        ax.scatter(mds_df['MDS1'], mds_df['MDS2'],
                   c='#E0E0E0', s=10, alpha=0.35, linewidths=0, zorder=1)

        # Focus group individuals and centroids
        for group in focus:
            sub = mds_df[mds_df['GROUP'] == group]
            if sub.empty:
                continue
            color = get_group_color(group)
            ax.scatter(sub['MDS1'], sub['MDS2'],
                       c=color, s=40, alpha=0.85, linewidths=0, zorder=3)
            cx, cy = sub['MDS1'].mean(), sub['MDS2'].mean()
            ax.scatter(cx, cy, c=color, s=280, marker='*',
                       edgecolors='white', linewidths=1.2, zorder=5)
            ax.annotate(
                group, (cx, cy),
                fontsize=10, fontweight='bold',
                xytext=(6, 6), textcoords='offset points',
                color='#111111', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', fc=color, alpha=0.3, ec='none')
            )

        ax.set_xlabel('MDS Axis 1', fontsize=13, fontweight='bold')
        ax.set_ylabel('MDS Axis 2', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.6)
        ax.tick_params(labelsize=11)

    fig.legend(handles=region_legend(), loc='lower center', ncol=3,
               fontsize=13, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03), edgecolor='#CCCCCC')
    fig.suptitle(
        'Figure 4. Regional Genetic Positioning: Focus on South Asian Populations',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# Figure 5: geographic map with genetic proximity links
def plot_geographic_map(meta, group_dist, out_path, proximity_percentile=15):
    """
    Figure 5: Geographic map showing sampled population locations.

    Dot size is proportional to sample size. Dashed lines connect
    population pairs in the bottom `proximity_percentile` of pairwise
    IBS distance (i.e., the most genetically similar pairs).

    Parameters
    ----------
    proximity_percentile : int
        Percentile threshold for drawing proximity links (default: 15).
    """
    print(f"\nGenerating Figure 5: Geographic map (proximity threshold = {proximity_percentile}th percentile)...")

    # Compute group geographic centroids and sample sizes
    geo = meta.groupby('GROUP').agg(
        LAT=('LATITUDES', 'mean'),
        LON=('LONGITUDES', 'mean'),
        N=('GROUP', 'count')
    ).reset_index()

    gdist    = group_dist.astype(float)
    groups   = list(gdist.index)
    all_vals = gdist.values[np.triu_indices(len(groups), k=1)]
    threshold = np.percentile(all_vals, proximity_percentile)

    geo_idx = geo.set_index('GROUP')

    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor('#DDEEFF')
    ax.set_facecolor('#DDEEFF')
    ax.set_xlim(20, 100)
    ax.set_ylim(5, 62)

    # Approximate land background
    ax.add_patch(plt.Rectangle((20, 5), 80, 57, color='#F5F0E8', zorder=0, linewidth=0))

    # Grid lines
    for lon in range(20, 105, 10):
        ax.axvline(lon, color='#CCCCCC', lw=0.5, alpha=0.6, zorder=1)
    for lat in range(10, 65, 10):
        ax.axhline(lat, color='#CCCCCC', lw=0.5, alpha=0.6, zorder=1)

    # Draw genetic proximity links
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i:
                continue
            d = gdist.loc[g1, g2]
            if d < threshold and g1 in geo_idx.index and g2 in geo_idx.index:
                x1 = geo_idx.loc[g1, 'LON']; y1 = geo_idx.loc[g1, 'LAT']
                x2 = geo_idx.loc[g2, 'LON']; y2 = geo_idx.loc[g2, 'LAT']
                alpha = 0.7 * (1 - (d - all_vals.min()) /
                               (threshold - all_vals.min() + 1e-9))
                ax.plot([x1, x2], [y1, y2], color='#777777',
                        lw=1.2, alpha=float(alpha), zorder=2, linestyle='--')

    # Population dots (size proportional to N)
    for _, row in geo.iterrows():
        if row['GROUP'] not in geo_idx.index:
            continue
        color = get_group_color(row['GROUP'])
        sz    = 80 + int(row['N']) * 0.35
        ax.scatter(row['LON'], row['LAT'],
                   c=color, s=sz, zorder=4,
                   edgecolors='white', linewidths=1.2, alpha=0.92)
        ax.annotate(
            row['GROUP'], (row['LON'], row['LAT']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=9.5, fontweight='bold',
            color='#111111', zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.3, ec='none')
        )

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude',  fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 5. Geographic Distribution of Sampled Populations\n'
        f'Dot size ∝ sample size  |  Dashed lines = genetic proximity '
        f'(bottom {proximity_percentile}th percentile IBS distance)',
        fontsize=13, fontweight='bold', pad=14
    )

    legend_elements = region_legend() + [
        plt.Line2D([0], [0], color='#777777', lw=1.5,
                   linestyle='--', label='Genetic proximity link')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              frameon=True, framealpha=0.95, edgecolor='#AAAAAA')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#DDEEFF')
    plt.close()
    print(f"  Saved: {out_path}")

# Figure 6: within- and between-region distance boxplots
def plot_region_boxplots(group_dist, out_path):
    """
    Figure 6: Boxplots comparing within-region and between-region IBS distances
    for all three regional pairings.

    Median values are annotated on each box. This confirms that regional
    groupings capture genuine genetic structure (within < between distances).
    """
    print("\nGenerating Figure 6: Within/between region distance boxplots...")

    gdist = group_dist.astype(float)
    region_pal = {'South Asian': '#E63946', 'West Asian': '#2196F3',
                  'Central Asian': '#4CAF50'}

    region_pairs = [
        ('South Asian',   'West Asian',    '(a) South Asian vs West Asian'),
        ('South Asian',   'Central Asian', '(b) South Asian vs Central Asian'),
        ('West Asian',    'Central Asian', '(c) West Asian vs Central Asian'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('white')

    for ax, (r1, r2, title) in zip(axes, region_pairs):

        g1_list = [g for g in gdist.index if get_region(g) == r1]
        g2_list = [g for g in gdist.index if get_region(g) == r2]

        between = gdist.loc[g1_list, g2_list].values.flatten()
        within1 = [gdist.loc[g, g] for g in g1_list if gdist.loc[g, g] > 0]
        within2 = [gdist.loc[g, g] for g in g2_list if gdist.loc[g, g] > 0]

        data   = [within1, within2, list(between)]
        labels = [f'Within\n{r1}', f'Within\n{r2}', 'Between\nRegions']
        colors = [region_pal[r1], region_pal[r2], '#FF9800']

        bp = ax.boxplot(
            data, patch_artist=True, notch=False, widths=0.55,
            medianprops=dict(color='white', linewidth=2.5),
            whiskerprops=dict(linewidth=1.5, color='#555555'),
            capprops=dict(linewidth=1.5, color='#555555'),
            flierprops=dict(marker='o', markersize=4, alpha=0.5,
                            markerfacecolor='#999999', markeredgecolor='none')
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

        # Annotate median values on each box
        for i, d in enumerate(data, 1):
            med = np.median(d)
            ax.text(i, med + 0.0003, f'{med:.4f}',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#333333')

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('IBS Genetic Distance', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_facecolor('#F8F9FA')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for sp in ax.spines.values():
            sp.set_color('#CCCCCC')
        ax.tick_params(axis='y', labelsize=10)

    fig.suptitle(
        'Figure 6. Within- and Between-Region Genetic Distance Distributions\n'
        '(IBS distance; median values annotated)',
        fontsize=14, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# Table 1: population summary statistics
def build_summary_table(meta, group_dist):
    """
    Build a summary table with per-population statistics.

    Columns: Population, Region, N, Within-group IBS distance,
             Mean distance to others, Nearest population, Nearest distance.
    """
    print("\nBuilding population summary statistics table...")

    gdist = group_dist.astype(float)
    rows  = []
    for grp in sorted(meta['GROUP'].unique()):
        n       = int((meta['GROUP'] == grp).sum())
        region  = get_region(grp)
        wd      = float(gdist.loc[grp, grp]) if grp in gdist.index else np.nan
        other   = gdist.loc[grp].drop(grp)   if grp in gdist.index else pd.Series()
        md      = float(other.mean())          if len(other) > 0 else np.nan
        nearest = str(other.idxmin())          if len(other) > 0 else 'N/A'
        nd      = float(other.min())           if len(other) > 0 else np.nan
        rows.append({
            'Population'              : grp,
            'Region'                  : region,
            'N'                       : n,
            'Within-group IBS dist.'  : round(wd, 4),
            'Mean dist. to others'    : round(md, 4),
            'Nearest population'      : nearest,
            'Nearest IBS dist.'       : round(nd, 4),
        })

    return pd.DataFrame(rows)

def plot_summary_table(summary_df, out_path):
    """Render summary_df as a publication-quality figure table image."""
    print("\nGenerating Table 1 figure...")

    region_bg = {
        'South Asian'  : '#FDECEA',
        'West Asian'   : '#E3F2FD',
        'Central Asian': '#E8F5E9',
    }
    cell_data = []
    row_colors_list = []
    for _, r in summary_df.iterrows():
        cell_data.append([
            r['Population'], r['Region'], r['N'],
            f"{r['Within-group IBS dist.']:.4f}",
            f"{r['Mean dist. to others']:.4f}",
            r['Nearest population'],
            f"{r['Nearest IBS dist.']:.4f}",
        ])
        c = region_bg.get(r['Region'], '#FFFFFF')
        row_colors_list.append([c] * 7)

    col_labels = [
        'Population', 'Region', 'N',
        'Within-group\nIBS dist.',
        'Mean dist.\nto others',
        'Nearest\npopulation',
        'Nearest\nIBS dist.',
    ]

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    tbl = ax.table(
        cellText=cell_data, colLabels=col_labels,
        cellLoc='center', loc='center',
        cellColours=row_colors_list
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BBBBBB')
        if row == 0:
            cell.set_facecolor('#263238')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)

    import matplotlib.patches as mpatches
    legend_els = [
        mpatches.Patch(facecolor='#FDECEA', label='South Asian',   edgecolor='#BBBBBB'),
        mpatches.Patch(facecolor='#E3F2FD', label='West Asian',    edgecolor='#BBBBBB'),
        mpatches.Patch(facecolor='#E8F5E9', label='Central Asian', edgecolor='#BBBBBB'),
    ]
    ax.legend(handles=legend_els, loc='lower right', fontsize=11,
              frameon=True, framealpha=0.9)
    ax.set_title(
        'Table 1. Population Summary Statistics and Pairwise Genetic Distance Metrics\n'
        '(IBS distance; n = 2,434 individuals across 24 population groups)',
        fontsize=13, fontweight='bold', pad=20, y=0.99
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# Export raw data to excel
def export_excel(mds_df, group_dist, summary_df, out_path):
    """
    Export all raw analysis data to a 4-sheet Excel workbook:
      Sheet 1 — Population summary statistics (Table 1)
      Sheet 2 — Full 24 x 24 pairwise distance matrix (Table 2)
      Sheet 3 — MDS coordinates per individual (Figures 1 & 4 source)
      Sheet 4 — Region-level distance statistics (Figure 6 source)
    """
    print(f"\nExporting raw data to Excel: {out_path}")

    writer = pd.ExcelWriter(
        out_path, engine='xlsxwriter',
        engine_kwargs={'options': {'nan_inf_to_errors': True}}
    )
    wb = writer.book

    # Common formats
    hdr  = wb.add_format({'bold': True, 'bg_color': '#263238', 'font_color': 'white',
                           'border': 1, 'align': 'center', 'font_size': 11})
    sa   = wb.add_format({'bg_color': '#FDECEA', 'border': 1, 'align': 'center', 'font_size': 10})
    wa   = wb.add_format({'bg_color': '#E3F2FD', 'border': 1, 'align': 'center', 'font_size': 10})
    ca   = wb.add_format({'bg_color': '#E8F5E9', 'border': 1, 'align': 'center', 'font_size': 10})
    num  = wb.add_format({'num_format': '0.0000', 'border': 1, 'align': 'center', 'font_size': 10})
    note = wb.add_format({'italic': True, 'font_size': 9, 'font_color': '#555555'})
    titl = wb.add_format({'bold': True, 'font_size': 13, 'font_color': '#263238'})

    reg_fmt = {'South Asian': sa, 'West Asian': wa, 'Central Asian': ca}

    # Sheet 1: Summary stats
    ws1 = wb.add_worksheet('Table1_Summary')
    ws1.write(0, 0, 'Table 1. Population Summary Statistics', titl)
    ws1.write(1, 0, 'IBS = Identity-by-State distance (PLINK); n=2,434 individuals, 24 groups', note)
    cols1 = list(summary_df.columns)
    for c, col in enumerate(cols1):
        ws1.write(3, c, col, hdr)
        ws1.set_column(c, c, max(16, len(col) + 2))
    for r, row in summary_df.iterrows():
        reg  = row['Region']
        rfmt = reg_fmt.get(reg, sa)
        ws1.write(r + 4, 0, row['Population'],             rfmt)
        ws1.write(r + 4, 1, row['Region'],                 rfmt)
        ws1.write(r + 4, 2, int(row['N']),                 rfmt)
        ws1.write(r + 4, 3, row['Within-group IBS dist.'], num)
        ws1.write(r + 4, 4, row['Mean dist. to others'],   num)
        ws1.write(r + 4, 5, row['Nearest population'],     rfmt)
        ws1.write(r + 4, 6, row['Nearest IBS dist.'],      num)
    ws1.freeze_panes(4, 0)

    # Sheet 2: Pairwise distance matrix
    gdist    = group_dist.astype(float)
    grp_list = list(gdist.index)
    flat     = gdist.values[np.triu_indices(len(grp_list), k=1)]
    lo_t, hi_t = np.percentile(flat, 25), np.percentile(flat, 75)

    lo_f  = wb.add_format({'num_format':'0.0000','bg_color':'#C8E6C9','border':1,'align':'center','font_size':9})
    hi_f  = wb.add_format({'num_format':'0.0000','bg_color':'#FFCDD2','border':1,'align':'center','font_size':9})
    mid_f = wb.add_format({'num_format':'0.0000','bg_color':'#FFF9C4','border':1,'align':'center','font_size':9})
    dg_f  = wb.add_format({'num_format':'0.0000','bg_color':'#B0BEC5','border':1,'align':'center','font_size':9,'bold':True})

    ws2 = wb.add_worksheet('Table2_PairwiseDist')
    ws2.write(0, 0, 'Table 2. Pairwise Average IBS Genetic Distance Between Population Groups', titl)
    ws2.write(3, 0, 'Population', hdr); ws2.set_column(0, 0, 14)
    for c, g in enumerate(grp_list):
        ws2.write(3, c + 1, g, hdr); ws2.set_column(c + 1, c + 1, 10)
    for r, g1 in enumerate(grp_list):
        ws2.write(r + 4, 0, g1, reg_fmt.get(get_region(g1), sa))
        for c, g2 in enumerate(grp_list):
            val = float(gdist.loc[g1, g2])
            fmt = dg_f if g1 == g2 else (lo_f if val <= lo_t else (hi_f if val >= hi_t else mid_f))
            ws2.write(r + 4, c + 1, val, fmt)
    ws2.freeze_panes(4, 1)
    ws2.write(len(grp_list) + 6, 0,
              'Green=similar | Yellow=intermediate | Red=distant | Grey=within-group', note)

    # Sheet 3: MDS coordinates
    ws3 = wb.add_worksheet('MDS_Coordinates')
    ws3.write(0, 0, 'MDS Coordinates — Source data for Figures 1 and 4', titl)
    ws3.write(1, 0,
              'MDS from 2,434x2,434 IBS distance matrix; '
              'scikit-learn MDS(n_components=3, dissimilarity="precomputed")', note)
    mds_out  = mds_df[['FIID','IID','GROUP','LABELS','COUNTRY','LATITUDES','LONGITUDES','MDS1','MDS2','MDS3']].copy()
    mds_cols = list(mds_out.columns)
    for c, col in enumerate(mds_cols):
        ws3.write(3, c, col, hdr); ws3.set_column(c, c, max(12, len(col) + 2))
    for r, row in mds_out.iterrows():
        rfmt  = reg_fmt.get(get_region(str(row['GROUP'])), sa)
        for c, col in enumerate(mds_cols):
            val = row[col]
            if col in ['MDS1', 'MDS2', 'MDS3', 'LATITUDES', 'LONGITUDES']:
                ws3.write(r + 4, c, float(val) if not pd.isna(val) else 0.0, num)
            else:
                ws3.write(r + 4, c, str(val), rfmt)
    ws3.freeze_panes(4, 0)

    # Sheet 4: Region-level stats
    ws4 = wb.add_worksheet('Region_Distance_Stats')
    ws4.write(0, 0, 'Region-Level Distance Statistics — Source data for Figure 6', titl)
    pairs = [('South Asian','West Asian'),('South Asian','Central Asian'),('West Asian','Central Asian')]
    stat_rows = []
    for r1, r2 in pairs:
        g1l = [g for g in gdist.index if get_region(g) == r1]
        g2l = [g for g in gdist.index if get_region(g) == r2]
        btwn = gdist.loc[g1l, g2l].values.flatten()
        w1   = [gdist.loc[g, g] for g in g1l if gdist.loc[g, g] > 0]
        w2   = [gdist.loc[g, g] for g in g2l if gdist.loc[g, g] > 0]
        for label, vals in [(f'Within {r1}', w1), (f'Within {r2}', w2), (f'Between {r1} & {r2}', btwn)]:
            stat_rows.append([
                label, len(vals),
                round(float(np.mean(vals)), 4), round(float(np.median(vals)), 4),
                round(float(np.std(vals)),  4), round(float(np.min(vals)),    4),
                round(float(np.max(vals)),  4),
            ])
    scols = ['Comparison', 'N pairs', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
    for c, col in enumerate(scols):
        ws4.write(3, c, col, hdr); ws4.set_column(c, c, max(14, len(col) + 2))
    for r, row in enumerate(stat_rows):
        ws4.write(r + 4, 0, row[0], wa)
        ws4.write(r + 4, 1, int(row[1]), wa)
        for c in range(2, 7):
            ws4.write(r + 4, c, row[c], num)
    ws4.freeze_panes(4, 0)

    writer.close()
    print(f"  Saved: {out_path}")

# Main entry point
def parse_args():
    parser = argparse.ArgumentParser(
        description='Population genetics analysis: MDS, clustering, figures'
    )
    parser.add_argument('--mdist',  default='PCA.mdist',    help='PLINK .mdist file')
    parser.add_argument('--meta',   default='pcAinfo.xlsx', help='Sample metadata Excel file')
    parser.add_argument('--outdir', default='./figures',    help='Output directory for figures')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Population genetics analysis — South / West / Central Asian populations")

    # Load data
    dist, meta = load_data(args.mdist, args.meta)

    # MDS
    coords, stress = run_mds(dist, n_components=3)
    mds_df = meta.copy()
    mds_df['MDS1'] = coords[:, 0]
    mds_df['MDS2'] = coords[:, 1]
    mds_df['MDS3'] = coords[:, 2]
    mds_df.to_csv(os.path.join(args.outdir, 'mds_coordinates.csv'), index=False)
    print(f"  MDS coordinates saved.")

    # Group-level distances
    group_dist = compute_group_distances(dist, meta['GROUP'].values)
    group_dist.to_csv(os.path.join(args.outdir, 'group_distances.csv'))
    print(f"  Group distances saved.")

    # Summary table
    summary_df = build_summary_table(meta, group_dist)

    # Figures
    plot_mds_all(mds_df,      os.path.join(args.outdir, 'Figure1_MDS.png'))
    plot_heatmap(group_dist,  os.path.join(args.outdir, 'Figure2_Heatmap.png'))
    plot_dendrogram(group_dist, os.path.join(args.outdir, 'Figure3_Dendrogram.png'))
    plot_focus_panels(mds_df, os.path.join(args.outdir, 'Figure4_FocusPanel.png'))
    plot_geographic_map(meta, group_dist, os.path.join(args.outdir, 'Figure5_GeoMap.png'))
    plot_region_boxplots(group_dist, os.path.join(args.outdir, 'Figure6_Boxplots.png'))
    plot_summary_table(summary_df, os.path.join(args.outdir, 'Table1_Summary.png'))

    # Excel export
    export_excel(
        mds_df, group_dist, summary_df,
        os.path.join(args.outdir, 'Raw_Analysis_Data.xlsx')
    )

    print(f"\nDone. All outputs saved to: {args.outdir}/")


if __name__ == '__main__':
    main()
