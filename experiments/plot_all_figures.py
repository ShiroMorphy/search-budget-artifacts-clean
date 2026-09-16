"""
Publication-Quality Figure Generator for Chaos, Solitons & Fractals
==========================================================
Generates Figures 1 to 4 in PDF and PNG formats:
- Fig 1: Fraction of analytical optimum attained vs operator Frobenius norm ||A||_F.
- Fig 2: Feasible-witness failure certificate on empirical regularization trajectory.
- Fig 3: Dimension corruption artifact: selected q* vs operator contraction.
- Fig 4: Generalization outside finance: Coupled network (coupling strength g)
         and LR vs. Iterations phase grid.
"""

import os
import sys

# Matplotlib and fontconfig need writable cache directories.  On machines where
# the default locations under the home directory are not writable, the font
# machinery aborts before any figure is drawn (on macOS this surfaces as
# "Fontconfig error: No writable cache directories" followed by "Abort trap: 6").
# Redirecting both caches into the repository removes that dependency.  These
# assignments must precede the matplotlib import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_REPO_ROOT, ".cache")
os.makedirs(os.path.join(_CACHE, "matplotlib"), exist_ok=True)
os.makedirs(os.path.join(_CACHE, "fontconfig"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Publication style for the Elsevier submission package.
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.7,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
})

os.makedirs("figures", exist_ok=True)

def plot_fig1():
    """Fig 1: Scaling of the attainment fraction vs operator norm."""
    df = pd.read_csv("experiments/operator_scaling.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.88, wspace=0.38)
    
    # Left: Fraction of analytical optimum attained
    ax1.plot(df["frob_norm"], df["attn_adaptive"] * 100, 'o-', color='#1b7837', label='Monotone backtracking (12/100)')
    ax1.plot(df["frob_norm"], df["attn_fixed"] * 100, 's--', color='#d73027', label=r'Fixed step $\eta=0.05$ (12/100)')
    ax1.plot(df["frob_norm"], df["attn_norm_q1"] * 100, '^-.', color='#4575b4', label='Normalized Fixed (12/100)')
    
    ax1.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Operator Norm $\|A\|_F$')
    ax1.set_ylabel('Attained Analytical Optimum (%)', labelpad=8)
    ax1.set_ylim(-2, 108)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.set_title('(a) Convergence Ratio vs. Scale')

    # Right: Riemannian Gradient Norm at initialization W0
    ax2.plot(df["frob_norm"], df["grad_norm_W0"], 'd-', color='#762a83', label=r'$\|\operatorname{grad}_{\mathrm{St}} f(W_0)\|$')
    # Reference quadratic scaling curve: c * ||A||^2
    ref_c = df["grad_norm_W0"].iloc[0] / (df["frob_norm"].iloc[0]**2)
    ref_y = ref_c * (df["frob_norm"]**2)
    ax2.plot(df["frob_norm"], ref_y, 'k:', label=r'Quadratic Scaling $\mathcal{O}(\|A\|_F^2)$')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Operator Norm $\|A\|_F$')
    ax2.set_ylabel(r'Riemannian Gradient Norm')
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax2.set_title(r'(b) Vanishing Gradient $\mathcal{O}(\|A\|^2)$')

    plt.savefig("figures/fig1_operator_scaling.pdf", pad_inches=0.08)
    plt.savefig("figures/fig1_operator_scaling.png", dpi=300, pad_inches=0.08)
    plt.close()
    print("Generated figures/fig1_operator_scaling.pdf and .png")

def plot_fig2():
    """Fig 2: Feasible-witness failure certificate on empirical regularization trajectory."""
    df_c1 = pd.read_csv("evidence/q1_bounds_summary.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.88, wspace=0.38)
    
    # Left: The three curves of the bracket
    ax1.plot(df_c1["lambda_0"], df_c1["q1_feasible_cefi"], 'o-', color='#1b7837', label=r'Feasible Lower Bound ($W \in \mathrm{St}$)')
    ax1.plot(df_c1["lambda_0"], df_c1["mean_cefi"], 's--', color='#d73027', label='Reported Production (12/100)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Regularization Parameter $\lambda_0$')
    ax1.set_ylabel(r'Causal-Emergence Density $\Delta J$', labelpad=8)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax1.set_title('(a) Feasible-Witness Suboptimality Certificate')
    
    # Right: Feasible-witness to reported-estimate ratio
    ax2.plot(df_c1["lambda_0"], df_c1["bound_to_reported"], 'D-', color='#b2182b', linewidth=1.6)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    # Annotate the peak deficit, read from the data
    peak_row = df_c1.loc[df_c1["bound_to_reported"].idxmax()]
    ax2.annotate(f'Peak {peak_row["bound_to_reported"]:.1f}x deficit\n'
                 f'at $\\lambda_0={peak_row["lambda_0"]:g}$',
                 xy=(peak_row["lambda_0"], peak_row["bound_to_reported"]),
                 xytext=(0.56, 0.82), textcoords='axes fraction',
                 ha='left', va='top',
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.1),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='none'),
                 fontsize=10)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Regularization Parameter $\lambda_0$')
    ax2.set_ylabel(r'Deficit Factor (Feasible / Reported)')
    ax2.set_title('(b) Optimization Deficit Ratio')
    
    plt.savefig("figures/fig2_diagnostic_bracket.pdf", pad_inches=0.08)
    plt.savefig("figures/fig2_diagnostic_bracket.png", dpi=300, pad_inches=0.08)
    plt.close()
    print("Generated figures/fig2_diagnostic_bracket.pdf and .png")

def plot_fig3():
    """Fig 3: Macro-dimension corruption artifact."""
    df_syn = pd.read_csv("experiments/synthetic_q_corruption.csv")
    df_emp = pd.read_csv("experiments/empirical_q_corruption.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.88, wspace=0.38)
    
    # Left: Synthetic system dimension selection
    ax1.plot(df_syn["alpha"], df_syn["q_true"], 'k--', label=r'Ground Truth ($q^*=1$)', linewidth=1.8)
    ax1.plot(df_syn["alpha"], df_syn["q_adapt_12_100"], 'o-', color='#1b7837', label='Monotone backtracking')
    ax1.plot(df_syn["alpha"], df_syn["q_fixed_12_100"], 's--', color='#d73027', label='Fixed 12/100')
    ax1.plot(df_syn["alpha"], df_syn["q_fixed_4_35"], '^:', color='#fc8d59', label='Fixed 4/35')
    
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Operator Scale $\alpha$')
    ax1.set_ylabel(r'Reported Selected Dimension $\hat q$')
    ax1.set_ylim(0.5, 3.5)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax1.set_title(r'(a) Synthetic System ($p=30$)')
    
    # Right: Empirical system modal dimension shift across regularization
    ax2.plot(df_emp["lambda_0"], df_emp["modal_q_svd"], 'k--', label='SVD upper-bound selection', linewidth=1.8)
    ax2.plot(df_emp["lambda_0"], df_emp["modal_q_adapt"], 'o-', color='#1b7837', label='Monotone-backtracking selection')
    ax2.plot(df_emp["lambda_0"], df_emp["modal_q_fixed"], 's--', color='#d73027', label='Fixed 12/100 selection')
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Regularization Parameter $\lambda_0$')
    ax2.set_ylabel(r'Modal Reported Dimension $\hat q$')
    ax2.set_ylim(-1, 31)
    ax2.legend(loc='center left', frameon=True, framealpha=0.9)
    ax2.set_title(r'(b) Empirical Trajectory ($p=30$)')
    
    plt.savefig("figures/fig3_dimension_corruption.pdf", pad_inches=0.08)
    plt.savefig("figures/fig3_dimension_corruption.png", dpi=300, pad_inches=0.08)
    plt.close()
    print("Generated figures/fig3_dimension_corruption.pdf and .png")

def plot_fig4():
    """Fig 4: Generalization outside finance (Coupled Network & LR vs Iteration Heatmap)."""
    df_coup = pd.read_csv("experiments/coupled_network_sweep.csv")
    df_grid = pd.read_csv("experiments/lr_vs_iter_sweep.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    fig.subplots_adjust(left=0.16, right=0.90, bottom=0.18, top=0.88, wspace=0.38)
    
    # Left: Coupled network emergence vs coupling strength g
    ax1.plot(df_coup["g"], df_coup["ce_svd"], 'k:', label='SVD Upper Bound')
    ax1.plot(df_coup["g"], df_coup["ce_adaptive"], 'o-', color='#1b7837', label='Monotone backtracking')
    ax1.plot(df_coup["g"], df_coup["ce_fixed"], 's--', color='#d73027', label='Fixed 12/100')
    
    ax1.set_xlabel(r'Coupling Strength $g$')
    ax1.set_ylabel(r'Causal Emergence Density $\Delta J$')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.set_title('(a) Coupled Dynamical Network')
    
    # Right: LR vs Iteration phase diagram / contours
    piv = df_grid.pivot(index="lr", columns="iterations", values="pct_opt")
    X = piv.columns.values
    Y = piv.index.values
    Z = piv.values
    
    # Heatmap / Contour plot
    im = ax2.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', vmin=10, vmax=100)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Iterations Budget $K$')
    ax2.set_ylabel(r'Learning Rate $\eta$')
    ax2.set_title(r'(b) Fraction of Analytical Optimum Attained (%)')
    
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label(r'% of Analytical Optimum', rotation=270, labelpad=12)
    
    # Mark production configuration (lr=0.05, K=100)
    prod_pct = float(df_grid[(df_grid["lr"] == 0.05) & (df_grid["iterations"] == 100)]["pct_opt"].iloc[0])
    ax2.plot(100, 0.05, 'r*', markersize=10, markeredgecolor='white',
             label=f'Production ({prod_pct:.1f}%)')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.savefig("figures/fig4_generalization_and_grid.pdf", pad_inches=0.08)
    plt.savefig("figures/fig4_generalization_and_grid.png", dpi=300, pad_inches=0.08)
    plt.close()
    print("Generated figures/fig4_generalization_and_grid.pdf and .png")

def main():
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    print("All figures successfully created in figures/")

if __name__ == "__main__":
    main()
