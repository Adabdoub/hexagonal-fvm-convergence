"""
================================================================================
Hexagonal Finite Volume Method - Convergence Analysis
================================================================================

Rigorous numerical validation of hexagonal finite volume methods for integration
over Lipschitz domains, as described in:

    Dabdoub, A.M. "Rigorous Error Analysis of Hexagonal Finite Volume Methods
    on Lipschitz Domains." ENS Paris-Saclay, 2026.

The method achieves O(n^{-2}) convergence with explicit constant:
    C_1 = (5 * A * M * d^2) / 24

Author: Abdullah M. Dabdoub
Affiliation: ENS Paris-Saclay, Singularity Computing
Date: June 2026
License: MIT

Repository: https://github.com/abdullah-dabdoub/hexagonal-fvm-convergence
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple, Dict, List
import warnings
import scipy.integrate as integrate

from shapely.geometry import Polygon, box
from shapely.ops import triangulate

warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

REFINEMENT_LEVELS: Tuple[int, ...] = (16, 32, 64, 128, 256)
MACHINE_PRECISION_THRESHOLD: float = 1e-12

OUTPUT_DIR = Path("results")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Matplotlib styling for publication-quality figures
matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (8.0, 6.0),
    'lines.linewidth': 2.0,
    'lines.markersize': 8.0,
})

# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def f1(x: float, y: float) -> float:
    """
    Affine/bilinear test function: f1(x,y) = 1 + x*y.

    Properties:
        - Second derivatives vanish (M = 0)
        - Exactly integrated by midpoint rule on symmetric cells
        - Integral over unit disk = pi (by symmetry)
        - Integral over unit square = 5/4
    """
    return 1.0 + x * y


def f2(x: float, y: float) -> float:
    """
    Smooth oscillatory test function: f2(x,y) = sin(pi*x) * cos(pi*y).

    Properties:
        - M = pi^2 ~ 9.87
        - Integral over unit disk = 0 (odd symmetry)
        - Tests method on non-polynomial, oscillatory integrands
    """
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def f3(x: float, y: float) -> float:
    """
    Isotropic quadratic test function: f3(x,y) = 1 + x^2 + y^2.

    Properties:
        - M = 2
        - Integral over unit disk = 3*pi/2
        - Integral over unit square = 5/3
        - Primary test for O(n^{-2}) convergence rate
    """
    return 1.0 + x**2 + y**2


def f4(x: float, y: float) -> float:
    """
    Near-singular radial test function.

    Properties:
        - M ~ 1
        - Mild singularity at (0.5, 0.5) regularized by +0.01
        - Tests robustness near singularities
    """
    return np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + 0.01)


# ==============================================================================
# DOMAIN GENERATORS
# ==============================================================================

def create_unit_disk(n_points: int = 1000) -> Tuple[Polygon, float, float, float]:
    """
    Create unit disk domain approximated by a regular n_points-gon.

    Returns:
        polygon: Shapely Polygon representing the disk
        diameter: Domain diameter d = 2
        area: Exact area A = pi
        perimeter: Exact perimeter K = 2*pi
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    poly = Polygon(np.column_stack([x, y]))
    return poly, 2.0, np.pi, 2.0 * np.pi


def create_unit_square() -> Tuple[Polygon, float, float, float]:
    """
    Create unit square domain [0,1]^2.

    Returns:
        polygon: Shapely Polygon
        diameter: d = sqrt(2)
        area: A = 1
        perimeter: K = 4
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = Polygon(coords)
    return poly, np.sqrt(2.0), 1.0, 4.0


def create_l_shaped() -> Tuple[Polygon, float, float, float]:
    """
    Create L-shaped domain: [0,1]^2 U [1,2]x[0,0.5].

    Returns:
        polygon: Shapely Polygon
        diameter: d = 2*sqrt(2)
        area: A = 1.5
        perimeter: K = 6
    """
    coords = [(0, 0), (2, 0), (2, 0.5), (1, 0.5), (1, 1), (0, 1)]
    poly = Polygon(coords)
    return poly, 2.0 * np.sqrt(2), 1.5, 6.0


# ==============================================================================
# ANALYTICAL REFERENCE INTEGRALS
# ==============================================================================

def analytical_integral_disk(f_name: str) -> float:
    """
    Compute analytical integral over the unit disk for known test functions.

    Args:
        f_name: Function identifier ('f1', 'f2', 'f3', or 'f4')

    Returns:
        Exact value of the integral over the unit disk

    Raises:
        ValueError: If function name is not recognized
    """
    if f_name == "f1":
        # Integral of (1 + xy) over disk = Area + 0 (xy is odd) = pi
        return np.pi
    elif f_name == "f2":
        # sin(pi*x)*cos(pi*y) is odd in both variables over symmetric disk
        return 0.0
    elif f_name == "f3":
        # Integral of (1 + x^2 + y^2) = pi + integral of r^2 over disk
        # = pi + 2*pi * integral_0^1 r^3 dr = pi + pi/2 = 3*pi/2
        return 3.0 * np.pi / 2.0
    elif f_name == "f4":
        # No closed form - use numerical integration
        result, _ = integrate.dblquad(
            lambda y, x: np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + 0.01),
            -1.0, 1.0,
            lambda x: -np.sqrt(max(0.0, 1.0 - x**2)),
            lambda x: np.sqrt(max(0.0, 1.0 - x**2))
        )
        return result
    else:
        raise ValueError(f"Unknown function identifier: {f_name}")


def analytical_integral_square(f_name: str) -> float:
    """
    Compute analytical integral over the unit square [0,1]^2.

    Args:
        f_name: Function identifier ('f1', 'f2', 'f3', or 'f4')

    Returns:
        Exact value of the integral over [0,1]^2
    """
    if f_name == "f1":
        # Integral of (1 + xy) from 0 to 1 in both variables
        return 5.0 / 4.0
    elif f_name == "f3":
        # Integral of (1 + x^2 + y^2) = 1 + 1/3 + 1/3 = 5/3
        return 5.0 / 3.0
    else:
        # Numerical fallback
        result, _ = integrate.dblquad(
            lambda y, x: eval(f_name)(x, y),
            0.0, 1.0, lambda x: 0.0, lambda x: 1.0
        )
        return result


def analytical_integral_lshape(f_name: str) -> float:
    """
    Compute analytical integral over the L-shaped domain.

    The L-shape is decomposed into [0,1]^2 and [1,2]x[0,0.5].

    Args:
        f_name: Function identifier

    Returns:
        Exact value of the integral over the L-shaped domain
    """
    if f_name == "f1":
        # [0,1]^2: 5/4 + [1,2]x[0,0.5]: area*1 + integral of xy
        # = 0.5 + (1/2)*(3/2)*(1/4) = 0.5 + 3/16 = 11/16
        return 5.0 / 4.0 + 11.0 / 16.0  # = 31/16 = 1.9375
    elif f_name == "f3":
        # [0,1]^2: 5/3 + [1,2]x[0,0.5]: 0.5 + (7/6)*0.5 + (1/24)
        return 5.0 / 3.0 + 41.0 / 24.0  # = 81/24 = 27/8 = 3.375
    else:
        # Numerical fallback via domain decomposition
        result1, _ = integrate.dblquad(
            lambda y, x: eval(f_name)(x, y), 0.0, 1.0, lambda x: 0.0, lambda x: 1.0
        )
        result2, _ = integrate.dblquad(
            lambda y, x: eval(f_name)(x, y), 1.0, 2.0, lambda x: 0.0, lambda x: 0.5
        )
        return result1 + result2


# ==============================================================================
# HEXAGONAL FINITE VOLUME METHOD CORE
# ==============================================================================

def create_hexagon(cx: float, cy: float, circumradius: float) -> Polygon:
    """
    Create a pointy-top regular hexagon centered at (cx, cy).

    Args:
        cx: x-coordinate of hexagon center
        cy: y-coordinate of hexagon center
        circumradius: Distance from center to each vertex

    Returns:
        Shapely Polygon representing the hexagon
    """
    vertices = []
    for k in range(6):
        angle = k * np.pi / 3.0
        vx = cx + circumradius * np.cos(angle)
        vy = cy + circumradius * np.sin(angle)
        vertices.append((vx, vy))
    return Polygon(vertices)


def fvm_integrate(domain: Polygon, d: float, n: int, func: Callable[[float, float], float]) -> float:
    """
    Standard hexagonal finite volume integration with midpoint quadrature.

    The grid uses pointy-top regular hexagons with circumradius a_n = d/n.
    Interior cells use the hexagon centroid. Boundary cut-cells use the
    centroid of the intersection polygon.

    Args:
        domain: Shapely Polygon representing the integration domain
        d: Domain diameter
        n: Refinement level (a_n = d/n)
        func: Integrand function f(x, y)

    Returns:
        Approximate integral value R_H(f, a_n)
    """
    a = d / n
    bounds = domain.bounds
    margin = a * 2.0

    x_min, y_min = bounds[0] - margin, bounds[1] - margin
    x_max, y_max = bounds[2] + margin, bounds[3] + margin

    # Lattice indices for pointy-top hexagons
    j_start = int(np.floor(x_min / (a * np.sqrt(3)))) - 1
    j_end = int(np.ceil(x_max / (a * np.sqrt(3)))) + 1
    i_start = int(np.floor(y_min / (3.0 * a / 2.0))) - 1
    i_end = int(np.ceil(y_max / (3.0 * a / 2.0))) + 1

    fvm_total = 0.0

    for j in range(j_start, j_end):
        for i in range(i_start, i_end):
            # Pointy-top hexagon center coordinates
            cx = a * np.sqrt(3) * (j + (i % 2) / 2.0)
            cy = 3.0 * a * i / 2.0

            # Fast bounding-box rejection
            if not domain.intersects(box(cx - a, cy - a, cx + a, cy + a)):
                continue

            hex_poly = create_hexagon(cx, cy, a)

            if domain.contains(hex_poly):
                # Interior cell: evaluate at hexagon centroid
                fvm_total += func(cx, cy) * hex_poly.area
            else:
                # Boundary cell: evaluate at intersection centroid
                intersect = hex_poly.intersection(domain)
                if not intersect.is_empty and intersect.area > 1e-12:
                    c = intersect.centroid
                    fvm_total += func(c.x, c.y) * intersect.area

    return fvm_total


# ==============================================================================
# THEORETICAL ANALYSIS
# ==============================================================================

def compute_theoretical_constant(A: float, M: float, d: float) -> float:
    """
    Compute the theoretical error constant C_1 = (5 * A * M * d^2) / 24.

    Args:
        A: Domain area
        M: W^{2,infty} norm of the integrand
        d: Domain diameter

    Returns:
        Theoretical error constant C_1
    """
    return (5.0 * A * M * d**2) / 24.0


def fit_convergence(ns: np.ndarray, errors: np.ndarray,
                    threshold: float = MACHINE_PRECISION_THRESHOLD):
    """
    Fit error data to the model e_n = C * n^{-p}.

    Detects cases where all errors are at machine precision (exact integration)
    and returns an appropriate label instead of a meaningless fitted order.

    Args:
        ns: Array of refinement levels
        errors: Array of corresponding absolute errors
        threshold: Threshold for machine precision detection

    Returns:
        Tuple of (p_value_or_label, C_fit, r_squared, is_exact)
    """
    errors = np.array(errors)
    ns = np.array(ns)

    # Check for exact integration (all errors at machine precision)
    if np.all(errors < threshold):
        return "Exact (machine precision)", None, None, True

    # Filter out machine-precision points for fitting
    valid_mask = errors > threshold
    n_valid = np.sum(valid_mask)

    if n_valid < 2:
        return "Insufficient data", None, None, False

    ns_valid = ns[valid_mask]
    errors_valid = errors[valid_mask]

    # Linear regression in log-log space: log(e) = log(C) - p*log(n)
    log_n = np.log(ns_valid)
    log_e = np.log(errors_valid)

    A_mat = np.vstack([np.ones(len(log_n)), log_n]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_e, rcond=None)
    intercept, slope = coeffs

    p = -slope
    C = np.exp(intercept)

    # Compute R^2 coefficient of determination
    y_pred = intercept + slope * log_n
    ss_res = np.sum((log_e - y_pred)**2)
    ss_tot = np.sum((log_e - np.mean(log_e))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return p, C, r_squared, False


# ==============================================================================
# CONVERGENCE STUDY & PLOTTING
# ==============================================================================

def run_convergence_study(domain_name: str,
                          domain_func: Callable,
                          test_func: Callable[[float, float], float],
                          f_name: str,
                          M: float,
                          func_description: str,
                          ref_integral: float) -> Dict:
    """
    Run a complete convergence study and generate publication-quality plot.

    Args:
        domain_name: Human-readable domain name
        domain_func: Function returning (polygon, diameter, area, perimeter)
        test_func: Integrand function
        f_name: Short function identifier
        M: W^{2,infty} norm bound
        func_description: Human-readable function description
        ref_integral: Exact reference integral value

    Returns:
        Dictionary containing all results and metadata
    """
    poly, diameter, A, K = domain_func()

    print(f"\n{'='*70}")
    print(f"STUDY: {domain_name} + {func_description}")
    print(f"{'='*70}")
    print(f"Domain parameters: A={A:.6f}, K={K:.6f}, d={diameter:.6f}")
    print(f"Function regularity: M={M:.6f}")

    C_theory = compute_theoretical_constant(A, M, diameter)
    print(f"Theoretical C_1 = {C_theory:.6e}")
    print(f"Reference I[f] = {ref_integral:.10f}")

    ns = np.array(REFINEMENT_LEVELS)
    errors = []
    c_obs_list = []

    for n in ns:
        fvm_val = fvm_integrate(poly, diameter, n, test_func)
        err = abs(fvm_val - ref_integral)
        err = max(err, 1e-16)  # Prevent log(0)
        errors.append(err)

        # Observed constant: C_obs = (e_n * n^2) / d^2
        c_obs = (err * n**2) / (diameter**2)
        c_obs_list.append(c_obs)

        print(f"  n={n:3d}: R_H = {fvm_val:.8f}, |error| = {err:.4e}, C_obs = {c_obs:.4e}")

    errors = np.array(errors)
    p, C_fit, r2, is_exact = fit_convergence(ns, errors)

    if is_exact:
        final_c_obs = 0.0
        ratio = "N/A (exact)"
        p_display = "Exact"
    else:
        final_c_obs = c_obs_list[-1]
        ratio = final_c_obs / C_theory if C_theory > 1e-10 else "N/A"
        p_display = f"{p:.3f}"

    print(f"\n{'─'*70}")
    print("SUMMARY")
    print(f"{'─'*70}")
    print(f"  Fitted order p: {p_display}")
    if not is_exact:
        print(f"  R^2 fit quality: {r2:.4f}")
    print(f"  Final C_obs: {final_c_obs:.6e}")
    print(f"  Theoretical C_1: {C_theory:.6e}")
    print(f"  Ratio C_obs/C_1: {ratio if isinstance(ratio, str) else f'{ratio:.2f}'}")
    print(f"{'='*70}")

    # Generate publication-quality figure
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # Observed errors
    ax.loglog(ns, errors, 'o-', linewidth=2.5, markersize=9,
             label='Observed error', color='#1f77b4', zorder=3)

    # O(n^{-2}) reference line
    valid_mask = errors > MACHINE_PRECISION_THRESHOLD
    if np.any(valid_mask):
        first_valid_idx = np.where(valid_mask)[0][0]
        e_ref = errors[first_valid_idx] * (ns[first_valid_idx] / ns)**2
    else:
        e_ref = errors[0] * (ns[0] / ns)**2

    ax.loglog(ns, e_ref, 's--', linewidth=1.5, markersize=7,
             label=r'$\mathcal{O}(n^{-2})$ reference', color='#ff7f0e', alpha=0.7, zorder=2)

    # Theoretical bound line
    if not is_exact and C_theory > 1e-10:
        e_theory = C_theory * (diameter / ns)**2
        ax.loglog(ns, e_theory, '^:', linewidth=1.5, markersize=6,
                 label=r'Theoretical bound $C_1 n^{-2}$', color='#d62728', alpha=0.6, zorder=1)

    ax.set_xlabel('Refinement level $n$', fontsize=12)
    ax.set_ylabel(r'Error $|R_H - I[f]|$', fontsize=12)

    if is_exact:
        title = f"{domain_name} + {func_description}: Exact (machine precision)"
    else:
        title = f"{domain_name} + {func_description}: $p = {p:.3f}$"
    ax.set_title(title, fontweight='bold', fontsize=13)

    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', framealpha=0.9)

    # Annotation with constant ratio
    if not is_exact and isinstance(ratio, float):
        textstr = f'$C_{{\text{{obs}}}}/C_1 = {ratio:.2f}$'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

    plt.tight_layout()

    base_name = f"conv_{domain_name.replace(' ', '_').replace('-', '')}_{f_name}"
    plt.savefig(FIG_DIR / f"{base_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / f"{base_name}.pdf", bbox_inches='tight')
    plt.close()

    return {
        "domain": domain_name,
        "function": func_description,
        "status": "Exact" if is_exact else "Convergent",
        "fitted_p": p_display,
        "r_squared": r2 if r2 is not None else "N/A",
        "C_obs": final_c_obs,
        "C_theory": C_theory,
        "C_ratio": ratio if isinstance(ratio, str) else round(ratio, 2),
        "ref_integral": ref_integral,
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run the complete convergence validation suite.

    This executes all (domain, function) combinations and generates:
        - Individual convergence plots (PNG + PDF)
        - Summary CSV file with all results
        - Console output with detailed diagnostics
    """
    print("="*70)
    print("HEXAGONAL FVM CONVERGENCE VALIDATION SUITE")
    print("="*70)
    print("Reference: Dabdoub (2026), 'Rigorous Error Analysis of Hexagonal")
    print("            Finite Volume Methods on Lipschitz Domains'")
    print("="*70)

    # Pre-compute all analytical reference integrals
    print("\n[1/3] Computing analytical reference integrals...")
    refs = {
        ("Disk", "f1"): analytical_integral_disk("f1"),
        ("Disk", "f2"): analytical_integral_disk("f2"),
        ("Disk", "f3"): analytical_integral_disk("f3"),
        ("Disk", "f4"): analytical_integral_disk("f4"),
        ("Square", "f1"): analytical_integral_square("f1"),
        ("Square", "f3"): analytical_integral_square("f3"),
        ("L-Shape", "f1"): analytical_integral_lshape("f1"),
        ("L-Shape", "f3"): analytical_integral_lshape("f3"),
    }

    for key, val in refs.items():
        print(f"  {key[0]:8s} {key[1]:3s}: {val:.10f}")

    # Define all convergence studies
    studies: List[Tuple] = [
        ("Disk", create_unit_disk, f1, "f1", 0.0, "Affine/Bilinear", refs[("Disk", "f1")]),
        ("Disk", create_unit_disk, f2, "f2", np.pi**2, "Oscillatory", refs[("Disk", "f2")]),
        ("Disk", create_unit_disk, f3, "f3", 2.0, "Quadratic", refs[("Disk", "f3")]),
        ("Disk", create_unit_disk, f4, "f4", 1.0, "Near-Singular", refs[("Disk", "f4")]),
        ("Square", create_unit_square, f1, "f1", 0.0, "Affine/Bilinear", refs[("Square", "f1")]),
        ("Square", create_unit_square, f3, "f3", 2.0, "Quadratic", refs[("Square", "f3")]),
        ("L-Shape", create_l_shaped, f1, "f1", 0.0, "Affine/Bilinear", refs[("L-Shape", "f1")]),
        ("L-Shape", create_l_shaped, f3, "f3", 2.0, "Quadratic", refs[("L-Shape", "f3")]),
    ]

    # Run all studies
    print(f"\n[2/3] Running {len(studies)} convergence studies...")
    results = []
    for domain_name, dom_func, test_func, f_name, M, description, ref_int in studies:
        result = run_convergence_study(
            domain_name, dom_func, test_func, f_name, M, description, ref_int
        )
        results.append(result)

    # Generate summary table
    print("\n[3/3] Generating summary...")
    print("\n" + "="*70)
    print("CONVERGENCE SUMMARY TABLE")
    print("="*70)

    df = pd.DataFrame(results)
    df_display = df[["domain", "function", "status", "fitted_p", "r_squared",
                       "C_obs", "C_theory", "C_ratio"]]
    print(df_display.to_string(index=False))

    # Save to CSV
    csv_path = OUTPUT_DIR / "convergence_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
