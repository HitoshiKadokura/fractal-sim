# fractal_sim.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import os  # ディレクトリ作成用

# Simulation parameters (based on the paper)
kappa_f = 1.15e-38
eta_fractal = 1e-6
M_pl = 1.22e19
D_f = 1.80
f_NL = 3.0
N_sim = 10000

# Generate density perturbations for the fractal field
def generate_density_perturbation(k, kappa_f, eta_fractal, D_f, f_NL, M_pl):
    """
    k: Wavenumber (Mpc^-1)
    Compute the power spectrum based on the fractal field's Lagrangian
    """
    A_s = 2.1e-9
    n_s = 0.96
    P_gaussian = A_s * (k / 0.05)**(n_s - 1)
    P_non_gaussian = f_NL * (P_gaussian**2) / M_pl
    P_fractal = P_gaussian * (1 + kappa_f * eta_fractal * (k**D_f))
    return P_fractal + P_non_gaussian

# Compute σ_8
def compute_sigma8(P_k, k):
    """
    Compute σ_8 from the power spectrum at the scale R=8 Mpc/h
    """
    R = 8.0
    W = np.exp(-0.5 * (k * R)**2)
    integrand = k**2 * P_k * W / (2 * np.pi**2)
    sigma2 = np.trapz(integrand, k)
    return np.sqrt(sigma2)

# Monte Carlo simulation
np.random.seed(42)
sigma8_samples = []
omega_m_samples = []
s8_samples = []

k = np.logspace(-3, 2, 100)

for _ in range(N_sim):
    kappa_f_noisy = kappa_f * np.random.normal(1, 0.1)
    eta_fractal_noisy = eta_fractal * np.random.normal(1, 0.1)
    f_NL_noisy = np.random.normal(f_NL, 0.5)
    P_k = generate_density_perturbation(k, kappa_f_noisy, eta_fractal_noisy, D_f, f_NL_noisy, M_pl)
    sigma8 = compute_sigma8(P_k, k)
    sigma8_samples.append(sigma8)
    omega_m = np.random.normal(0.29, 0.02)
    omega_m_samples.append(omega_m)
    s8 = sigma8 * np.sqrt(omega_m / 0.3)
    s8_samples.append(s8)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save results
results = pd.DataFrame({
    'sigma8': sigma8_samples,
    'omega_m': omega_m_samples,
    's8': s8_samples
})
results.to_csv('data/sim_results.csv', index=False)

# Output statistics
print(f"σ_8: {np.mean(sigma8_samples):.3f} ± {np.std(sigma8_samples):.3f}")
print(f"Ω_m: {np.mean(omega_m_samples):.3f} ± {np.std(omega_m_samples):.3f}")
print(f"S_8: {np.mean(s8_samples):.3f} ± {np.std(s8_samples):.3f}")