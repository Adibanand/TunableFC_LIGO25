import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


# Transmission Models
def simulate_transmission(R1, R2, L1, L2, R3=0.999, wavelength=1064e-9):
    k = 2 * np.pi / wavelength
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)
    t1, t2, t3 = np.sqrt(1 - R1), np.sqrt(1 - R2), np.sqrt(1 - R3)

    phi = k * (L1 + L2)
    denom = (
        np.exp(2j * phi)
        - r1 * r2 * np.exp(2j * k * L2)
        - r2 * r3 * np.exp(2j * k * L1)
        + r1 * r3 * (r2**2 + t2**2)
    )
    t_total = (-t1 * t2 * t3 * np.exp(1j * phi)) / denom
    return np.abs(t_total) ** 2

def T_three_mirror(k, L1, L2, r1, r2, r3, t1, t2, t3):
    e_total = np.exp(2j * k * (L1 + L2))
    e_L2 = np.exp(2j * k * L2)
    e_L1 = np.exp(2j * k * L1)

    num = -t1 * t2 * t3 * np.exp(1j * k * (L1 + L2))
    denom = (e_total - r1 * r2 * e_L2 - r2 * r3 * e_L1 + r1 * r3 * (r2**2 + t2**2))
    return np.abs(num / denom) ** 2

def T_fp(k, L, R1, R2, T1, T2):
    delta = 2 * k * L
    return (T1 * T2) / (1 + R1 * R2 - 2 * np.sqrt(R1 * R2) * np.cos(delta))


# Plotting
def plot_transmission_spectrum(R1, R2, L1, L2, save_dir, lambda0=1064e-9, R3=0.999):
    
    # Optical constants
    c = 3e8  # m/s
    nu0 = c / lambda0  # Hz

    # Detuning and wavenumber
    dnu = np.linspace(-150e6, 150e6, 1000)  # Hz
    nu = nu0 + dnu
    k = 2 * np.pi * nu / c

    t1, t2, t3 = np.sqrt(1 - R1), np.sqrt(1 - R2), np.sqrt(1 - R3)
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)

    # Cavity lengths
    L_fp = L1  # for comparison

    T3 = [T_three_mirror(ki, L1, L2, r1, r2, r3, t1, t2, t3) for ki in k]
    TFP = [T_fp(ki, L_fp, R1, R2, 1 - R1, 1 - R2) for ki in k]

    plt.figure(figsize=(10, 6))
    plt.semilogy(dnu * 1e-6, T3, label=f'Three-mirror: $L_1={L1:.3f}$ m, $L_2={L2:.3f}$ m')
    plt.semilogy(dnu * 1e-6, TFP, '--', label=f'FP cavity: $L={L_fp}$ m')

    plt.xlabel("Frequency detuning $\\Delta\\nu$ [MHz]")
    plt.ylabel("Transmitted Power [arb. units]")
    plt.title("Transmission vs Frequency Detuning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Filename with parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detuning_L1_{L1:.3f}_L2_{L2:.3f}_R1_{R1:.3f}_R2_{R2:.3f}_{timestamp}.png"
    save_dir = "detuning_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path)

    print(f"Saved detuning plot to: {save_path}")


# Objective Function
def objective(trial):
    R1 = trial.suggest_float("R1", 0.87, 0.999)
    R2 = trial.suggest_float("R2", 0.87, 0.999)
    L1 = trial.suggest_float("L1", 0.5, 3.0)
    L2 = trial.suggest_float("L2", 0.5, 3.0)

    T = simulate_transmission(R1, R2, L1, L2)
    ppm = T * 1e6
    return (ppm - 550.0) ** 2


def main(n_trials=500, n_best=10, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_trials = sorted(study.trials, key=lambda t: t.value)[:n_best]
    records = []
    for t in best_trials:
        params = t.params
        T = simulate_transmission(**params)
        ppm = T * 1e6
        records.append({**params, "Transmission_ppm": ppm, "Loss": t.value})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(save_dir, "best_parameters.csv"), index=False)
    print(f"\nSaved top {n_best} results to: {save_dir}/best_parameters.csv")

    # Trial diagnostics
    transmissions = [simulate_transmission(**t.params) * 1e6 for t in study.trials]
    plt.figure()
    plt.plot(range(len(transmissions)), transmissions, ".", alpha=0.6)
    plt.axhline(550, color="red", linestyle="--", label="Target: 550 ppm")
    plt.xlabel("Trial Index")
    plt.ylabel("Transmission (ppm)")
    plt.title("Transmission vs Trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "transmission_vs_trial.png"))
    plt.close()

    # Parameter vs transmission plots
    for param in ["R1", "R2", "L1", "L2"]:
        values = [t.params[param] for t in study.trials]
        plt.figure()
        plt.scatter(values, transmissions, alpha=0.6, s=10)
        plt.xlabel(param)
        plt.ylabel("Transmission (ppm)")
        plt.title(f"{param} vs Transmission")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{param}_vs_transmission.png"))
        plt.close()

    # Spectrum plots with filenames
    for idx, row in df.iterrows():
        plot_transmission_spectrum(row["R1"], row["R2"], row["L1"], row["L2"], save_dir)

    print(f"Saved transmission spectra with timestamps and parameters.")

if __name__ == "__main__":
    main()




