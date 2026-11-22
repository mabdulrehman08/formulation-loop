"""Clean a formulation notebook, rank new blends, and simulate lab rounds."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebook.csv"
INGREDIENTS = ("resin", "plasticizer", "filler", "modifier")
LOWER = np.array([0.25, 0.05, 0.12, 0.03])
UPPER = np.array([0.65, 0.28, 0.48, 0.18])
TARGET = "viscosity_cp"
BATCH_SIZE = 5
ALIASES = {
    "runid": "run_id",
    "sampleno": "run_id",
    "resin": "resin",
    "resn": "resin",
    "plasticizer": "plasticizer",
    "plast": "plasticizer",
    "filler": "filler",
    "filer": "filler",
    "modifier": "modifier",
    "mod": "modifier",
    "batchmass": "batch_mass",
    "viscositycp": TARGET,
    "viscocitycp": TARGET,
}
MISSING = {"", "-", "na", "n/a", "?"}
QUANTITY = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(wt\s*%|kg|g)", re.I)


def notebook_fields(row):
    fields = {}
    for header, value in row.items():
        key = ALIASES.get(re.sub(r"[^a-z0-9]", "", header.lower()))
        if key and value.strip().lower() not in MISSING:
            if key in fields:
                raise ValueError(f"multiple values for {key}")
            fields[key] = value.strip()
    return fields


def quantity(value):
    match = QUANTITY.fullmatch(value)
    if not match:
        raise ValueError(f"invalid quantity {value!r}")
    amount = float(match[1])
    if not np.isfinite(amount) or amount < 0:
        raise ValueError(f"invalid amount {value!r}")
    unit = re.sub(r"\s", "", match[2].lower())
    if unit == "wt%":
        return amount / 100, True
    return amount * (1000 if unit == "kg" else 1), False


def clean_row(fields):
    if "run_id" not in fields:
        raise ValueError("missing run ID")
    if TARGET not in fields:
        raise ValueError("missing viscosity")
    target = float(fields[TARGET])
    if not np.isfinite(target) or target <= 0:
        raise ValueError("viscosity must be finite and positive")

    fractions = np.full(len(INGREDIENTS), np.nan)
    masses = {}
    for i, name in enumerate(INGREDIENTS):
        if name in fields:
            amount, is_fraction = quantity(fields[name])
            if is_fraction:
                fractions[i] = amount
            else:
                masses[i] = amount

    missing = [i for i in np.flatnonzero(np.isnan(fractions)) if i not in masses]
    if len(missing) > 1:
        raise ValueError("more than one ingredient is missing")
    if "batch_mass" in fields:
        batch_mass, is_fraction = quantity(fields["batch_mass"])
        if is_fraction or batch_mass <= 0:
            raise ValueError("batch mass must be positive and use g or kg")
    elif masses:
        if missing:
            raise ValueError("missing ingredient requires a recorded batch mass")
        remainder = 1 - np.nansum(fractions)
        batch_mass = sum(masses.values()) / remainder if remainder > 0 else 0
        if batch_mass <= 0:
            raise ValueError("mass entries require a positive remaining fraction")
    else:
        batch_mass = None

    for i, mass in masses.items():
        fractions[i] = mass / batch_mass
    if missing:
        fractions[missing[0]] = 1 - np.nansum(fractions)
    # Tolerate floating point roundoff, and reject inconsistent notebook totals.
    if not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError("ingredient fractions do not sum to 1")
    if np.any(fractions < LOWER - 1e-9) or np.any(fractions > UPPER + 1e-9):
        raise ValueError("ingredient outside formulation bounds")
    return [fields["run_id"], *fractions, target], bool(missing)


def normalize(path):
    records, rejected, recovered = [], [], []
    run_ids = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("notebook is missing a CSV header")
        if any(not header.strip() for header in reader.fieldnames):
            raise ValueError("notebook has a blank CSV header")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("notebook has duplicate CSV headers")
        for row in reader:
            try:
                if None in row or None in row.values():
                    raise ValueError("wrong number of CSV columns")
                record, filled = clean_row(notebook_fields(row))
                run_id = record[0]
                if run_id in run_ids:
                    raise ValueError(f"duplicate run ID {run_id}")
            except ValueError as exc:
                rejected.append(f"line {reader.line_num}: {exc}")
                continue
            run_ids.add(run_id)
            records.append(record)
            if filled:
                recovered.append(run_id)
    table = pd.DataFrame(records, columns=["run_id", *INGREDIENTS, TARGET])
    return table, rejected, recovered


def sample_formulations(rng, size):
    """Sample uniformly from the bounded mixture by simplex rejection."""
    samples = []
    remaining = size
    while remaining:
        proposals = rng.dirichlet(np.ones(len(INGREDIENTS)), size=max(512, 15 * remaining))
        feasible = proposals[np.all((proposals >= LOWER) & (proposals <= UPPER), axis=1)]
        chosen = feasible[:remaining]
        samples.append(chosen)
        remaining -= len(chosen)
    return np.concatenate(samples)


def fit_surrogate(x, y):
    model = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.16, nu=2.5),
        alpha=0.015**2,
        normalize_y=True,
        optimizer=None,
    )
    return model.fit(x, y)


def expected_improvement(mean, std, best):
    gain = mean - best
    z = np.divide(gain, std, out=np.zeros_like(gain), where=std > 0)
    density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return np.where(std > 0, gain * ndtr(z) + std * density, np.maximum(gain, 0))


def propose(x, y, pool, available):
    if len(available) < BATCH_SIZE:
        raise ValueError(f"need at least {BATCH_SIZE} unseen candidates")
    candidates = pool[available]
    mean, std = fit_surrogate(x, y).predict(candidates, return_std=True)
    ei = expected_improvement(mean, std, y.max())
    ranked = np.argsort(-ei, kind="stable")
    selected = []
    for i in ranked:
        distances = np.linalg.norm(candidates[selected] - candidates[i], axis=1)
        if np.all(distances >= 0.04):
            selected.append(i)
            if len(selected) == BATCH_SIZE:
                break
    if len(selected) < BATCH_SIZE:
        remaining = [i for i in ranked if i not in selected]
        selected.extend(remaining[: BATCH_SIZE - len(selected)])
    return available[selected], mean[selected], std[selected], ei[selected]


def unseen(pool, x):
    observed = np.isclose(pool[:, None, :], x[None, :, :], rtol=0, atol=1e-9).all(axis=2)
    return np.flatnonzero(~observed.any(axis=1))


def synthetic_viscosity(x):
    """Deterministic toy response in cP, with two Gaussian peaks."""
    main = np.sum((x - [0.33, 0.16, 0.37, 0.14]) ** 2, axis=1)
    secondary = np.sum((x - [0.56, 0.22, 0.16, 0.06]) ** 2, axis=1)
    return 330 + 110 * x[:, 2] + 290 * np.exp(-main / 0.020) + 215 * np.exp(-secondary / 0.012)


def generate_notebook(path):
    x = sample_formulations(np.random.default_rng(21), 21)
    x = np.round(x, 4)
    x[:, -1] = 1 - x[:, :-1].sum(axis=1)
    y = synthetic_viscosity(x)
    with path.open("w", encoding="ascii", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([" Sample No ", " Resin ", "RESN", "PLAST", "filer",
                         "MOD", "Batch Mass", "Viscocity (cP)", "Notes"])
        for i, (blend, value) in enumerate(zip(x, y)):
            grams = 1000 * blend
            amounts = [f"{mass:.1f} g" for mass in grams]
            total, note = "", ""
            if i % 3 == 1:
                amounts = [f"{blend[0]:.4f} KG", f"{100 * blend[1]:.2f} wt%",
                           f"{grams[2]:.1f}g", f"{100 * blend[3]:.2f} WT %"]
            elif i % 3 == 2:
                amounts = [f"{100 * fraction:.2f} wt%" for fraction in blend]
            if i == len(x) - 1:
                amounts = [f"{grams[0]:.1f} g", "N/A", f"{grams[2]:.1f}g", f"{blend[3]:.4f} kg"]
                total, note = "1 kg", "plasticizer left blank"
            resin, rest = amounts[0], amounts[1:]
            resin_cells = [resin, "-"] if i % 2 == 0 else ["", resin]
            writer.writerow([f"S{i + 1:02d}", *resin_cells, *rest, total, f"{value:.2f}", note])
        writer.writerow(["S22", "350 mL", "", "150 g", "380 g", "120 g", "", "527", "bad unit"])
        writer.writerow(["S23", "35 wt%", "", "15 wt%", "38 wt%", "12 wt%", "", "?", "no result"])
        writer.writerow(["S24", "350 g", "", "150 g", "NA", "120 g", "", "527", "no batch mass"])
    print(f"Wrote 24 synthetic notebook rows to {path}")


def arrays(table):
    if len(table) < 2:
        raise ValueError("need at least two accepted observations")
    return table[list(INGREDIENTS)].to_numpy(), table[TARGET].to_numpy()


def run_notebook(path, clean_path):
    table, rejected, recovered = normalize(path)
    x, y = arrays(table)
    table.to_csv(clean_path, index=False, float_format="%.6f", lineterminator="\n")
    print(f"Accepted {len(table)} rows. Rejected {len(rejected)} rows.")
    if recovered:
        print("Recovered one missing ingredient in each of: " + ", ".join(recovered))
    for message in rejected:
        print(f"  Rejected {message}")
    print(f"Wrote clean table to {clean_path}")
    print(f"Best observed viscosity: {y.max():.2f} cP")
    pool = sample_formulations(np.random.default_rng(31), 4096)
    selected, mean, std, ei = propose(x, y, pool, unseen(pool, x))
    print("Next five experiments (mass %, predictions in cP):")
    print(" #    resin  plasticizer   filler  modifier   mean +/- std       EI")
    for rank, (blend, prediction, uncertainty, improvement) in enumerate(
        zip(pool[selected], mean, std, ei), start=1
    ):
        resin, plasticizer, filler, modifier = 100 * blend
        print(f"{rank:2d}  {resin:7.3f}  {plasticizer:11.3f}  {filler:7.3f}  {modifier:8.3f}"
              f"  {prediction:6.2f} +/- {uncertainty:5.2f}  {improvement:7.2f}")


def simulate(path, plot_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start_x, start_y = arrays(normalize(path)[0])
    if not np.allclose(start_y, synthetic_viscosity(start_x), rtol=0, atol=0.0051):
        raise ValueError("simulation requires notebook targets from the synthetic generator")
    rounds, trials = 8, 10
    ei_histories, random_histories = [], []
    for trial in range(trials):
        pool_seed, random_seed = np.random.SeedSequence(2400 + trial).spawn(2)
        pool = sample_formulations(np.random.default_rng(pool_seed), 2048)
        eligible = unseen(pool, start_x)
        random_picks = np.random.default_rng(random_seed).permutation(eligible)[: rounds * BATCH_SIZE]
        observed_x, observed_y = start_x.copy(), start_y.copy()
        ei_best = [start_y.max()]
        random_y = synthetic_viscosity(pool[random_picks]).reshape(rounds, BATCH_SIZE)
        random_best = np.maximum.accumulate(np.r_[start_y.max(), random_y.max(axis=1)])
        for _ in range(rounds):
            picked, _, _, _ = propose(observed_x, observed_y, pool, eligible)
            observed_x = np.vstack([observed_x, pool[picked]])
            observed_y = np.r_[observed_y, synthetic_viscosity(pool[picked])]
            eligible = eligible[~np.isin(eligible, picked)]
            ei_best.append(observed_y.max())
        ei_histories.append(ei_best)
        random_histories.append(random_best)

    ei_histories, random_histories = np.array(ei_histories), np.array(random_histories)
    ei_median = np.median(ei_histories, axis=0)
    random_median = np.median(random_histories, axis=0)
    wins = np.count_nonzero(ei_histories[:, -1] > random_histories[:, -1])
    print(f"Synthetic closed loop: {trials} paired trials, {rounds} rounds, {BATCH_SIZE} experiments per round.")
    print("Median best viscosity (cP)")
    print("Round   Expected improvement   Random search")
    for i, (ei_best, random_best) in enumerate(zip(ei_median, random_median)):
        print(f"{i:5d}   {ei_best:20.2f}   {random_best:13.2f}")
    print(f"Expected improvement finished ahead in {wins}/{trials} trials.")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for histories, median, label, color in (
        (ei_histories, ei_median, "Expected improvement", "#096c65"),
        (random_histories, random_median, "Random search", "#b45729"),
    ):
        low, high = np.percentile(histories, [25, 75], axis=0)
        ax.fill_between(range(rounds + 1), low, high, color=color, alpha=0.14)
        ax.plot(range(rounds + 1), median, color=color, marker="o", linewidth=2, label=label)
    ax.set(title="Best synthetic viscosity over eight rounds", xlabel="Round (5 experiments per round)",
           ylabel="Best observed viscosity (cP)", xticks=range(rounds + 1))
    fig.supxlabel("Median and middle 50% across 10 paired trials", fontsize=9, color="#4b5258")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="regenerate the synthetic notebook")
    generate.add_argument("--output", type=Path, default=Path(NOTEBOOK.name))
    run = commands.add_parser("run", help="clean a notebook and propose experiments")
    run.add_argument("notebook", type=Path, nargs="?", default=NOTEBOOK)
    run.add_argument("--clean", type=Path, default=Path("clean_formulations.csv"))
    simulation = commands.add_parser("simulate", help="compare closed-loop search with random search")
    simulation.add_argument("--notebook", type=Path, default=NOTEBOOK)
    simulation.add_argument("--plot", type=Path, default=Path("closed_loop.png"))
    args = parser.parse_args()
    try:
        if args.command == "generate":
            generate_notebook(args.output)
        elif args.command == "run":
            run_notebook(args.notebook, args.clean)
        else:
            simulate(args.notebook, args.plot)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
