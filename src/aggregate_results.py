
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="./eval_out")
    parser.add_argument("--out", type=str, default="./eval_out/summary.png")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.eval_dir, "metrics_*.csv")))
    if not files:
        raise SystemExit("No metrics_*.csv found in eval_dir")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        tag = os.path.basename(f).replace("metrics_", "").replace(".csv","")
        rows.append({
            "tag": tag,
            "mean_energy_per_m": df["energy_per_m"].mean(),
            "fall_rate": df["fallen"].mean(),
            "time_limit_rate": df["truncated"].mean(),
            "mean_distance": df["distance"].mean(),
        })

    out_df = pd.DataFrame(rows).sort_values("tag")
    print(out_df)

    
    plt.figure()
    plt.bar(out_df["tag"], out_df["mean_energy_per_m"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("mean energy_per_m (proxy)")
    plt.title("Energy efficiency (lower is better)")
    plt.tight_layout()
    plt.savefig(args.out.replace(".png","_energy.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(out_df["tag"], out_df["fall_rate"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("fall rate (terminated fraction)")
    plt.title("Stability (lower is better)")
    plt.tight_layout()
    plt.savefig(args.out.replace(".png","_falls.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(out_df["tag"], out_df["mean_distance"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("mean distance")
    plt.title("Distance per episode")
    plt.tight_layout()
    plt.savefig(args.out.replace(".png","_dist.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
