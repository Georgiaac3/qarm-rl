import csv
from pathlib import Path

import matplotlib.pyplot as plt


def find_default_csv() -> Path:
    latest_csv = Path("plot_data") / "live_plot_latest.csv"
    if latest_csv.exists():
        return latest_csv

    archived = sorted(Path("plot_data").glob("live_plot_*.csv"))
    if archived:
        return archived[-1]

    raise FileNotFoundError("No saved plot file found in plot_data/")


def replay_saved_plot(csv_path: Path):
    times = []
    x_vel = []
    y_vel = []
    z_vel = []
    base_vel = []
    shoulder_vel = []
    elbow_vel = []
    wrist_vel = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        has_cartesian = {"x_vel", "y_vel", "z_vel"}.issubset(fields)
        has_joint = {"base_vel", "shoulder_vel", "elbow_vel", "wrist_vel"}.issubset(fields)

        if not has_cartesian and not has_joint:
            raise ValueError(f"Unsupported CSV format in {csv_path}")

        for row in reader:
            times.append(float(row["time_s"]))
            if has_cartesian:
                x_vel.append(float(row["x_vel"]))
                y_vel.append(float(row["y_vel"]))
                z_vel.append(float(row["z_vel"]))
            if has_joint:
                base_vel.append(float(row["base_vel"]))
                shoulder_vel.append(float(row["shoulder_vel"]))
                elbow_vel.append(float(row["elbow_vel"]))
                wrist_vel.append(float(row["wrist_vel"]))

    if not times:
        raise ValueError(f"No data found in {csv_path}")

    if has_cartesian and has_joint:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(times, x_vel, label="x_vel")
        axes[0].plot(times, y_vel, label="y_vel")
        axes[0].plot(times, z_vel, label="z_vel")
        axes[0].set_title("Saved End-Effector Velocity")
        axes[0].set_ylabel("Velocity (m/s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(times, base_vel, label="base_vel")
        axes[1].plot(times, shoulder_vel, label="shoulder_vel")
        axes[1].plot(times, elbow_vel, label="elbow_vel")
        axes[1].plot(times, wrist_vel, label="wrist_vel")
        axes[1].set_title("Saved Joint Velocities")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Angular velocity (rad/s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
    elif has_cartesian:
        plt.figure(figsize=(10, 5))
        plt.plot(times, x_vel, label="x_vel")
        plt.plot(times, y_vel, label="y_vel")
        plt.plot(times, z_vel, label="z_vel")
        plt.title("Saved End-Effector Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(times, base_vel, label="base_vel")
        plt.plot(times, shoulder_vel, label="shoulder_vel")
        plt.plot(times, elbow_vel, label="elbow_vel")
        plt.plot(times, wrist_vel, label="wrist_vel")
        plt.title("Saved Joint Velocities")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular velocity (rad/s)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    plt.show()


def main():
    csv_path = find_default_csv()
    print(f"Relecture du plot depuis: {csv_path}")

    replay_saved_plot(csv_path)


if __name__ == "__main__":
    main()
