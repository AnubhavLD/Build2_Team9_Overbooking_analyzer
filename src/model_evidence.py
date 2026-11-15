"""Regenerate the model comparison chart, architecture and Choose snapshot diagnostic."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

if __package__:
    from . import backtest, contract
else:
    import backtest
    import contract

LABELS = {"flat": "Client flat", "lookup": "Historical lookup",
          "logistic": "Logistic regression", "forest": "Random forest"}


def buffer_chart(grid, comparison, output):
    fig, (cost, walks) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    # Show the practical range around the optima. The CSV retains the full grid.
    limit = int(comparison["buffer"].max()) + 8
    for (method, rows), color in zip(grid.groupby("method", sort=False),
                                     ("#485E73", "#006D77", "#9C4800", "#704693")):
        rows = rows[rows.buffer.le(limit)]
        best = comparison.loc[comparison.method.eq(method)].iloc[0]
        cost.plot(rows.buffer, rows.cost_eur / 1000, label=LABELS[method], color=color)
        walks.plot(rows.buffer, rows.walks_per_1000, color=color)
        cost.scatter(best.buffer, best.cost_eur / 1000, color=color, s=28, zorder=3)
        walks.scatter(best.buffer, best.walks_per_1000, color=color, s=28, zorder=3)
    walks.axhline(contract.WALK_CEILING, color="#333333", linestyle="--", linewidth=1,
                  label="Client ceiling")
    cost.set_title("Choose-period loss and walking rate by buffer", loc="left", weight="bold")
    cost.set_ylabel("Loss (€ thousands)")
    walks.set_ylabel("Walks per 1,000 arrivals")
    walks.set_xlabel("Safety buffer (rooms)")
    cost.legend(ncols=2, frameon=False)
    walks.legend(frameon=False)
    for ax in (cost, walks):
        ax.grid(axis="y", alpha=.22)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def architecture(output):
    """Draw the manager's view and the actual Python execution path."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set(xlim=(0, 13), ylim=(0, 8))
    ax.axis("off")

    def box(x, y, title, detail, color, width):
        patch = FancyBboxPatch(
            (x, y), width, .86, boxstyle="round,pad=.045,rounding_size=.08",
            facecolor=color, edgecolor="#7692a2",
        )
        ax.add_patch(patch)
        ax.text(x + .13, y + .59, title, fontsize=10, weight="bold", color="#17354a")
        ax.text(x + .13, y + .17, detail, fontsize=8.5, color="#344c5c", linespacing=1.28)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#557588", "lw": 1.4})

    ax.text(.2, 7.67, "Hotel overbooking recommendation system", fontsize=21,
            weight="bold", color="#17354a")
    ax.text(.2, 7.15, "A. What the revenue manager sees", fontsize=13,
            weight="bold", color="#17354a")
    boxes = [
        ("Booking export", "Arrival fields + booking-time\nfeatures"),
        ("Recommendation system", "Reads a saved model and\nroom buffer"),
        ("14-night CSV", "Bookings, expected cancellations,\nbuffer and extra rooms"),
        ("Revenue manager", "Reviews and decides how\nmany extra rooms to sell"),
    ]
    for index, (title, detail) in enumerate(boxes):
        x = .22 + 3.22 * index
        box(x, 5.97, title, detail, "#e4f0f2", 2.83)
        if index:
            arrow((x - .32, 6.39), (x - .07, 6.39))

    ax.text(.2, 5.39, "B. How the Python code makes the CSV", fontsize=13,
            weight="bold", color="#17354a")
    steps = [
        ("Original bookings", "data.py · read_raw()/download()"),
        ("Validated archive or handover", "clean_data.py · build_master()\nProvisional Bin 1 CSV is supplied"),
        ("Booking-time predictors", "features.py · predictors()\nDates route rows to arrival nights"),
        ("Saved model and its inputs", "backtest.py · load_bundle()\nReads fitted preparation and model"),
        ("Chance for every booking", "modeling.py · predict()"),
        ("One total per hotel-night", "nightly.py · nightly_recommendations()\nCounts bookings and sums chances"),
        ("Frozen safety buffer", "Read buffers[selected] from the bundle"),
        ("Whole rooms to oversell", "modeling.py · recommend()\nmax(0, floor(expected − buffer))"),
        ("Recommendation CSV", "nightly.py · main()"),
    ]
    xs = (.22, 4.52, 8.82)
    positions = [(x, 4.19) for x in xs] + [(x, 2.78) for x in reversed(xs)]
    positions += [(x, 1.37) for x in xs]
    width = 3.9
    for (title, detail), (x, y) in zip(steps, positions):
        box(x, y, title, detail, "#edf4f8", width)
    for (x, y), (nx, ny) in zip(positions, positions[1:]):
        if y != ny:
            arrow((x + width / 2, y - .07), (nx + width / 2, ny + .93))
        elif x < nx:
            arrow((x + width + .07, y + .43), (nx - .07, ny + .43))
        else:
            arrow((x - .07, y + .43), (nx + width + .07, ny + .43))

    ax.text(.22, .71, "Historical scoring runs separately in backtest.py and contract.py.",
            fontsize=10, color="#17354a", weight="bold")
    ax.text(.22, .36, "The classroom demo uses the final booking list. The nightly command does not read actual cancellations.",
            fontsize=9.5, color="#344c5c")
    fig.savefig(output, dpi=165, bbox_inches="tight", pad_inches=.23)
    plt.close(fig)


def main():
    output = backtest.EVIDENCE
    grid = pd.read_csv(output / "choose_buffer_grid.csv")
    comparison = pd.read_csv(output / "choose_comparison.csv")
    buffer_chart(grid, comparison, output / "choose_buffer_tradeoff.png")
    architecture(backtest.ROOT / "docs" / "architecture.png")

    needed = {"hotel", "distribution_channel", "is_canceled", "lead_time", "arrival_date"}
    rows = backtest.read_bookings(backtest.INPUT, usecols=needed.__contains__)
    train, choose, _ = contract.split_periods(backtest.prepare_dates(rows))
    lookup = train.groupby(["hotel", "distribution_channel"])["is_canceled"].agg(
        bookings="size", cancellations="sum", raw_rate="mean",
    )
    lookup.to_csv(output / "train_lookup_segments.csv", float_format="%.10f")
    snapshot = choose.assign(known_by_day_14=choose["lead_time"].ge(14)).groupby("hotel").agg(
        final_bookings=("lead_time", "size"),
        estimated_booked_by_day_14=("known_by_day_14", "sum"),
    )
    snapshot["estimated_share"] = snapshot.estimated_booked_by_day_14 / snapshot.final_bookings
    snapshot.to_csv(output / "choose_day14_snapshot_proxy.csv", float_format="%.10f")


if __name__ == "__main__":
    main()

