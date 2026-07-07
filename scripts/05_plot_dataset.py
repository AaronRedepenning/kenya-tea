import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = "data/tbk_dataset.csv"

SEASONAL_REGION = "all"

MONTHLY_SEASONAL_OUTPUT = "production_seasonal.png"
COMPONENTS_OUTPUT = "production_components.png"

SOURCE_TEXT = "Source: Tea Board of Kenya — https://teaboard.or.ke/"


def clean_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["value_million_kg"] = df["value"] / 1_000_000

    df = df.dropna(subset=["date", "value_million_kg"]).copy()

    # Make sure sorting is truly chronological
    df = df.sort_values(["date", "region"]).reset_index(drop=True)

    return df


def add_source_note(ax):
    ax.text(
        0.99,
        -0.18,
        SOURCE_TEXT,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        alpha=0.75,
    )


def plot_months_overlayed(df: pd.DataFrame, region: str, output_path: str):
    plot_df = df[df["region"] == region].copy()

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    plot_df["month"] = pd.Categorical(
        plot_df["month"],
        categories=month_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(["year", "month"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    for year, group in plot_df.groupby("year"):
        group = group.sort_values("month")

        ax.plot(
            group["month"],
            group["value_million_kg"],
            marker="o",
            linewidth=2.2,
            label=str(year),
            alpha=0.9 if year == plot_df["year"].max() else 0.75,
        )

    region_title = region.replace("_", " ").title()

    ax.set_title(
        f"Kenya Monthly Tea Production — {region_title} Seasonal Pattern",
        fontsize=18,
        pad=16,
    )
    ax.set_ylabel("Tea production [million kg]", fontsize=12)
    ax.set_xlabel("")

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Year", frameon=False, ncol=3)

    add_source_note(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_components_over_time(df: pd.DataFrame, output_path: str):
    plot_df = df[df["region"].isin(["all", "west_of_rift", "east_of_rift"])].copy()

    # Optional: force a clean legend order
    region_order = ["all", "west_of_rift", "east_of_rift"]
    region_labels = {
        "all": "All",
        "west_of_rift": "West of Rift",
        "east_of_rift": "East of Rift",
    }

    plot_df = plot_df.sort_values(["date", "region"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 7))

    for region in region_order:
        group = plot_df[plot_df["region"] == region].sort_values("date")

        if group.empty:
            continue

        ax.plot(
            group["date"],
            group["value_million_kg"],
            marker="o",
            linewidth=2.2,
            label=region_labels[region],
            alpha=0.9,
        )

    ax.set_title(
        "Kenya Monthly Tea Production by Region",
        fontsize=18,
        pad=16,
    )
    ax.set_ylabel("Tea production [million kg]", fontsize=12)
    ax.set_xlabel("")

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Region", frameon=False)

    add_source_note(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    df = clean_dataset(DATA_FILE)

    plot_months_overlayed(
        df=df,
        region=SEASONAL_REGION,
        output_path=MONTHLY_SEASONAL_OUTPUT,
    )

    plot_components_over_time(
        df=df,
        output_path=COMPONENTS_OUTPUT,
    )

    print(f"Saved {MONTHLY_SEASONAL_OUTPUT}")
    print(f"Saved {COMPONENTS_OUTPUT}")


if __name__ == "__main__":
    main()
