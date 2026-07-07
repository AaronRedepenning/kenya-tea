import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/tbk_dataset.csv")

df = df[df["region"] == "east_of_rift"].copy()

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

df["value_million_kg"] = df["value"] / 1_000_000

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

df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
df = df.sort_values(["year", "month"])

fig, ax = plt.subplots(figsize=(12, 7))

for year, group in df.groupby("year"):
    ax.plot(
        group["month"],
        group["value_million_kg"],
        marker="o",
        linewidth=2.2,
        label=str(year),
        alpha=0.9 if year == df["year"].max() else 0.75,
    )

ax.set_title("Kenya Monthly Tea Production (Seasonal)", fontsize=18, pad=16)
ax.set_ylabel("tea_production [million kg]", fontsize=12)
ax.set_xlabel("")

ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", rotation=35)
ax.legend(title="Year", frameon=False, ncol=3)

ax.text(
    0.99,
    -0.18,
    "Source: Tea Board of Kenya — https://teaboard.or.ke/",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    alpha=0.75,
)

plt.tight_layout()
# plt.show()

plt.savefig("production.png")
