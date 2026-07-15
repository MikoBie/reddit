# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PNG = ROOT / "png"
COLORS = ["#A5C9E6", "#73C0C1", "#E6B830", "white"]

# %%
## Load data
df = pd.read_csv(PROC / "food_texts.csv", index_col=None).assign(
    cap=lambda x: x["cap_psychological"] + x["cap_physical"],
    mot=lambda x: x["mot_refl"] + x["mot_auto"],
    opp=lambda x: x["opp_social"] + x["opp_physical"],
)

df[["cap", "mot", "opp"]] = df.loc[:, "cap":"opp"].map(lambda x: {2: 1}.get(x, x))

total = df.groupby("country")["body"].count().to_dict()
freq = df.groupby(["country"])[["cap", "mot", "opp"]].sum().transpose()
empty = (
    df.query("cap == 0 and mot == 0 and opp == 0")
    .groupby("country")["body"]
    .count()
    .reset_index()
    .rename(columns={"body": "empty"})
    .set_index("country")
    .transpose()
)
freq = pd.concat([freq, empty])

freq["poland"] = freq["poland"].apply(lambda x: 100 * x / total["poland"])
freq["portugal"] = freq["portugal"].apply(lambda x: 100 * x / total["portugal"])
freq["uk"] = freq["uk"].apply(lambda x: 100 * x / total["uk"])

freq = freq.sort_values("poland", ascending=True).rename(
    {
        "cap": "Capabilities",
        "opp": "Opportunities",
        "mot": "Motivations",
        "empty": "Empty",
    }
)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
tick_labels_fmt = [
    item.capitalize() + "\n" + f"(n = {total[item]})"
    if item != "uk"
    else "United Kingodm" + "\n" + f"(n = {total[item]})"
    for item in freq.columns.tolist()
]
res = ax.grouped_bar(
    freq.transpose(), tick_labels=tick_labels_fmt, colors=COLORS, edgecolor="black"
)
for container in res.bar_containers:
    ax.bar_label(container, fmt=lambda x: f"{int(round(x,0))}%")
ax.set_ylim(0, 100)
plt.legend(ncols=4, loc="upper center")
for spine in ax.spines:
    if spine != "bottom":
        ax.spines[spine].set_visible(False)
ax.yaxis.set_visible(False)

fig.tight_layout()
fig.savefig(PNG / "frequencies.png", dpi=200)

# %%
