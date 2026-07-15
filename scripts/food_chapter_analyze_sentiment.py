# %%
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_test

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PNG = ROOT / "png"

map_sentiment = {"positive": 1, "negative": -1}

COLORS = {"yellow": "#E6B830", "blue": "#A5C9E6", "green": "#73C0C1"}


# %%
## Define functions
def plot_boxplot(
    df: pd.DataFrame,
    tick_labels: list = ["Reflecisve", "Automatic"],
    left_query: str = "mot_refl",
    right_query: str = "mot_auto",
    COLORS: list = [COLORS["yellow"], COLORS["yellow"]],
):
    """
    Plot 3 boxplots in one line.

    Parameters:
    -----------
    df (pd.DataFrame): a dataframe in a long format.
    tick_labels (list): a list of two strings with tick labels.
    left_query (str): a string denotating the name of the variable for left boxplot.
    right_query (str): a string denotating the name of the variable for right boxplot.
    COLORS (list): a list of two strings with colors of the boxplots.

    Returns:
    plt.Figure: a figure with 3 boxplots in one line.
    """
    fig, axs = plt.subplots(figsize=(9, 4), nrows=1, ncols=3)

    for n, t in enumerate(df.groupby("country")):
        country, tmp = t
        left = tmp.query("variable == @left_query")["sentiment"].tolist()
        n_left = f"(n = {len(left)})"
        right = tmp.query("variable == @right_query")["sentiment"].tolist()
        n_right = f"(n = {len(right)})"
        tick_labels_formatted = [
            item + "\n" + n for item, n in zip(tick_labels, [n_left, n_right])
        ]
        bplot = axs[n].boxplot(
            [left, right],
            patch_artist=True,
            tick_labels=tick_labels_formatted,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=(0.75, 0.75),
        )
        for patch, color in zip(bplot["boxes"], COLORS):
            patch.set_facecolor(color)
        axs[n].set_ylim(-1.4, 1.4)
        axs[n].title.set_text(country.upper())
        for spin in axs[n].spines:
            if spin != "bottom" and spin != "left" and n == 0:
                axs[n].spines[spin].set_visible(False)
            elif n != 0 and spin != "bottom":
                axs[n].spines[spin].set_visible(False)
                axs[n].get_yaxis().set_visible(False)
    return fig


def test_median(
    df: pd.DataFrame, country_a: str = "poland", country_b: str = "uk"
) -> None:
    """Prints results of Mood Median Test

    Parameters
    ----------
    df
        a data frame with "sentiment", "variable", and country columns.
    country_a, optional
        name of the first country, by default "poland"
    country_b, optional
        name of the second country, by default "uk"
    """
    for group_var, tmp in df.groupby("variable"):
        x = tmp.query("country == @country_a")["sentiment"]
        y = tmp.query("country == @country_b")["sentiment"]
        if len(x) < 1 or len(y) < 1:
            continue
        results = median_test(x, y)
        print("======Median Test======")
        print(f"Category {group_var}")
        print(f"{country_a} median = {np.median(x)}")
        print(f"{country_b} median = {np.median(y)}")
        print(
            f"Statistics (Chi^2) = {results[0]}, p-value = {results[1]}, median = {results[2]}"
        )
        print("=======================")


# %%
## Load data
df = pd.read_csv(PROC / "food_texts.csv", index_col=None)
sentiment_lst = []
for line in open(PROC / "food_texts_sentiment.jsonl", "r"):
    tmp = json.loads(line)
    sentiment = [max(item, key=lambda x: x["score"]) for item in tmp["sentiment"]]
    sentiment = [
        map_sentiment.get(item["label"].lower(), 0) * item["score"]
        for item in sentiment
    ]
    sentiment = np.mean(sentiment)
    sentiment_lst.append({"id": tmp["id"], "sentiment": sentiment})

df_sentiment = pd.DataFrame.from_dict(sentiment_lst)
df = pd.merge(df, df_sentiment, on="id").melt(
    id_vars=["id", "body", "country", "sentiment"]
)
# %%
## MOTIVATION
df_mot = df.query("variable == 'mot_auto' or variable == 'mot_refl'").query("value > 0")

fig = plot_boxplot(df=df_mot)
fig.tight_layout()
fig.savefig(PNG / "motivation_sentiment.png", dpi=200)
# %%
## Poland vs UK
test_median(df_mot)
## Poland vs Portugal
test_median(df_mot, country_b="portugal")
## Portugal vs UK
test_median(df_mot, country_a="portugal")

# %%
## CAPABILITIES
df_cap = df.query(
    "variable == 'cap_psychological' or variable == 'cap_physical'"
).query("value > 0")

fig = plot_boxplot(
    df=df_cap,
    tick_labels=["Psychological", "Physical"],
    left_query="cap_psychological",
    right_query="cap_physical",
    COLORS=[COLORS["blue"], COLORS["blue"]],
)

fig.tight_layout()
fig.savefig(PNG / "capabilities_sentiment.png", dpi=200)
# %%
## Poland vs UK
test_median(df_cap)

# %%
## OPPORTUNITIES
df_opp = df.query("variable == 'opp_social' or variable == 'opp_physical'").query(
    "value > 0"
)

fig = plot_boxplot(
    df=df_opp,
    tick_labels=["Social", "Physical"],
    left_query="opp_social",
    right_query="opp_physical",
    COLORS=[COLORS["green"], COLORS["green"]],
)

fig.tight_layout()
fig.savefig(PNG / "opportunities_sentiment.png", dpi=200)
# %%
## Poland vs UK
test_median(df_cap)

# %%
