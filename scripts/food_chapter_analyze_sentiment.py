# %%
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import median_test, permutation_test

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PNG = ROOT / "png"

map_sentiment = {"positive": 1, "negative": -1}

COLORS = {"yellow": "#E6B830", "blue": "#A5C9E6", "green": "#73C0C1"}
font = {"size": 10}

rc("font", **font)
rng = np.random.default_rng(8710)


# %%
## Define functions
def plot_boxplot(
    df: pd.DataFrame,
    color: str = COLORS["yellow"],
    title: str = "Motivations",
    sig: bool = False,
    sig_level: list = ["***", "***"],
) -> plt.Figure:
    """
    Plots boxplots for countries. Optionally, it may show a significance level.

    Parameters:
    -----------
    df (pd.DataFrame): a dataframe in a long format.
    color (str): a list of two strings with colors of the boxplots.
    title (str): the title of the graph.

    Returns:
    plt.Figure: a figure with 3 boxplots in one line.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)

    for n, t in enumerate(df.groupby("country")):
        country, tmp = t
        ax.boxplot(
            x=tmp["sentiment"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": color},
            positions=[n],
            tick_labels=[country.upper() + "\n" + f"(n = {len(tmp)})"],
        )

    ax.set_ylim(-1.4, 1.4)
    ax.title.set_text(title)
    ax.set_ylabel("Sentiment")
    for spin in ax.spines:
        if spin != "bottom" and spin != "left":
            ax.spines[spin].set_visible(False)

    if sig:
        ax.plot([0, 1], [1.1, 1.1], linestyle=":", color="black")
        ax.text(0.5, 1.12, sig_level[0], horizontalalignment="center", fontsize=12)
        ax.plot([1, 2], [1.2, 1.2], linestyle=":", color="black")
        ax.text(1.5, 1.22, sig_level[1], horizontalalignment="center", fontsize=12)
    return fig


def test_median(df: pd.DataFrame) -> None:
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
    dct = {}
    print("======Median Test======")
    for _, tmp in df.groupby("country"):
        if len(tmp) < 10:
            continue
        dct[_] = tmp["sentiment"].tolist()
        print(f"{_} median = {np.median(tmp["sentiment"].tolist())}")

    results = median_test(*dct.values())
    print(
        f"Statistics (Chi^2) = {results[0]}, p-value = {results[1]}, median = {results[2]}"
    )
    print("=======================")


def run_pairwise_median_permutation(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
    n_resamples: int = 10000,
    alpha_adj: float = 0.0167,
    rng=rng,
):
    """
    Runs a non-parametric permutation test to compare the medians of two distance groups.

    Parameters
    ----------
    df
        _description_
    label_a
        _description_
    label_b
        _description_
    n_resamples, optional
        _description_, by default 10000
    alpha_adj, optional
        _description_, by default 0.0167

    Returns
    -------
        _description_
    """
    group_a = df.query("country == @label_a")["sentiment"].tolist()
    group_b = df.query("country == @label_b")["sentiment"].tolist()
    # Run the exact or randomized permutation test
    res = permutation_test(
        (group_a, group_b),
        statistic=lambda x, y: np.abs(np.median(x) - np.median(y)),
        permutation_type="independent",
        n_resamples=n_resamples,
        random_state=rng,
    )

    obs_diff = np.abs(np.median(group_a) - np.median(group_b))
    p_val = res.pvalue
    status = "SIGNIFICANT" if p_val < alpha_adj else "NOT SIGNIFICANT"
    print("======Post Hoc======")
    print(f"{label_a} vs {label_b}:")
    print(f"  Observed |Mdn_A - Mdn_B| = {obs_diff}")
    print(f"  p-value = {p_val} ({status})")
    print("=====================")


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
df = pd.merge(df, df_sentiment, on="id").drop_duplicates("id")
# %%
## MOTIVATION
df_mot = df.query("mot_refl > 0 | mot_auto > 0")

fig = plot_boxplot(df=df_mot, sig=True)
fig.tight_layout()
fig.savefig(PNG / "motivation_sentiment.png", dpi=200)
# %%
## Omnibus test Motivation
test_median(df_mot)
run_pairwise_median_permutation(df=df_mot, label_a="poland", label_b="uk")
run_pairwise_median_permutation(df=df_mot, label_a="poland", label_b="portugal")
run_pairwise_median_permutation(df=df_mot, label_a="uk", label_b="portugal")

# %%
## CAPABILITIES
df_cap = df.query("cap_psychological > 0 | cap_physical > 0")

fig = plot_boxplot(df=df_cap, color=COLORS["blue"], title="Capabilities")

fig.tight_layout()
fig.savefig(PNG / "capabilities_sentiment.png", dpi=200)
# %%
## Omnibus test for Psychological Capabilities
test_median(df_cap)

# %%
## OPPORTUNITIES
df_opp = df.query("opp_physical > 0 | opp_social > 0")

fig = plot_boxplot(
    df=df_opp,
    color=COLORS["green"],
    title="Opportunities",
    sig=True,
    sig_level=["**", "*"],
)

fig.tight_layout()
fig.savefig(PNG / "opportunities_sentiment.png", dpi=200)
# %%
## Omnibus test for Social Opportunities
test_median(df_opp)
run_pairwise_median_permutation(df=df_opp, label_a="poland", label_b="uk")
run_pairwise_median_permutation(df=df_opp, label_a="poland", label_b="portugal")
run_pairwise_median_permutation(df=df_opp, label_a="uk", label_b="portugal")
# %%
