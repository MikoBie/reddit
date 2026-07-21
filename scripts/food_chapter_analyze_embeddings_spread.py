# %%
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import levene

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PNG = ROOT / "png"
COLORS = {"yellow": "#E6B830", "blue": "#A5C9E6", "green": "#73C0C1"}
font = {"size": 10}

rc("font", **font)
rng = np.random.default_rng(8710)


# %%
## Define functions
def levene_test(df: pd.DataFrame, countries: list = ["poland", "uk"]):
    """Computes levene test of variance

    Parameters
    ----------
    df
        a data frame with columns counry and distances
    countries, optional
        a list with name of countries to compare, by default ["poland", "uk"]
    """
    dct = {}
    print("=====Levene Test=====")
    for country in countries:
        tmp = df.query("country == @country")["distances"].tolist()
        dct[country] = tmp
        print(f"{country} standard deviation = {np.std(tmp)}")
        print(f"{country} median = {np.median(tmp)}")
    stat, pvalue = levene(*dct.values(), center="trimmed")
    print(f"Test Value = {stat}")
    print(f"p-value = {pvalue}")


def plot_boxplot(
    df: pd.DataFrame,
    color: str = COLORS["yellow"],
    title: str = "",
    sig: bool = False,
    sig_level: str = "*",
) -> plt.Figure:
    """Plots boxplots for countries. Optionally, it may show a significance level between Poland and Portugal.

    Parameters
    ----------
    df
        a data frame with country and distances columns.
    color, optional
        color of the boxplots, by default COLORS["yellow"]
    sig, optional
        True if the significance level should be plotted, by default False
    sig_level, optional
        level of the significance (* -- 0.05, ** -- < 0.01), by default "*"

    Returns
    -------
        Boxplots for countries.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    for n, t in enumerate(df.groupby("country")):
        country, tmp = t
        ax.boxplot(
            x=tmp["distances"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": color},
            positions=[n],
            tick_labels=[country.upper() + "\n" + f"(n = {len(tmp)})"],
        )

    for spin in ax.spines:
        if spin != "bottom" and spin != "left":
            ax.spines[spin].set_visible(False)

    if sig:
        ax.plot([0, 2], [0.4, 0.4], linestyle=":", color="black")
        ax.text(1, 0.42, sig_level, horizontalalignment="center", fontsize=12)
    ax.title.set_text(title)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Semantic spread\n(cosine similarity)")
    return fig


def permutation_test_pairwise(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
    num_permutations=10000,
    alpha_adj=0.0167,
):
    """
    Runs a non-parametric permutation test to compare the variances of two distance groups.

    Parameters:
    -----------
    df
        a data frame with country and distances columns.
    label_a
        a name of a country
    label_b
        a name of a country
    num_permutations, optional
        number of permutations, by default 10000
    alpha_adj, optional
        adjustment of the p value, by default 0.0167
    """
    group_a = df.query("country == @label_a")["distances"].tolist()
    group_b = df.query("country == @label_b")["distances"].tolist()
    dev_a = np.abs(group_a - np.median(group_a))
    dev_b = np.abs(group_b - np.median(group_b))
    observed_diff = np.abs(np.mean(dev_a) - np.mean(dev_b))

    # Combine the data
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    extreme_count = 0
    for _ in range(num_permutations):
        # Shuffle the labels
        shuffled = rng.permutation(combined)
        # Split into fake groups
        fake_a = shuffled[:n_a]
        fake_b = shuffled[n_a:]
        dev_fake_a = np.abs(fake_a - np.median(fake_a))
        dev_fake_b = np.abs(fake_b - np.median(fake_b))

        fake_diff = np.abs(np.mean(dev_fake_a) - np.mean(dev_fake_b))
        if fake_diff >= observed_diff:
            extreme_count += 1

    # p-value is the proportion of shuffles that produced a difference as large or larger
    p_val = extreme_count / num_permutations
    status = "SIGNIFICANT" if p_val < alpha_adj else "NOT SIGNIFICANT"

    print(f"{label_a} vs {label_b} (Permutation):")
    print(f"  Observed Mean Diff: {observed_diff}")
    print(f"  p-value:            {p_val} ({status})")


# %%
## Load data
df = pd.read_csv(PROC / "food_texts.csv", index_col=None)

embeddings_lst = []
for line in open(PROC / "food_texts_embeddings.jsonl", "r"):
    tmp = json.loads(line)
    embeddings_lst.append({"id": tmp["id"], "embeddings": tmp["embeddings"]})

df_embeddings = pd.DataFrame.from_dict(embeddings_lst)
df = pd.merge(df, df_embeddings, on="id").drop_duplicates("id")
# %%
## Compute PCA on normalized embeddings
df_lst = []
for _, tmp in df.groupby("country"):
    vectors = np.array(tmp["embeddings"].tolist())
    centroid = np.mean(vectors, axis=0, keepdims=True)
    ## Calculate cosine distance
    distances_2d = cdist(vectors, centroid, metric="cosine")
    cosine_distances = distances_2d.flatten()
    cosine_df = pd.DataFrame(cosine_distances, columns=["distances"])
    tmp = tmp.reset_index(drop=True).join(cosine_df)
    df_lst.append(tmp)

df = pd.concat(df_lst)

# %%
## MOTIVATION
df_mot = df.query("mot_refl > 0 | mot_auto > 0")

fig = plot_boxplot(df=df_mot, color=COLORS["yellow"], title="Motivations")
plt.tight_layout()
plt.savefig(PNG / "motivation_semantic_spread.png", dpi=200)

# %%
## Run the test on these distances
levene_test(df_mot, countries=["poland", "portugal", "uk"])
# %%
## CAPABILITIES
df_cap = df.query("cap_psychological > 0 | cap_physical > 0")

fig = plot_boxplot(
    df=df_cap, color=COLORS["blue"], sig=False, sig_level="*", title="Capabilities"
)
plt.tight_layout()
plt.savefig(PNG / "capabilities_semantic_spread.png", dpi=200)
# %%
## Run the test on these distances
levene_test(df_cap, countries=["poland", "uk"])
# %%
## OPPORTUNITIES
df_opp = df.query("opp_physical > 0 | opp_social > 0")

fig = plot_boxplot(
    df=df_opp, color=COLORS["green"], sig=True, sig_level="**", title="Opportunities"
)
plt.tight_layout()
plt.savefig(PNG / "opportunities_semantic_spread.png", dpi=200)
# %%
## Run the test on these distances
levene_test(df_opp, countries=["poland", "portugal", "uk"])

# %%
## Premutation test
permutation_test_pairwise(df_opp, label_a="poland", label_b="uk")
permutation_test_pairwise(df_opp, label_a="poland", label_b="portugal")
permutation_test_pairwise(df_opp, label_a="portugal", label_b="uk")
# %%
