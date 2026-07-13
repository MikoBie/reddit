# %%
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PNG = ROOT / "png"

# %%
## Load data
df = pd.read_csv(PROC / "food_texts.csv", index_col=None)

embeddings_lst = []
for line in open(PROC / "food_texts_embeddings.jsonl", "r"):
    tmp = json.loads(line)
    embeddings_lst.append({"id": tmp["id"], "embeddings": tmp["embeddings"]})

df_embeddings = pd.DataFrame.from_dict(embeddings_lst)
df = pd.merge(df, df_embeddings, on="id")
# %%
df_lst = []
for _, tmp in df.groupby("country"):
    vectors = np.array(tmp["embeddings"].tolist())
    matrix = np.vstack(vectors)
    # Calculate Euclidean distance
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms
    pca = PCA(n_components=2, random_state=8710).fit_transform(matrix)
    if _ == "portugal":
        print(pca)
    pca_df = pd.DataFrame(pca, columns=["Component A", "Component B"])
    tmp = tmp.join(pca_df)
    df_lst.append(tmp)

df = pd.concat(df_lst)
# %%
df_cap = df.query("cap_psychological > 0 | cap_physical > 0")

fix, axs = plt.subplots(figsize=(9, 4), nrows=1, ncols=3)
for n, t in enumerate(df_cap.groupby("country")):
    country, tmp = t
    plot = axs[n].plot(tmp["Component A"], tmp["Component B"], "o")
    axs[n].title.set_text(country.upper())

plt.show()


# %%
df_opp = df.query("opp_physical > 0 | opp_social > 0")

fix, axs = plt.subplots(figsize=(9, 4), nrows=1, ncols=3)
for n, t in enumerate(df_opp.groupby("country")):
    country, tmp = t
    plot = axs[n].plot(tmp["Component A"], tmp["Component B"], "o")
    axs[n].title.set_text(country.upper())

plt.show()

# %%
df_mot = df.query("mot_refl > 0 | mot_auto > 0")

fix, axs = plt.subplots(figsize=(9, 4), nrows=1, ncols=3)
for n, t in enumerate(df_mot.groupby("country")):
    country, tmp = t
    plot = axs[n].plot(tmp["Component A"], tmp["Component B"], "o")
    axs[n].title.set_text(country.upper())

plt.show()
# %%
