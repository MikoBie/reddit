# %%
## Load modules
import pandas as pd
from pathlib import Path

## %
## Define Globals
ROOT = Path().absolute().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"

# %%
pl_df = (
    pd.read_excel(RAW / "Reddit_Poland_COMB.xlsx", header=[0, 1])
    .loc[:, ["Unnamed: 1_level_0", "Capabilities", "Motivation", "Opportunities"]]
    .reset_index()
    .rename(columns={"Unnamed: 1_level_0": "", "index": "id"})
)
pl_df.columns = [" ".join(col).strip() for col in pl_df.columns.values]
pl_df = pl_df.query("body == body").melt(id_vars=["id", "body"], var_name="comb")
pl_df.loc[:, "value"] = pl_df.apply(
    lambda x: x.comb.split()[0] if isinstance(x.value, str) else "None", axis=1
)

pl_df = (
    pl_df.drop_duplicates(["body", "value"])
    .pivot(index=["id", "body"], columns="value", values="value")
    .reset_index()
    .drop(["None"], axis=1)
)
pl_df.id = pl_df.id.apply(lambda x: f"pl_{int(x)}")
# %%
uk_df = (
    pd.read_excel(RAW / "Reddit_UK.xlsx")
    .rename(columns={"Unnamed: 0": "id", "Comment": "body"})
    .drop(["Unnamed: 27", "Unnamed: 28", "Unnamed: 29"], axis=1)
    .query("body == body")
    .melt(id_vars=["id", "body"], var_name="comb")
)

uk_df.loc[:, "value"] = uk_df.apply(
    lambda x: x.comb.split("_")[0] if x.value == x.value else "None", axis=1
)

uk_df = (
    uk_df.drop_duplicates(["body", "value"])
    .pivot(index=["id", "body"], columns="value", values="value")
    .reset_index()
    .drop(["None", "Bar", "Responsability"], axis=1)
    .rename(
        columns={"Cap": "Capabilities", "Mot": "Motivation", "Opp": "Opportunities"}
    )
)
uk_df.id = uk_df.id.apply(lambda x: f"uk_{int(x)}")
# %%
por_df = (
    pd.read_excel(RAW / "Reddit_Portugal.xlsx")
    .rename(columns={"Unnamed: 0": "id", "Comment": "body"})
    .query("body == body")
    .melt(id_vars=["id", "body"], var_name="comb")
)

por_df.loc[:, "value"] = por_df.apply(
    lambda x: x.comb.split("_")[0] if x.value == x.value else "None", axis=1
)

por_df = (
    por_df.drop_duplicates(["body", "value"])
    .pivot(index=["id", "body"], columns="value", values="value")
    .reset_index()
    .drop(["None", "Responsability"], axis=1)
    .rename(
        columns={"Cap": "Capabilities", "Mot": "Motivation", "Opp": "Opportunities"}
    )
)
por_df.id = por_df.id.apply(lambda x: f"por_{int(x)}")
# %%
df = pd.concat([pl_df, por_df, uk_df]).replace(
    {"Cap": "Capabilities", "Mot": "Motivation", "Opp": "Opportunities"}
)
df.to_excel(PROC / "df.xlsx")
