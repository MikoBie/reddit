# %%
## Load modules
import pandas as pd
from pathlib import Path

## %
## Define Globals
ROOT = Path().absolute().parent
DATA = ROOT / "data"
RAW = DATA / "raw"


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
# %%
