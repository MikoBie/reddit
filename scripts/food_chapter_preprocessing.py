# %%
import pandas as pd
from pathlib import Path
from hashlib import blake2b

# %%
## Global variables
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
XLSX = DATA / "xlsx"
PROC = DATA / "processed"

column_poland = {
    "Unnamed: 1": "body",
    "Capabilities": "cap_psychological",
    "Unnamed: 3": "cap_physical",
    "Opportunities": "opp_accessability",
    "Unnamed: 5": "opp_affordability",
    "Unnamed: 6": "opp_social_influence",
    "Unnamed: 7": "opp_tradition",
    "Motivation": "mot_refl_moral",
    "Unnamed: 9": "mot_refl_environmental",
    "Unnamed: 10": "mot_refl_health_wellbeing",
    "Unnamed: 11": "mot_refl_easiness",
    "Unnamed: 12": "mot_refl_decision",
    "Unnamed: 13": "mot_auto",
}

column_comment = {
    "Comment": "body",
}

rgx = r"^cap_|^opp_|^mot_"

selected_columns = [
    "body",
    "cap_psychological",
    "cap_physical",
    "opp_physical",
    "opp_social",
    "mot_refl",
    "mot_auto",
    "country",
]
# %%
## POLAND
poland = pd.read_excel(XLSX / "Reddit_Poland.xlsx", index_col=0).rename(
    columns=column_poland
)

columns_to_map = poland.filter(regex=rgx).columns.tolist()

poland[columns_to_map] = (
    poland.loc[:, "cap_psychological":"mot_auto"]
    .map(lambda x: 1 if not pd.isna(x) else 0)
    .fillna(0)
    .filter(regex=rgx)
)

# %%
## PORTUGAL
portugal = pd.read_excel(XLSX / "Reddit_Portugal_COMB.xlsx", index_col=0).rename(
    columns=column_comment
)

columns_to_map = portugal.loc[:, "Health":"Responsibility"].columns.tolist()

portugal[columns_to_map] = (
    portugal.loc[:, columns_to_map].map(lambda x: 1 if not pd.isna(x) else 0).fillna(0)
)
portugal = portugal.assign(
    cap_psychological=lambda x: x["Knowledge"],
    cap_physical=lambda x: x["Health"],
    opp_physical=lambda x: x["Accessibility"] + x["Affordability"],
    opp_social=lambda x: x["Social Environment"] + x["Traditions"],
    mot_refl=lambda x: x["Moral"]
    + x["Environment"]
    + x["Health.1"]
    + x["Responsibility"],
    mot_auto=lambda x: x["Personal Preference"],
).filter(regex=r"body|^cap_|^opp_|^mot_")

# %%
## UK
uk = pd.read_excel(XLSX / "Reddit_UK_COMB.xlsx", index_col=0).rename(
    columns=column_comment
)

columns_to_map = uk.loc[:, "Health":"Personal Preference"].columns.tolist()

uk[columns_to_map] = (
    uk.loc[:, "Health":"Personal Preference"]
    .map(lambda x: 1 if not pd.isna(x) else 0)
    .fillna(0)
)
uk = uk.assign(
    cap_psychological=lambda x: x["Cap_Psych_Bar_Knowledge"],
    cap_physical=lambda x: x["Health"],
    opp_physical=lambda x: x["Accessibility"] + x["Opp_Physical_Bar_Affordability"],
    opp_social=lambda x: x["Traditions"] + x["Social Circle"],
    mot_refl=lambda x: x["Mot_Ref_Ena_Moral"]
    + x["Mot_Ref_Ena_Environment"]
    + x["Responsability"]
    + x["Health.1"],
    mot_auto=lambda x: x["Personal Preference"],
).filter(regex=r"body|^cap_|^opp_|^mot_")

# %%
## Compute macro categories
poland = poland.assign(
    mot_refl=lambda x: x.filter(regex=r"^mot_refl_").sum(axis=1),
    opp_physical=lambda x: x["opp_accessability"] + x["opp_affordability"],
    opp_social=lambda x: x["opp_social_influence"] + x["opp_tradition"],
    country="poland",
).filter(items=selected_columns)
portugal = portugal.assign(
    country="portugal",
).filter(items=selected_columns)

uk = uk.assign(
    country="uk",
).filter(items=selected_columns)

# %%
## Concatenate all dataframes and create unique id for each text
data = (
    pd.concat([poland, portugal, uk], axis=0)
    .reset_index(drop=True)
    .assign(
        id=lambda x: x["body"].apply(
            lambda y: blake2b(y.encode("utf-8"), digest_size=5).hexdigest()
        )
    )
)

columns_to_map = data.filter(regex=rgx).columns.tolist()

data[columns_to_map] = data.loc[:, columns_to_map].map(lambda x: 0 if x == 0 else x / x)

data[["id"] + selected_columns].to_csv(PROC / "food_texts.csv", index=False)
# %%
