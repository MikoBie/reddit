# %%
import pandas as pd
from pathlib import Path

# %%
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
XLSX = DATA / "xlsx"

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

column_uk = {
    "Comment": "body",
}


# %%
## Load the data
poland = pd.read_excel(XLSX / "Reddit_Poland.xlsx", index_col=0).rename(
    columns=column_poland
)
poland.loc[:, "cap_psychological":"mot_auto"] = (
    poland.loc[:, "cap_psychological":"mot_auto"]
    .map(lambda x: 1 if not pd.isna(x) else 0)
    .fillna(0)
    .filter(regex=r"body|^cap_|^opp_|^mot_")
)
portugal = pd.read_excel(XLSX / "Reddit_Portugal.xlsx", index_col=0)
portugal.loc[:, "En_Motiv_Refle_Moral (animals)":"Bar_Oppor_accesability"] = (
    portugal.loc[:, "En_Motiv_Refle_Moral (animals)":"Bar_Oppor_accesability"]
    .map(lambda x: 1 if not pd.isna(x) else 0)
    .fillna(0)
)
portugal = portugal.assign(
    cap_psychological=lambda x: x["Bar_Cap_knowledge about vege cuisine"],
    cap_physical=lambda x: x["En_Capab_Phys_Economic"] + x["Bar_cap_Phys_health"],
    opp_accessability=lambda x: x["Bar_Oppor_accesability"],
    opp_affordability=lambda x: x["Bar_Oport_Phys_economic"],
    opp_social_influence=lambda x: x["En_Opport_Social_Circle"],
    opp_tradition=lambda x: x["Bar_Opport_Social_family traditions"],
    mot_refl_moral=lambda x: x["En_Motiv_Refle_Moral (animals)"],
    mot_refl_environmental=lambda x: x["En_Motiv_Refle_Environment"],
    mot_refl_health_wellbeing=lambda x: x["En_Motiv_Refle_Health"]
    + x["En_Motiv_Refle_well-being"]
    + x["Bar_Mot_Ref_health"],
    mot_refl_disonance=lambda x: x["En_Motiv_Ref_Reduction of cognitive disonance"],
).filter(regex=r"body|^cap_|^opp_|^mot_")
uk = pd.read_excel(XLSX / "Reddit_UK.xlsx", index_col=0).rename(columns=column_uk)
uk.loc[:, "Cap_Physical_Bar_Lack of Time":"Mot_Aut_Bar_Personal Preference"] = (
    uk.loc[:, "Cap_Physical_Bar_Lack of Time":"Mot_Aut_Bar_Personal Preference"]
    .map(lambda x: 1 if not pd.isna(x) else 0)
    .fillna(0)
)
uk = uk.assign(
    cap_psychological=lambda x: x["Cap_Psych_Bar_Knowledge"]
    + x["Cap_Psych_Bar_Cooking Skills"],
    cap_physical=lambda x: x["Cap_Physical_Bar_Lack of Time"]
    + x["Cap_Physical_Ena_Health"]
    + x["Cap_Physical_Bar_Health"],
    opp_affordability=lambda x: x["Opp_Physical_Ena_Affordability"]
    + x["Opp_Physical_Bar_Affordability"],
    opp_accessability=lambda x: x["Opp_Physical_Ena_Accessibility"],
    opp_social_influence=lambda x: x["Opp_Soc_Ena_Social Circle"]
    + x["Opp_Soc_Bar_Social Circle"],
    opp_tradition=lambda x: x["Opp_Soc_Ena_Family Traditions"]
    + x["Opp_Soc_Bar_Family Traditions"],
    mot_refl_moral=lambda x: x["Mot_Ref_Ena_Moral"],
    mot_refl_environmental=lambda x: x["Mot_Ref_Ena_Environment"],
    mot_refl_health_wellbeing=lambda x: x["Mot_Ref_Ena_Health"]
    + x["Mot_Ref_Ena_Well Being"]
    + x["Mot_Ref_Bar_Health"],
    mot_refl_disonance=lambda x: x["Mot_Ref_Ena_Decrease of cognitive dissonance"],
    mot_auto=lambda x: x["Mot_Aut_Ena_Personal Preference"]
    + x["Mot_Aut_Bar_Personal Preference"],
).filter(regex=r"body|^cap_|^opp_|^mot_")
# %%
portugal_texts = portugal.sample(100, random_state=8710).loc[:, "body"].to_list()
uk_texts = uk.sample(100, random_state=8710).loc[:, "body"].to_list()
poland_texts = poland.sample(100, random_state=8710).loc[:, "body"].to_list()
# %%
pd.DataFrame(
    {"poland": poland_texts, "portugal": portugal_texts, "uk": uk_texts}
).to_csv(DATA / "food_texts.csv", index=False)

# %%
