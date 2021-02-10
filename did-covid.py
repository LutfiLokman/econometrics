import pandas as pd
from statsmodels.formula.api import ols
from stargazer.stargazer import Stargazer

df = pd.read_csv(
    "https://raw.githubusercontent.com/narxiss24/datasets/master/owid-covid.csv"
)

mob_th = pd.read_csv(
    "https://raw.githubusercontent.com/narxiss24/datasets/master/2020_TH_Region_Mobility_Report.csv"
)

mob_th = mob_th[mob_th["sub_region_1"].isna()]

mob_th["date"] = pd.to_datetime(mob_th["date"])

mob_my = pd.read_csv(
    "https://raw.githubusercontent.com/narxiss24/datasets/master/2020_MY_Region_Mobility_Report.csv"
)

mob_my = mob_my[mob_my["sub_region_1"].isna()]

mob_my["date"] = pd.to_datetime(mob_my["date"])

country = pd.get_dummies(df["location"])

df = pd.concat([country, df], axis=1)

df.drop(["Thailand", "location"], axis=1, inplace=True)

df.dropna(subset=["new_tests_per_thousand"], inplace=True)

df["date"] = pd.to_datetime(df["date"])

df_my = df[df["Malaysia"] == 1]

df_my = df_my.merge(mob_my, on="date")

df_th = df[df["Malaysia"] == 0]

df_th = df_th.merge(mob_th, on="date")

df = pd.concat([df_my, df_th])

df["post_election"] = df["date"].apply(
    lambda x: 0 if x <= pd.to_datetime("2020-09-26") else 1
)

mdl_joined_el = ols(
    "new_cases_per_million ~ Malaysia * post_election + new_tests_per_thousand + retail + grocery + parks + transit", data=df
)
results_joined_el = mdl_joined_el.fit()
results_joined_el.summary()

stargazer = Stargazer([results_joined_el])

stargazer.render_latex()
