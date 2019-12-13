
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import seaborn as sns


# In[ ]:


df = pd.read_csv("for_imputation_5Dec.csv")
df = df[~df.tree_cover.isna()]
df = df[df.state_id != "AK"]
df = df.eval("pop_density = population / total_area")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


CAT_COLS = ["unique_id", "state_id", "county_id", "state_name", "county_name",
            "tree_diversity", "class", "jenks"]

def make_covariate_string(covariates):
    ret_str = ""
    
    for cov in covariates:
        if cov in ["state_id", "county_id"]:
            ret_str += "C("
            ret_str += cov
            ret_str += ")"
        else:
            ret_str += cov
        ret_str += " + "
    
    return ret_str[:-3]


def impute_value(df, value_name, predictors, non_transformed_cols = CAT_COLS):
    
    na_mask = df.loc[:, value_name].isnull()
    known_vals = df[~na_mask]
    na_vals = df[na_mask]
    
    cont_known_vals = np.log1p(known_vals.loc[:, ~known_vals.columns.isin(CAT_COLS)])
    cont_na_vals = np.log1p(na_vals.loc[:, ~na_vals.columns.isin(CAT_COLS)])
    
    value_index = cont_known_vals.columns.tolist().index(value_name)
    
    scaler = StandardScaler()
    cont_known_vals = pd.DataFrame(scaler.fit_transform(cont_known_vals.values),
                                index = cont_known_vals.index,
                                columns = cont_known_vals.columns)
    
    cont_na_vals = pd.DataFrame(scaler.transform(cont_na_vals.values),
                                   index = cont_na_vals.index,
                                   columns = cont_na_vals.columns)
    
    cont_known_vals["state_id"] = df.loc[cont_known_vals.index, "state_id"]
    cont_na_vals["state_id"] = df.loc[cont_na_vals.index, "state_id"]
    cont_known_vals["county_id"] = df.loc[cont_known_vals.index, "county_id"]
    cont_na_vals["county_id"] = df.loc[cont_na_vals.index, "county_id"]

    pred_str = make_covariate_string(predictors)
    
    model = ols("{} ~ {}".format(value_name, pred_str),
                        data = cont_known_vals).fit()
    
    cont_na_vals.loc[:, value_name] = model.predict(cont_na_vals)
    
    no_state_county = cont_na_vals.drop(["state_id", "county_id"], axis = 1)
    
    cont_na_vals = pd.DataFrame(scaler.inverse_transform(no_state_county),
                                   index = no_state_county.index,
                                   columns = no_state_county.columns)
    
    df.loc[cont_na_vals.index, value_name] = (np.e ** cont_na_vals.loc[:, value_name]) - 1
    
    return df
    


# In[ ]:


imputed_df = impute_value(df, "any_pa", ["state_id", "year", "median_household_income"])
imputed_df = impute_value(imputed_df, "sufficient_pa", ["state_id", "year", "median_household_income"])
imputed_df = impute_value(imputed_df, "obesity_pct", ["state_id", "year", "median_household_income"])
imputed_df = impute_value(imputed_df, "avg_max_temp", ["year", "county_id"])
imputed_df = impute_value(imputed_df, "avg_pm25_cdc", ["year", "county_id"])
imputed_df = impute_value(imputed_df, "avg_precip", ["year", "county_id"])
imputed_df = impute_value(imputed_df, "avg_sun", ["year", "county_id"])


# In[ ]:


sns.boxplot(data = imputed_df, x = "year", y = "obesity_pct")
plt.ylabel("Percentage Obese")


# In[ ]:


imputed_df = imputed_df[~imputed_df.population.isna()]
df = df[~df.population.isna()]


# In[ ]:


cont_vals = df.loc[:, ~df.columns.isin(CAT_COLS + ["year"])]
cont_vals = np.log1p(cont_vals)

cont_vals.mean().to_csv("log_transformed_mean_vals.csv")
cont_vals.std().to_csv("log_transformed_std_vals.csv")

scaler = StandardScaler()
z_cont_vals = pd.DataFrame(scaler.fit_transform(cont_vals.values),
                           index = cont_vals.index,
                           columns = cont_vals.columns)

df.loc[z_cont_vals.index, z_cont_vals.columns] = z_cont_vals
df = df.dropna(axis = 1)
df.to_csv("cleaned_imputed_tree_air_poverty_log1p_z_scored.csv")

