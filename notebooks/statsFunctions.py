import statsmodels.api as sm
import pandas as pd
import scipy.stats as scs
import numpy as np

# Create a function to build a statsmodels ols model
def build_sm_ols(df, features_to_use, target, add_constant=False, show_summary=True):
    X = df[features_to_use]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X).fit()
    if show_summary:
        print(ols.summary())
    return ols


# create a function to check the validity of your model
# it should measure multicollinearity using vif of features
# it should test the normality of your residuals 
# it should plot residuals against an xaxis to check for homoskedacity
# it should implement the Breusch Pagan Test for Heteroskedasticity
##  Ho: the variance is constant            
##  Ha: the variance is not constant


# assumptions of ols
# residuals are normally distributed
def check_residuals_normal(ols):
    residuals = ols.resid
    t, p = scs.shapiro(residuals)
    if p <= 0.05:
        return False
    return True


# residuals are homoskedasticitous
def check_residuals_homoskedasticity(ols):
    import statsmodels.stats.api as sms
    resid = ols.resid
    exog = ols.model.exog
    lg, p, f, fp = sms.het_breuschpagan(resid=resid, exog_het=exog)
    if p >= 0.05:
        return True
    return False




def check_vif(df, features_to_use, target_feature):
    ols = build_sm_ols(df=df, features_to_use=features_to_use, target=target_feature, show_summary=False)
    r2 = ols.rsquared
    return 1 / (1 - r2)
    
    
    
# no multicollinearity in our feature space
def check_vif_feature_space(df, features_to_use, vif_threshold=3.0):
    all_good_vif = True
    for feature in features_to_use:
        target_feature = feature
        _features_to_use = [f for f in features_to_use if f!=target_feature]
        vif = check_vif(df=df, features_to_use=_features_to_use, target_feature=target_feature)
        if vif >= vif_threshold:
            print(f"{target_feature} surpassed threshold with vif={vif}")
            all_good_vif = False
    return all_good_vif
        
        


def check_model(df, 
                features_to_use, 
                target_col, 
                add_constant=False, 
                show_summary=False, 
                vif_threshold=3.0):
    has_multicollinearity = check_vif_feature_space(df=df, 
                                                    features_to_use=features_to_use, 
                                                    vif_threshold=vif_threshold)
    if not has_multicollinearity:
        print("Model contains multicollinear features")
    
    # build model 
    ols = build_sm_ols(df=df, features_to_use=features_to_use, 
                       target=target_col, add_constant=add_constant, 
                       show_summary=show_summary)
    
    # check residuals
    resids_are_norm = check_residuals_normal(ols)
    resids_are_homo = check_residuals_homoskedasticity(ols)
    
    if not resids_are_norm or not resids_are_homo:
        print("Residuals failed test/tests")
    return ols

def corr_function(x, y):
    try:
        return x.corr(y)
    except:
        return None
    
def gen_range(start, stop, step):
    current = start
    while current < stop:
        next_current = current + step
        if next_current < stop:
            yield (current + 1, next_current)
        else:
            yield (current + 1, stop)
        current = next_current
        
def bootstrap(arr):
    return np.random.choice(arr, size=arr.shape, replace=True)


def generate_sample_mus(arr, num_samples=30):
    sample_mus = [] 
    for i in range(num_samples):
        sample = bootstrap(arr)
        mu = np.mean(sample)
        sample_mus.append(mu)
    return sample_mus


def test_for_normality(arr, confidence=0.95):
    t, p = scs.shapiro(arr)
    if p <= 1 - confidence:
        print("reject the null")
        return False
    print("fail to reject the null")
    return True


def cohens_d(arr1, arr2):
    narr1 = len(arr1)
    narr2 = len(arr2)
    dof = narr1 + narr2 - 2
    return (np.mean(arr1) - np.mean(arr2)) / np.sqrt(((narr1-1)*np.std(arr1, ddof=1) ** 2 + (narr2-1)*np.std(arr2, ddof=1) ** 2) / dof)