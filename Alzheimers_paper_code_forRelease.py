# %% Import relevant libraries, define custom functions 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
from umap import UMAP
from statsmodels.stats.multitest import fdrcorrection
plt.rcParams["svg.fonttype"] = "none"

def plot_kde(V, ax, colour = 'k', label=None):
    # V is a matrix where columns = variables and rows = samples
    # ax is the axis to attach the plot to
    kernel                  = gaussian_kde(V.T)
    xmin, ymin, xmax, ymax  = np.r_[V.min(axis = 0)-2, V.max(axis = 0)+2]
    
    # Peform the kernel density estimate
    xx, yy                  = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions               = np.vstack([xx.ravel(), yy.ravel()])
    f                       = np.reshape(kernel(positions).T, xx.shape)
    
    ax.contour(xx, yy, f, colors=colour, levels = 2, linewidths = 2, label=label)
    
    return ax

# %% Clustering of features using UMAP

# Load raw feature set # TODO n.b. data are withheld
file_name           = "features.csv"
output_large        = pd.read_csv(file_name, index_col="Unnamed: 0")

# Note: slight variation is expected due to setting different random seeds
pca                 = UMAP(random_state=42) # default is 2 components

# Clean out FTD patients, we are only looking at HC vs AD
subset              = output_large.copy().iloc[:, 5:].loc[output_large.Group!="F", :]
subset_clinical     = output_large.copy().iloc[:, :5].loc[output_large.Group!="F", :]
subset[subset==0]   = np.nan
subset.dropna(axis=1, inplace=True)
pca_out             = pca.fit_transform(subset)
out                 = spearmanr(pca_out, subset_clinical['MMSE'])[0] # Statistic

# Boolean selector for the correct participant subset. Makes plotting easier
criterion           = subset_clinical['Group']=="A"

# Define axes, plot
fig, ax             = plt.subplots()
ax.scatter(pca_out[:, 0], pca_out[:, 1], c = criterion)
plot_kde(V=pca_out[criterion.values==True, :], ax = ax, colour = "y")
plot_kde(V=pca_out[criterion.values==False, :], ax = ax, colour = "m")
plt.xlabel("UMAP component 1 (a.u.)")
plt.ylabel("UMAP component 2 (a.u.)")

# %% Univariate tests - t-tests and correlations with MMSE and features (<-patients only)

# Load raw feature set # TODO n.b. data are withheld
file_name           = "features.csv"
output_large        = pd.read_csv(file_name, index_col="Unnamed: 0")

subset              = output_large.copy().loc[:, [("M1" not in o) and ("M2" not in o) for o in output_large]] # and ("Power" not in o)
subset[subset==0]   = np.nan
subset.dropna(axis=1, inplace=True)

# Subset for just AD and HC
output_large        = output_large.loc[(output_large.Group=="A") | (output_large.Group=="C"), :]

# Stats tests - loop over features
stats_summary       = {}

for c in output_large.columns[5:]:
    ttest_res           = ttest_ind(output_large.loc[output_large.Group=="A", c].values, output_large.loc[output_large.Group=="C", c].values)
    spear_res           = spearmanr(output_large.loc[output_large.Group=="A", "MMSE"].values, output_large.loc[output_large.Group=="A", c].values)
    
    stats_summary.update({c: {"r": spear_res.statistic,
                             "r_p": spear_res.pvalue,
                             "t": ttest_res.statistic,
                             "t_p": ttest_res.pvalue}})

# Convert results to DF
stats_summary_df    = pd.DataFrame(stats_summary).T
stats_summary_df.dropna(inplace=True) # Handles degenerate connectivity columns - # TODO can also do this as preprocessing

# Perform FDR adjustment of p-values
fdrc                = fdrcorrection(np.r_[stats_summary_df["r_p"].values.reshape(-1,1), stats_summary_df["t_p"].values.reshape(-1,1)].ravel())
stats_summary_df["r_p_is_sig"] = fdrc[0][:stats_summary_df.shape[0]]
stats_summary_df["r_p_corrected"] = fdrc[1][:stats_summary_df.shape[0]]
stats_summary_df["t_p_is_sig"] = fdrc[0][stats_summary_df.shape[0]:]
stats_summary_df["t_p_corrected"] = fdrc[1][stats_summary_df.shape[0]:]

# Some basic reporting for the Results section in the paper
np.sum(stats_summary_df["t_p_is_sig"]) # 628 signifciant comparisons between groups
np.sum(stats_summary_df["r_p_is_sig"]) # 70 signifciant correlations within AD group vs MMSE

# %% Sueprvised ML analysis - Nested CV approach

lab_                = "A" # Positive class label. "A" = AD
seed_results        = {}
demo_deltas_seed    = {}

for seed in np.arange(20):
    # Set seed per-iteration for reproducibility
    np.random.seed(seed)
    
    
    all_results         = {}
    output_large        = pd.read_csv("C:/Users/leifs/Documents/CoveNeuro/CoveNeuro/Alzheimers EEG/AD_results_new_1.0_sec_V2.csv",
                                      index_col="Unnamed: 0")
    output_large        = output_large.copy().loc[:, [("M1" not in o) and ("M2" not in o) for o in output_large]]
    output_large[output_large==0] = np.nan
    output_large.dropna(axis=1, inplace=True)
    
    # Subset features
    output_large        = output_large.loc[(output_large.Group==lab_) | (output_large.Group=="C"), :]
    X                   = output_large.iloc[:, 5:].copy()
    y                   = output_large['Group'].copy().values
    
    # Wrangle labels into useful format
    y1                  = y.copy()
    y1[y1=="A"]         = 1
    y1[y1=="C"]         = 0
    y1                  = y1.astype(int)
    
    # Define pipeline
    sel1                = SelectFromModel(RandomForestClassifier(class_weight="balanced", random_state=seed))
    pipe                = Pipeline(steps=[
                                    ("sel", sel1),
                                    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=seed))
                                    ])
    # Parameter grid for CV. Provided as a pass-through
    param_grid          = {
                            'classifier__max_depth': [None],
                            }

    # Target feature subsets. n.b. some added here vs paper due to later presentations based on these analyses, no change to core paper results
    targets             = ["app_entropy", "PE_", "F3", "F4", "P3", "P4", "ms_", "trans", "katz", "higuchi", "hjorth", "decorr", "zero_crossings", "conn_theta", "conn_alpha", "conn_beta", "None", "std_"]
    
    # Iterate through target feature subsets
    demo_deltas_target  = {}
    
    for target in targets:
        print(target)

        if target=="None":
            sel_cols        = X.columns
            X1              = X.copy().loc[:, sel_cols]
            colnames        = X1.columns
            
        else:
            sel_cols        = [target in c for c in X.columns]
            X1              = X.copy().loc[:, sel_cols]
            colnames        = X1.columns
    
        inner_cv        = KFold(n_splits=5, shuffle=True, random_state=seed)
        outer_cv        = KFold(n_splits=5, shuffle=True, random_state=seed)
        
        clf             = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, scoring = "roc_auc", verbose=1, n_jobs=2)
        nested_model    = cross_validate(estimator = clf, 
                                         X = X1, 
                                         y = y1, 
                                         cv=outer_cv,
                                         scoring = "roc_auc", 
                                         return_train_score=False,
                                         return_estimator=True,
                                         n_jobs=-1
                                         )
        scores          = nested_model['test_score']  # Equivalent to output of cross_val_score()
        estimators      = nested_model['estimator']
        
        # Same mechanics as above(i.e., external test set, etc.), but generates comprehensive 
        # predictions for generating confusion matrices
        nested_preds    = cross_val_predict(estimator = clf, 
                                            X = X1, 
                                            y = y1, 
                                            cv=outer_cv,
                                            n_jobs=-1, 
                                            method='predict_proba'
                                            )
        
        y_pred      = nested_preds[:, 1].round()
        cf_mat      = confusion_matrix(y1, y_pred)
        
        # TODO new - Rev 3 comment about demos - MMSE, Age, Sex
        preds_demos = output_large.loc[:, ["Gender", "Group", "Age", "MMSE"]].copy()
        preds_demos.insert(0, "y_pred", y_pred)
        preds_demos.insert(0, "y_true", y1)
        preds_demos.insert(2, "false_neg", ((y1==1) & (y_pred==0)).astype(int))
        preds_demos.insert(2, "false_pos", ((y1==0) & (y_pred==1)).astype(int))
        
        false_neg_MMSE_1 = preds_demos.loc[(preds_demos['false_neg']==1)&(preds_demos['Group']=="A"), "MMSE"].values
        false_neg_MMSE_0 = preds_demos.loc[(preds_demos['false_neg']==0)&(preds_demos['Group']=="A"), "MMSE"].values
        t_MMSE, p_MMSE = ttest_ind(false_neg_MMSE_1, false_neg_MMSE_0)
        
        false_neg_Age_1 = preds_demos.loc[(preds_demos['false_neg']==1)&(preds_demos['Group']=="A"), "Age"].values
        false_neg_Age_0 = preds_demos.loc[(preds_demos['false_neg']==0)&(preds_demos['Group']=="A"), "Age"].values
        t_Age, p_Age = ttest_ind(false_neg_Age_1, false_neg_Age_0)
        
        false_neg_Sex_1 = preds_demos.loc[(preds_demos['false_neg']==1)&(preds_demos['Group']=="A"), "Gender"].values
        false_neg_Sex_0 = preds_demos.loc[(preds_demos['false_neg']==0)&(preds_demos['Group']=="A"), "Gender"].values
        false_neg_Sex_1[false_neg_Sex_1=="F"] = 1
        false_neg_Sex_1[false_neg_Sex_1=="M"] = 0
        false_neg_Sex_0[false_neg_Sex_0=="F"] = 1
        false_neg_Sex_0[false_neg_Sex_0=="M"] = 0
        t_Sex, p_Sex = ttest_ind(false_neg_Sex_1.astype(int), false_neg_Sex_0.astype(int))
        
        ### Plotting confusion matrices - TURN OFF WHEN RUNNING THE WHOLE LOOP
        # fig, ax = plt.subplots(1, 1, figsize = (5, 5))
        # ax.set_title("Confusion matrix", fontsize=12, fontweight = "bold")
        # sns.heatmap(data = pd.DataFrame(cf_mat, columns = ["HC", "AD"], index = ["HC", "AD"]), 
        #             vmax=cf_mat.max(), vmin=0,
        #             annot=True,
        #             ax = ax,
        #             cbar=False)
        
        # Eval
        fpr, tpr, _ = roc_curve(y1, nested_preds[:, 1])
        roc_auc     = auc(fpr, tpr)
        
        ### Plotting ROC and confusion matrices - TURN OFF WHEN RUNNING THE WHOLE LOOP
        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % np.mean(scores))
        # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("Receiver operating characteristic: AD vs HC", fontsize = 12, fontweight = "bold")
        # plt.legend(loc="lower right")
        # plt.show()
        
        # Selected feature analysis
        selected    = [estimators[i].best_estimator_.named_steps.sel.get_support() for i in range(len(estimators))]
        selected    = pd.DataFrame(np.r_[selected].T.astype(int), index = colnames)

        # Update storage dicts        
        all_results.update({target: {"scores": scores,
                                     "selected": selected,
                                     "fpr": fpr,
                                     "tpr": tpr,
                                     "cf_mat": cf_mat}})
        demo_deltas_target.update({target: {"t_MMSE": t_MMSE,
                                     "p_MMSE": p_MMSE,
                                     "t_Age": t_Age,
                                     "p_Age": p_Age,
                                     "t_Sex": t_Sex,
                                     "p_Sex": p_Sex}})
    
    # Update storage dicts
    seed_results.update({seed: all_results})
    demo_deltas_seed.update({seed: demo_deltas_target})

# Easier for manual viewing
df_temps_list       = []

for i in list(demo_deltas_seed.keys()):
    df_temp             = pd.DataFrame(demo_deltas_seed[i])
    df_temp.index       = [f"{i}_{j}" for j in df_temp.index]
    df_temps_list.append(df_temp)

df_temps_all = pd.concat(df_temps_list)#.T

# %% Aggregating results and plotting performance characteristics

# Iterate through - and save - AUROC results
keyed_results       = []
all_keys            = ["app_entropy", "PE_", "F3", "F4", "P3", "P4", "ms_", "trans", "katz", "higuchi", "hjorth", "decorr", "zero_crossings", "conn_theta", "conn_alpha", "conn_beta", "None", "std_"]#list(seed_results[1].keys())#["F3", "F4", "P3", "P4", "higuchi"]#

for key in all_keys:
    keyed_results.append(np.array([seed_results[s+1][key]['scores'] for s in range(len(seed_results))]).mean(axis=1))

keyed_results_df    = pd.DataFrame(data = np.array(keyed_results), 
                                index = all_keys,
                                columns = np.arange(len(seed_results))+1).T
keyed_results_df.to_csv("AUROCS_per_seed.csv")

# Plotting saved results (get stats)
keyed_results_df    = pd.read_csv("AUROCS_per_seed.csv", index_col="Unnamed: 0")
keyed_means         = keyed_results_df.median(axis=0)
keyed_SDs           = keyed_results_df.std(axis=0)
sorted_             = np.argsort(keyed_means)[::-1]
keyed_means         = keyed_means.values[sorted_]
keyed_SDs           = keyed_SDs.values[sorted_]

# Plotting saved results (Generate plots)
fig, ax             = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title("Performance of different feature subsets")
plt.subplots_adjust()#bottom=0.25
ax.axhline(0.5, linestyle=":", c="k")
sns.boxplot(keyed_results_df.iloc[:, sorted_], ax = ax)
ax.set_ylim(0.45, 1)
ax.set_ylabel("AUROC")
labs                = list(all_keys)
ax.set_xticklabels(np.array(labs)[sorted_], rotation=45, fontsize=12)

for idx, (m, s) in enumerate(zip(keyed_means, keyed_SDs)):
    ax.text(idx, 0.95, f"{np.round(m, 2)}+/-\n{np.round(s, 2)}", ha="center")

# %% Compiling feature importance matrices

for key in targets:
    # Define a list to enable easier storage for now
    importances_list    = []
    
    for s in seed_results:
        importances_list.append(np.array(seed_results[s][key]["selected"])[:, :, np.newaxis])
    
    # Convert list --> array
    importances_array   = np.concatenate(importances_list, axis=2)
    importances_array_1 = np.mean(importances_array, axis=2)
    
    # Compile to DF
    importances_df      = pd.DataFrame(importances_array_1, index = seed_results[s][key]["selected"].index)
    importances_df["mean"] = importances_df.mean(axis=1)
    importances_df.sort_values(by="mean", inplace=True, ascending=False)
    
    importances_df.to_csv(f"importances_{key}_V2.csv")

# %% Plotting boxplots of individual features

# Boxplots of feature values
feature = "mean_hjorth_complexity_chan_F4_alpha"

plt.figure()
plt.title(f"Comparison of automatically-selected important features:\n{feature}", fontsize=12, fontweight="bold")

controls = output_large[feature][y1==0]
patients = output_large[feature][y1==1]
plt.boxplot(controls, positions = [0])
plt.boxplot(patients, positions = [1])

plt.xticks([0, 1], labels=["Controls", "Patients"])
ttest_res = ttest_ind(controls, patients)
plt.text(0, np.max(controls)+0.05*np.median(controls), f"p-value: \n{np.round(ttest_res[1], 5)}")
