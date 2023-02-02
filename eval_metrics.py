from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive


def calculate_overall_auc(df, model_name, TOXICITY_COLUMN):
    true_labels = df[TOXICITY_COLUMN]>0.5
    predicted_labels = df[model_name]
#     print(type(predicted_labels), type(true_labels))
#     print(predicted_labels.shape, true_labels.shape)
#     with open("debug.pkl","wb") as f:
#         import pickle
#         pickle.dump((predicted_labels, true_labels), f)
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
#     print("series", series)
#     print("len:", len(series), "total", total)
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
#     print(SUBGROUP_AUC, power_mean(bias_df[SUBGROUP_AUC], POWER), len(bias_df[SUBGROUP_AUC]))
#     print("==================================================================================")
#     print(BPSN_AUC, power_mean(bias_df[BPSN_AUC], POWER))
#     print("==================================================================================")
    
#     print(BNSP_AUC, power_mean(bias_df[BNSP_AUC], POWER), len(bias_df[BNSP_AUC]))
#     print("==================================================================================")
    
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
#     print("bias_score:", bias_score, "overall_auc", overall_auc)
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
#     print("label", subgroup_examples[label]>0.5)
#     print("pred",subgroup_examples[model_name])
#     print("auc:", compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name]))
#     print("***********************************************************************")
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,subgroups,model,label_col,include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    print("building bias df...")
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
#         print("##################################################################")
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
#         print("subgroup:", subgroup, "record:", record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)