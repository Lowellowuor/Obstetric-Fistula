import numpy as np
from sklearn.metrics import confusion_matrix

def clinical_safety_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    if cm.shape[0] < 3:
        return {"urgent_sensitivity": 0.0, "urgent_ppv": 0.0, "false_negative_rate": 1.0}
    urgent_recall = cm[2,2] / (cm[2,:].sum() + 1e-9)
    urgent_precision = cm[2,2] / (cm[:,2].sum() + 1e-9)
    false_negative_rate = 1 - urgent_recall
    return {
        "urgent_sensitivity": urgent_recall,
        "urgent_ppv": urgent_precision,
        "false_negative_rate": false_negative_rate
    }