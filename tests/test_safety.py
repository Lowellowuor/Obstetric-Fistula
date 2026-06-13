import pytest
import numpy as np
from src.evaluation.clinical_metrics import clinical_safety_metrics

def test_clinical_metrics():
    y_true = np.array([0,1,2,2,2])
    y_pred = np.array([0,1,2,2,1])  # one false negative
    metrics = clinical_safety_metrics(y_true, y_pred)
    assert metrics['urgent_sensitivity'] == 2/3
    assert metrics['false_negative_rate'] == 1/3