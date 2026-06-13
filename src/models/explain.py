import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib

def explain_prediction(model_path: str, text: str, embedder, feature_names=None):
    model = joblib.load(model_path)
    emb = embedder.embed([text])[0].reshape(1, -1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(emb)
    # For multi-class, shap_values is list; class 2 (urgent) is index 2
    shap.summary_plot(shap_values[2], emb, feature_names=feature_names, show=False)
    plt.savefig("models_artifacts/shap_urgent_explanation.png")
    return explainer