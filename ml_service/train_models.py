import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import generate_security_dataset

def train_and_evaluate_all():
    print("=== Iniciando proceso de entrenamiento y evaluación de modelos ML / Deep Learning ===")
    
    # 1. Carga de Dataset
    df = generate_security_dataset(n_samples=1600, random_state=42)
    feature_cols = [
        "motion_level", "ai_confidence", "person_count", "vehicle_count",
        "duration_seconds", "is_night", "zone_risk_level", "anomaly_score"
    ]
    target_col = "threat_level"
    
    X = df[feature_cols].values
    y_raw = df[target_col].values
    
    # Codificación de clases
    label_encoder = LabelEncoder()
    # Forzamos orden explícito de clases
    classes = ["Baja", "Media", "Alta", "Critica"]
    label_encoder.fit(classes)
    y = label_encoder.transform(y_raw)
    
    # 2. División Train/Test (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Normalización de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
    
    # 3. Definición de los 3 Algoritmos
    models = {
        "RandomForest": {
            "name": "Random Forest Classifier",
            "type": "Tree Ensemble",
            "model": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
        },
        "GradientBoosting": {
            "name": "Gradient Boosting Classifier",
            "type": "Boosting Ensemble",
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        },
        "MultiLayerPerceptron": {
            "name": "Red Neuronal MLP (Deep Learning)",
            "type": "Multi-Layer Perceptron",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, activation='relu', random_state=42)
        }
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_f1 = -1.0
    best_model_key = None

    for key, info in models.items():
        clf = info["model"]
        print(f"\nEntrenando y evaluando: {info['name']}...")
        
        # A) Entrenamiento en Train Set
        clf.fit(X_train_scaled, y_train)
        
        # B) Evaluación en Test Set
        y_pred = clf.predict(X_test_scaled)
        acc_test = accuracy_score(y_test, y_pred)
        prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
        
        # C) Validación Cruzada (5-Fold CV)
        cv_scores = cross_val_score(clf, X_all_scaled, y, cv=kf, scoring="accuracy")
        cv_f1_scores = cross_val_score(clf, X_all_scaled, y, cv=kf, scoring="f1_macro")
        
        metrics = {
            "key": key,
            "name": info["name"],
            "type": info["type"],
            "train_test": {
                "accuracy": round(float(acc_test), 4),
                "precision": round(float(prec_test), 4),
                "recall": round(float(rec_test), 4),
                "f1_score": round(float(f1_test), 4),
                "confusion_matrix": cm.tolist()
            },
            "cross_validation": {
                "folds": 5,
                "scores": [round(float(s), 4) for s in cv_scores],
                "mean_accuracy": round(float(np.mean(cv_scores)), 4),
                "std_accuracy": round(float(np.std(cv_scores)), 4),
                "mean_f1": round(float(np.mean(cv_f1_scores)), 4),
                "std_f1": round(float(np.std(cv_f1_scores)), 4)
            }
        }
        
        results[key] = metrics
        print(f" -> Accuracy Test: {acc_test*100:.2f}% | F1 Test: {f1_test*100:.2f}%")
        print(f" -> 5-Fold CV Mean Acc: {np.mean(cv_scores)*100:.2f}% (± {np.std(cv_scores)*100:.2f}%)")
        
        if f1_test > best_f1:
            best_f1 = f1_test
            best_model_key = key

    # 4. Marcado del mejor modelo seleccionado
    for key in results:
        results[key]["is_selected"] = (key == best_model_key)
        
    print(f"\n[MEJOR MODELO] Modelo seleccionado por mejor F1-Score: {models[best_model_key]['name']} ({best_f1*100:.2f}%)")
    
    # 5. Exportar artefactos del modelo y resultados
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Guardar cada uno de los 3 modelos entrenados para permitir conmutación dinámica
    for key, info in models.items():
        joblib.dump(info["model"], os.path.join(base_dir, f"model_{key}.joblib"))
        
    selected_clf = models[best_model_key]["model"]
    joblib.dump(selected_clf, os.path.join(base_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(base_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(base_dir, "label_encoder.joblib"))
    
    output_meta = {
        "dataset_samples": len(df),
        "classes": list(classes),
        "selected_model": best_model_key,
        "selected_model_name": models[best_model_key]["name"],
        "feature_names": feature_cols,
        "models": results
    }
    
    with open(os.path.join(base_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2, ensure_ascii=False)
        
    print(f"Artefactos guardados exitosamente en {base_dir}")
    return output_meta

if __name__ == "__main__":
    train_and_evaluate_all()
