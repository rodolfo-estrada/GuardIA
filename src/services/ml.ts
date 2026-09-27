import { apiGet, apiSend } from "./apiClient";

export type ModelMetric = {
  key: string;
  name: string;
  type: string;
  is_selected: boolean;
  train_test: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    confusion_matrix: number[][];
  };
  cross_validation: {
    folds: number;
    scores: number[];
    mean_accuracy: number;
    std_accuracy: number;
    mean_f1: number;
    std_f1: number;
  };
};

export type MlEvaluationResults = {
  dataset_samples: number;
  classes: string[];
  selected_model: string;
  selected_model_name: string;
  feature_names: string[];
  models: Record<string, ModelMetric>;
};

export type MlPredictInput = {
  motion_level: number;
  ai_confidence: number;
  person_count: number;
  vehicle_count: number;
  duration_seconds: number;
  is_night: number;
  zone_risk_level: number;
  anomaly_score?: number;
};

export type MlPredictOutput = {
  threat_level: "Baja" | "Media" | "Alta" | "Critica";
  confidence: number;
  probabilities: Record<string, number>;
  selected_model_name: string;
  execution_time_ms: number;
};

export async function fetchMlEvaluation(): Promise<MlEvaluationResults> {
  return apiGet<MlEvaluationResults>("/ml/evaluation");
}

export async function predictMlThreat(input: MlPredictInput): Promise<MlPredictOutput> {
  return apiSend<MlPredictOutput>("/ml/predict", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function selectMlModel(modelKey: string): Promise<{ ok: boolean; selected_model: string }> {
  return apiSend<{ ok: boolean; selected_model: string }>("/ml/select-model", {
    method: "POST",
    body: JSON.stringify({ modelKey }),
  });
}
