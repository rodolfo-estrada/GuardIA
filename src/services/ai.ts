import { apiGet, apiSend } from "./apiClient";

export type DetectionStatus = {
  running: boolean;
  cameraId: string | null;
  source: string | null;
  startedAt: number | null;
  device: string | null;
  model: string | null;
  personCount?: number;
  motionLevel?: number;
};

export async function fetchDetectionStatus() {
  return apiGet<DetectionStatus>("/status");
}

export async function startDetection(input: {
  cameraId: string;
  model?: string;
  confidence?: number;
  saveCooldown?: number;
}) {
  return apiSend<{ ok: boolean }>("/detections/start", {
    method: "POST",
    body: JSON.stringify({
      cameraId: input.cameraId,
      model: input.model ?? "yolov8n.pt",
      confidence: input.confidence ?? 0.5,
      saveCooldown: input.saveCooldown ?? 2,
    }),
  });
}

export async function stopDetection() {
  return apiSend<{ ok: boolean }>("/detections/stop", {
    method: "POST",
    body: JSON.stringify({}),
  });
}
