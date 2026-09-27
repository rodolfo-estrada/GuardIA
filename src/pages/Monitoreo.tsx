import { useEffect, useMemo, useState } from "react";
import { fetchDetectionStatus, startDetection, stopDetection, type DetectionStatus } from "../services/ai";
import { fetchCameras, updateCameraRecord, type CameraRecord } from "../services/cameras";
import CameraConfigModal from "./components/CameraConfigModal";
import CameraFullscreenModal from "./components/CameraFullscreenModal";
import VideoTile from "./components/VideoTile";
import type { Camera } from "./components/VideoTile";

type CameraOverrides = {
  recording: boolean;
  sensitivity: number;
  status: Camera["status"];
};

function toTileCamera(
  camera: CameraRecord,
  status: DetectionStatus | null,
  override?: CameraOverrides,
): Camera {
  return {
    id: camera.id,
    name: camera.name,
    location: camera.zone,
    streamUrl: camera.previewUrl || camera.streamUrl,
    status: override?.status ?? (camera.status === "offline" ? "offline" : "live"),
    latencyMs: status?.cameraId === camera.id && status.running ? 95 : 0,
    lastSeen: status?.cameraId === camera.id && status.running ? "Ahora" : "En espera",
    recording: override?.recording ?? true,
    sensitivity: override?.sensitivity ?? 50,
  };
}

function buildDefaultPlayingState(items: CameraRecord[]) {
  return items.reduce<Record<string, boolean>>((acc, camera) => {
    acc[camera.id] = camera.status !== "offline";
    return acc;
  }, {});
}

export default function Monitoreo() {
  const [query, setQuery] = useState("");
  const [cameraRecords, setCameraRecords] = useState<CameraRecord[]>([]);
  const [playing, setPlaying] = useState<Record<string, boolean>>({});
  const [fullscreenId, setFullscreenId] = useState<string | null>(null);
  const [configId, setConfigId] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, CameraOverrides>>({});
  const [status, setStatus] = useState<DetectionStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void refreshData();

    const intervalId = window.setInterval(() => {
      void refreshStatus();
    }, 4000);

    return () => window.clearInterval(intervalId);
  }, []);

  const cameras = useMemo(
    () => cameraRecords.map((camera) => toTileCamera(camera, status, overrides[camera.id])),
    [cameraRecords, overrides, status],
  );

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return cameras;
    return cameras.filter(
      (cam) =>
        cam.name.toLowerCase().includes(normalized) ||
        cam.location.toLowerCase().includes(normalized),
    );
  }, [cameras, query]);

  const fullscreenCamera = cameras.find((cam) => cam.id === fullscreenId) ?? null;
  const configCamera = cameras.find((cam) => cam.id === configId) ?? null;
  const configRecord = cameraRecords.find((cam) => cam.id === configId) ?? null;

  async function refreshStatus() {
    try {
      const nextStatus = await fetchDetectionStatus();
      setStatus(nextStatus);
      if (nextStatus.running && nextStatus.cameraId) {
        setPlaying({ [nextStatus.cameraId]: true });
      } else {
        setPlaying({});
      }
    } catch {
      setError("No fue posible consultar el estado de la IA.");
    }
  }

  async function refreshData() {
    try {
      setIsLoading(true);
      const [items, nextStatus] = await Promise.all([fetchCameras(), fetchDetectionStatus()]);
      setCameraRecords(items);
      setStatus(nextStatus);
      if (nextStatus.running && nextStatus.cameraId) {
        setPlaying({ [nextStatus.cameraId]: true });
      } else {
        setPlaying(buildDefaultPlayingState(items));
      }
      setError("");
    } catch {
      setError("No fue posible conectar el monitoreo con la API de GuardIA.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleRefresh() {
    await refreshData();
  }

  async function handleTogglePlay(cameraId: string) {
    const target = cameras.find((cam) => cam.id === cameraId);
    if (!target || target.status === "offline") return;

    try {
      setIsSubmitting(true);
      if (playing[cameraId]) {
        await stopDetection();
      } else {
        if (status?.running) {
          await stopDetection();
        }
        await startDetection({
          cameraId,
          confidence: 0.5,
          saveCooldown: 2,
          model: "yolov8n.pt",
        });
      }

      await refreshStatus();
      setError("");
    } catch {
      setError("No fue posible iniciar o detener la deteccion en la IA.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleSaveConfig(patch: Pick<Camera, "recording" | "sensitivity" | "status" | "streamUrl">) {
    if (!configRecord) return;

    try {
      setIsSubmitting(true);
      await updateCameraRecord(configRecord.id, {
        name: configRecord.name,
        zone: configRecord.zone,
        protocol: configRecord.protocol,
        resolution: configRecord.resolution,
        streamUrl: patch.streamUrl?.trim() ?? "",
        aiProfiles: configRecord.aiProfiles,
        retentionDays: configRecord.retentionDays,
      });
      setOverrides((current) => ({
        ...current,
        [configRecord.id]: {
          recording: patch.recording,
          sensitivity: patch.sensitivity,
          status: patch.status,
        },
      }));
      setConfigId(null);
      await refreshData();
      setError("");
    } catch {
      setError("No fue posible actualizar la configuracion de la camara.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <style>{`
        @keyframes monitoringReveal {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes monitoringSweep {
          0% { transform: translateX(-35%); opacity: 0; }
          20% { opacity: 0.45; }
          100% { transform: translateX(140%); opacity: 0; }
        }
      `}</style>

      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <input
          className="w-full rounded-xl border border-cyan-300/20 bg-slate-900/80 px-4 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-400 focus:border-cyan-300 focus:ring-2 focus:ring-cyan-400/30 sm:w-80"
          placeholder="Buscar camara..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:brightness-110 disabled:opacity-60"
          type="button"
          onClick={handleRefresh}
          disabled={isSubmitting}
        >
          Refrescar
        </button>
      </div>

      <div className="mb-4 rounded-2xl border border-cyan-300/20 bg-slate-900/70 px-4 py-3 text-sm text-slate-200">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span>
            Estado IA: {status?.running ? "Activa" : "En espera"}
            {status?.device ? ` | Dispositivo: ${status.device}` : ""}
          </span>
          <span className="text-xs text-slate-400">
            {status?.cameraId ? `Camara activa: ${status.cameraId}` : "Sin deteccion en curso"}
          </span>
        </div>
      </div>

      {error ? (
        <div className="mb-4 rounded-xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {isLoading ? (
          <div className="rounded-2xl border border-slate-700 bg-slate-950/70 p-4 text-sm text-slate-300">
            Cargando camaras desde GuardIA...
          </div>
        ) : (
          filtered.map((cam, index) => (
            <div
              key={cam.id}
              className="relative overflow-hidden rounded-2xl"
              style={{ animation: `monitoringReveal 340ms ease-out ${index * 80}ms both` }}
            >
              <div
                className="pointer-events-none absolute inset-y-0 -left-1/3 z-10 w-1/2 bg-gradient-to-r from-transparent via-cyan-300/15 to-transparent"
                style={{ animation: "monitoringSweep 4.2s linear infinite" }}
              />
              <VideoTile
                cam={cam}
                isPlaying={Boolean(playing[cam.id])}
                onTogglePlay={() => void handleTogglePlay(cam.id)}
                onOpenFullscreen={() => setFullscreenId(cam.id)}
                onOpenConfig={() => setConfigId(cam.id)}
              />
            </div>
          ))
        )}
      </section>

      <CameraFullscreenModal
        camera={fullscreenCamera}
        isPlaying={fullscreenCamera ? Boolean(playing[fullscreenCamera.id]) : false}
        onClose={() => setFullscreenId(null)}
      />

      <CameraConfigModal
        camera={configCamera}
        onClose={() => setConfigId(null)}
        onSave={(patch) => void handleSaveConfig(patch)}
      />
    </>
  );
}
