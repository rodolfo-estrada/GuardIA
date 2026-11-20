import { useEffect, useState } from "react";
import { collection, limit, onSnapshot, orderBy, query, Timestamp } from "firebase/firestore";
import { firebaseDb } from "../lib/firebase";

type IncidentReport = {
  id: string;
  type: string;
  priority: string;
  description: string;
  ubicacion: string;
  created_at: Timestamp;
};

function formatLocation(value: string) {
  try {
    const { lat, lng } = JSON.parse(value);
    if (typeof lat === "number" && typeof lng === "number") {
      return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    }
  } catch {
    return "Sin ubicación";
  }
  return "Sin ubicación";
}

export default function IncidentReportsPage() {
  const [reports, setReports] = useState<IncidentReport[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const recentReports = query(
      collection(firebaseDb, "incident_reports"),
      orderBy("created_at", "desc"),
      limit(100),
    );
    return onSnapshot(
      recentReports,
      (snapshot) => setReports(snapshot.docs.map((doc) => ({
        ...(doc.data() as Omit<IncidentReport, "id">),
        id: doc.id,
      }))),
      () => setError(true),
    );
  }, []);

  return (
    <section className="space-y-5 text-slate-100">
      <h1 className="text-2xl font-semibold">Reportes de incidentes</h1>
      {error ? (
        <p role="alert" className="text-rose-300">No se pudieron consultar los reportes.</p>
      ) : reports === null ? (
        <p role="status" className="text-slate-300">Cargando reportes…</p>
      ) : reports.length === 0 ? (
        <p className="text-slate-300">Todavía no hay reportes.</p>
      ) : (
        <ul className="space-y-3">
          {reports.map((report) => (
            <li key={report.id} className="rounded-xl border border-white/10 bg-slate-900 p-4">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="font-semibold">{report.type}</h2>
                <span className="text-sm text-slate-300">
                  {report.created_at.toDate().toLocaleString("es-MX")}
                </span>
              </div>
              <p className="mt-2 whitespace-pre-wrap break-words">{report.description}</p>
              <p className="mt-3 text-sm text-slate-300">
                Prioridad: {report.priority} · Ubicación: {formatLocation(report.ubicacion)}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
