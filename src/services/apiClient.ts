import { firebaseAuth } from "../lib/firebase";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:4000").replace(/\/$/, "");

async function getAuthHeaders() {
  const user = firebaseAuth.currentUser;
  if (!user) return {};

  const token = await user.getIdToken();
  return {
    Authorization: `Bearer ${token}`,
  };
}

export async function apiGet<T>(path: string): Promise<T> {
  const authHeaders = (await getAuthHeaders()) as Record<string, string>;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      ...authHeaders,
      "ngrok-skip-browser-warning": "true",
    },
  });

  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export async function apiSend<T>(path: string, init: RequestInit): Promise<T> {
  const authHeaders = (await getAuthHeaders()) as Record<string, string>;
  const customHeaders = (init.headers ?? {}) as Record<string, string>;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
      ...authHeaders,
      ...customHeaders,
    },
    ...init,
  });

  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}
