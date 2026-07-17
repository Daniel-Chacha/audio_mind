import type { ClassifyResponse } from "./types";
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) { super(message); this.code = code; }
}

export async function classify(file: Blob, filename = "clip.wav"): Promise<ClassifyResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  let res: Response;
  try {
    res = await fetch(`${BASE}/classify`, { method: "POST", body: form });
  } catch {
    throw new ApiError("network", "Couldn't reach the model. Is the service running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.error ?? "server_error", body.message ?? "Something went wrong.");
  }
  return res.json();
}
