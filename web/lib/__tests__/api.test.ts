import { classify, ApiError } from "../api";
import { afterEach, expect, test, vi } from "vitest";
afterEach(() => vi.restoreAllMocks());

test("returns parsed body on 200", async () => {
  const body = { genre: "jazz", confidence: 0.7, probabilities: [], spectrogram: {bands:128,frames:130,data:[]}, meta:{segments:2,duration_s:6,sample_rate:22050} };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => body }));
  const r = await classify(new Blob(["x"]), "c.wav");
  expect(r.genre).toBe("jazz");
});

test("maps server error to ApiError with code", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ error: "clip_too_short", message: "Clips need to be at least 3 seconds." }) }));
  await expect(classify(new Blob(["x"]))).rejects.toMatchObject({ code: "clip_too_short" });
});

test("maps network failure to ApiError('network')", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
  await expect(classify(new Blob(["x"]))).rejects.toMatchObject({ code: "network" });
});
