/* eslint-disable @typescript-eslint/no-explicit-any -- verbatim reducer test from task brief */
import { reducer } from "../machine";
import type { ClassifyResponse } from "../types";
const idle = { status: "idle" } as const;
test("START → analyzing carries source", () => {
  expect(reducer(idle, { type: "START", source: "a.wav" })).toMatchObject({ status: "analyzing", source: "a.wav" });
});
test("SUCCESS → result", () => {
  const r: any = { genre: "rock" };
  expect(reducer({ status: "analyzing", source: "a.wav" }, { type: "SUCCESS", result: r })).toMatchObject({ status: "result", result: r });
});
test("FAIL → error carries message", () => {
  expect(reducer({ status: "analyzing", source: "a.wav" }, { type: "FAIL", message: "Clips need to be at least 3 seconds." })).toMatchObject({ status: "error", error: "Clips need to be at least 3 seconds." });
});
test("RESET → idle", () => {
  const result: ClassifyResponse = {
    genre: "rock",
    confidence: 0.9,
    probabilities: [],
    spectrogram: { bands: 128, frames: 130, data: [] },
    meta: { segments: 2, duration_s: 6, sample_rate: 22050 },
  };
  expect(reducer({ status: "result", result }, { type: "RESET" })).toEqual({ status: "idle" });
});
