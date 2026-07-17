/* eslint-disable @typescript-eslint/no-explicit-any -- verbatim fake MediaRecorder stub from task brief */
import { renderHook, act } from "@testing-library/react";
import { useRecorder } from "../useRecorder";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

beforeEach(() => {
  vi.useFakeTimers();
  const chunks = [new Blob(["a"])];
  class FakeRec { ondataavailable: any; onstop: any; state = "inactive";
    start() { this.state = "recording"; this.ondataavailable?.({ data: chunks[0] }); }
    stop() { this.state = "inactive"; this.onstop?.(); } }
  vi.stubGlobal("MediaRecorder", FakeRec as any);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop() {} }] }) } });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

test("counts down and returns a blob", async () => {
  const onDone = vi.fn();
  const { result } = renderHook(() => useRecorder({ onDone, onError: () => {} }));
  await act(async () => { result.current.start(); });
  await act(async () => { vi.advanceTimersByTime(3000); });
  expect(onDone).toHaveBeenCalledWith(expect.any(Blob));
});
