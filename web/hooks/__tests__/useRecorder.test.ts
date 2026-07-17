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

test("reduced motion: still records ~3s but keeps countdown at 0 (no visible ticks)", async () => {
  // jsdom lacks matchMedia, so stub it to report reduced motion.
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true }));
  const onDone = vi.fn();
  const { result } = renderHook(() => useRecorder({ onDone, onError: () => {} }));
  await act(async () => { result.current.start(); });
  expect(result.current.recording).toBe(true);
  expect(result.current.countdown).toBe(0);
  await act(async () => { vi.advanceTimersByTime(1000); });
  expect(result.current.countdown).toBe(0); // no visible 3→2→1 while reduced
  await act(async () => { vi.advanceTimersByTime(2000); });
  expect(onDone).toHaveBeenCalledWith(expect.any(Blob));
});

test("unmount mid-recording cleans up: no onDone, no state update, no throw", async () => {
  const onDone = vi.fn();
  const { result, unmount } = renderHook(() => useRecorder({ onDone, onError: () => {} }));
  await act(async () => { result.current.start(); });
  await act(async () => { vi.advanceTimersByTime(1500); }); // mid-countdown
  unmount();
  await act(async () => { vi.advanceTimersByTime(5000); }); // would fire if interval leaked
  expect(onDone).not.toHaveBeenCalled();
});

test("mic denied surfaces onError('mic_denied') and never calls onDone", async () => {
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn().mockRejectedValue(new Error("denied")) } });
  const onDone = vi.fn();
  const onError = vi.fn();
  const { result } = renderHook(() => useRecorder({ onDone, onError }));
  await act(async () => { result.current.start(); });
  expect(onError).toHaveBeenCalledWith("mic_denied");
  expect(onDone).not.toHaveBeenCalled();
  expect(result.current.recording).toBe(false);
});
