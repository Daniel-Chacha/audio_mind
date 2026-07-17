/* eslint-disable @typescript-eslint/no-explicit-any -- verbatim reducer test from task brief */
import { reducer } from "../machine";
const idle = { status: "idle" } as const;
test("START → analyzing carries source", () => {
  expect(reducer(idle, { type: "START", source: "a.wav" })).toMatchObject({ status: "analyzing", source: "a.wav" });
});
test("SUCCESS → result", () => {
  const r: any = { genre: "rock" };
  expect(reducer({ status: "analyzing" }, { type: "SUCCESS", result: r })).toMatchObject({ status: "result", result: r });
});
test("FAIL → error carries message", () => {
  expect(reducer({ status: "analyzing" }, { type: "FAIL", message: "Clips need to be at least 3 seconds." })).toMatchObject({ status: "error", error: "Clips need to be at least 3 seconds." });
});
test("RESET → idle", () => {
  expect(reducer({ status: "result" }, { type: "RESET" })).toEqual({ status: "idle" });
});
