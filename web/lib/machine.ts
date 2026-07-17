import type { ClassifyResponse } from "./types";
export type State =
  | { status: "idle" }
  | { status: "analyzing"; source: string }
  | { status: "result"; result: ClassifyResponse }
  | { status: "error"; error: string };
export type Action =
  | { type: "START"; source: string }
  | { type: "SUCCESS"; result: ClassifyResponse }
  | { type: "FAIL"; message: string }
  | { type: "RESET" };
export function reducer(_state: State, action: Action): State {
  switch (action.type) {
    case "START": return { status: "analyzing", source: action.source };
    case "SUCCESS": return { status: "result", result: action.result };
    case "FAIL": return { status: "error", error: action.message };
    case "RESET": return { status: "idle" };
  }
}
