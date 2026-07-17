import "@testing-library/jest-dom";

// jsdom has no real <canvas> renderer and logs a noisy "Not implemented:
// HTMLCanvasElement.getContext()" console error every time a component
// mounts a real canvas (SpectrogramCanvas, AnalyzingView, ...). Stub it to
// return null — components guard on a null context, so this keeps that
// guarded path under test instead of masking it with a fake context.
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = (() =>
    null) as typeof HTMLCanvasElement.prototype.getContext;
}
