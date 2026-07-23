"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { magma } from "@/lib/magma";
import type { ClassifyPhase } from "@/lib/classifier";
import styles from "./AnalyzingView.module.css";

const W = 130; // frames (time)
const H = 128; // mel bands
const SWEEP_MS = 2100;
const SCAN_MS = 1800; // one pass of the post-build scan line

// AnalyzingView never knows the eventual genre (the API call is still in
// flight), so the build uses one neutral, pleasant-looking spectral profile
// rather than a genre-true one — it's illustrative, not diagnostic.
const PROFILE = { lowE: 0.58, highE: 0.46, sub: 0.3, harm: 0.32, hmid: 0.38, onsets: 7, ob: 0.68 };

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

// Ported from the prototype's genGrid(genre) — same shape, one fixed profile.
function genGrid(): Float32Array {
  const seed = Math.random() * Math.PI * 2;
  const grid = new Float32Array(W * H);
  const beats: number[] = [];
  const step = Math.max(6, Math.round(26 - PROFILE.onsets * 1.7));
  for (
    let c = Math.floor(Math.random() * step);
    c < W;
    c += step + (Math.floor(Math.random() * 4) - 2)
  ) {
    beats.push(c);
  }

  for (let c = 0; c < W; c++) {
    let onset = 0;
    for (const b of beats) {
      const d = c - b;
      onset = Math.max(onset, Math.exp(-(d * d) / 3.2));
    }
    const swing = 0.62 + 0.38 * Math.sin(c * 0.11 + seed);
    for (let r = 0; r < H; r++) {
      const band = r / (H - 1); // 0 = top (high freq), 1 = bottom (low freq)
      let v = lerp(PROFILE.highE, PROFILE.lowE, band);
      v += PROFILE.sub * Math.pow(band, 3);
      v +=
        PROFILE.harm *
        Math.exp(-Math.pow((band - PROFILE.hmid) / 0.05, 2)) *
        (0.6 + 0.4 * Math.sin(c * 0.28 + seed));
      v += PROFILE.harm * 0.55 * Math.exp(-Math.pow((band - PROFILE.hmid * 0.62) / 0.045, 2));
      v += onset * PROFILE.ob * (0.42 + 0.58 * (1 - band * 0.5));
      v += (Math.random() - 0.5) * 0.11;
      v *= swing * 0.5 + 0.6;
      grid[c * H + r] = v;
    }
  }

  let min = Infinity;
  let max = -Infinity;
  for (const v of grid) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const inv = 1 / (max - min + 1e-6);
  for (let i = 0; i < grid.length; i++) {
    grid[i] = Math.pow((grid[i] - min) * inv, 0.82);
  }
  return grid;
}

function fitCanvas(cv: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = cv.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  cv.width = width;
  cv.height = height;
  return { ctx: cv.getContext("2d"), width, height };
}

// Ported from the prototype's drawSpectro(cv, grid, upto). `upto === null`
// draws the full grid (no sweep highlight); jsdom's canvas has no 2d
// context, so this is a deliberate no-op there rather than a throw.
function drawSpectro(cv: HTMLCanvasElement, grid: Float32Array, upto: number | null) {
  const { ctx, width: cw, height: ch } = fitCanvas(cv);
  if (!ctx) return;
  const colW = cw / W;
  const rowH = ch / H;
  ctx.fillStyle = "#05060a";
  ctx.fillRect(0, 0, cw, ch);

  const maxC = upto == null ? W : upto;
  for (let c = 0; c < maxC; c++) {
    for (let r = 0; r < H; r++) {
      const [rr, gg, bb] = magma(grid[c * H + r]);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.fillRect(c * colW, r * rowH, colW + 1, rowH + 1);
    }
  }

  if (upto != null && upto < W) {
    ctx.fillStyle = "rgba(252,211,77,.9)";
    ctx.fillRect(upto * colW, 0, 2, ch);
    ctx.fillStyle = "rgba(252,211,77,.12)";
    ctx.fillRect(upto * colW - 14, 0, 14, ch);
  }
}

/** Copy the finished spectrogram so the scan loop can blit instead of redrawing. */
function snapshot(cv: HTMLCanvasElement): HTMLCanvasElement | null {
  const off = document.createElement("canvas");
  off.width = cv.width;
  off.height = cv.height;
  const ctx = off.getContext("2d");
  if (!ctx) return null; // jsdom
  ctx.drawImage(cv, 0, 0);
  return off;
}

/**
 * A scan line sweeping the finished spectrogram, drawn while the model is
 * still downloading or running. Without it the view freezes on a full progress
 * bar and reads as hung.
 */
function drawScan(cv: HTMLCanvasElement, off: HTMLCanvasElement, t: number) {
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(off, 0, 0);
  const x = t * cv.width;
  const trail = Math.max(40, cv.width * 0.09);
  const grad = ctx.createLinearGradient(x - trail, 0, x, 0);
  grad.addColorStop(0, "rgba(252,211,77,0)");
  grad.addColorStop(1, "rgba(252,211,77,0.18)");
  ctx.fillStyle = grad;
  ctx.fillRect(x - trail, 0, trail, cv.height);
  ctx.fillStyle = "rgba(252,211,77,0.8)";
  ctx.fillRect(x, 0, 2, cv.height);
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

// What the pipeline is actually doing, so the status is truthful rather than
// a decorative "loading". Verb + subject keeps the head's emphasis styling.
const PHASE_TEXT: Record<ClassifyPhase, [string, string]> = {
  loading: ["loading", "the model"],
  decoding: ["decoding", "audio"],
  spectrogram: ["extracting", "mel spectrogram"],
  inferring: ["running", "the cnn"],
};

export function AnalyzingView({
  sourceLabel,
  phase = "spectrogram",
}: {
  sourceLabel: string;
  phase?: ClassifyPhase;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const reduced = useMemo(() => prefersReducedMotion(), []);
  // Lazy init (not a setState-in-effect) so the reduced-motion case renders
  // its final readout on the very first paint, with no animation frame.
  const [frame, setFrame] = useState(() => (prefersReducedMotion() ? W : 0));

  useEffect(() => {
    const grid = genGrid();
    const cv = canvasRef.current;

    if (reduced) {
      if (cv) drawSpectro(cv, grid, null);
      return;
    }

    let start: number | null = null;
    let off: HTMLCanvasElement | null = null;
    let captured = false;

    const step = (ts: number) => {
      if (start == null) start = ts;
      const elapsed = ts - start;

      if (elapsed < SWEEP_MS) {
        const col = Math.round((elapsed / SWEEP_MS) * W);
        if (cv) drawSpectro(cv, grid, col);
        setFrame(col);
      } else {
        if (!captured) {
          if (cv) {
            drawSpectro(cv, grid, null);
            off = snapshot(cv);
          }
          setFrame(W);
          captured = true;
        }
        // The build is done but the model may still be downloading or running,
        // so keep sweeping — a static full bar reads as a hang.
        if (cv && off) drawScan(cv, off, ((elapsed - SWEEP_MS) % SCAN_MS) / SCAN_MS);
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [reduced]);

  const pct = Math.min(100, (frame / W) * 100);
  const frameLabel = String(Math.min(frame, W)).padStart(3, "0");
  const [verb, subject] = PHASE_TEXT[phase];
  // Once the illustrative build finishes, real progress is unknowable — so show
  // an indeterminate meter rather than a full bar that implies "done".
  const indeterminate = !reduced && frame >= W;

  return (
    <div className={styles.analyzer}>
      <div className={styles.anHead}>
        <span>
          {verb} <b>{subject}</b>
          <span className={styles.dots} aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
        </span>
        <span>— {sourceLabel}</span>
      </div>
      <div className={styles.anCanvasWrap}>
        <canvas ref={canvasRef} className={styles.spectroLive} />
      </div>
      <div className={styles.anFoot}>
        <span className={styles.anNums}>
          frame {frameLabel} / 130 · 128 bands · 22.05 kHz
        </span>
        <span
          className={`${styles.meter} ${indeterminate ? styles.meterIndeterminate : ""}`}
          role="progressbar"
          aria-label={`${verb} ${subject}`}
          {...(indeterminate ? {} : { "aria-valuenow": Math.round(pct), "aria-valuemin": 0, "aria-valuemax": 100 })}
        >
          <i style={indeterminate ? undefined : { width: `${pct}%` }} />
        </span>
      </div>
    </div>
  );
}
