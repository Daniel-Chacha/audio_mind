"use client";
import { useEffect, useRef, useState } from "react";
import { magma } from "@/lib/magma";
import styles from "./AnalyzingView.module.css";

const W = 130; // frames (time)
const H = 128; // mel bands
const SWEEP_MS = 2100;

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

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function AnalyzingView({ sourceLabel }: { sourceLabel: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  // Lazy init (not a setState-in-effect) so the reduced-motion case renders
  // its final readout on the very first paint, with no animation frame.
  const [frame, setFrame] = useState(() => (prefersReducedMotion() ? W : 0));

  useEffect(() => {
    const grid = genGrid();
    const cv = canvasRef.current;

    if (prefersReducedMotion()) {
      if (cv) drawSpectro(cv, grid, null);
      return;
    }

    let start: number | null = null;
    const step = (ts: number) => {
      if (start == null) start = ts;
      const prog = Math.min(1, (ts - start) / SWEEP_MS);
      const col = Math.round(prog * W);
      if (cv) drawSpectro(cv, grid, col < W ? col : null);
      setFrame(col);
      if (prog < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    };
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const pct = Math.min(100, (frame / W) * 100);
  const frameLabel = String(Math.min(frame, W)).padStart(3, "0");

  return (
    <div className={styles.analyzer}>
      <div className={styles.anHead}>
        <span>
          extracting <b>mel spectrogram</b>
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
        <span className={styles.meter}>
          <i style={{ width: `${pct}%` }} />
        </span>
      </div>
    </div>
  );
}
