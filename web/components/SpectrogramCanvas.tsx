"use client";
import { useEffect, useRef } from "react";
import { gridToImageData } from "@/lib/spectro";
import styles from "./SpectrogramCanvas.module.css";

export function SpectrogramCanvas({ data }: { data: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !data.length) return;
    const rect = cv.getBoundingClientRect();
    const cellW = Math.max(1, Math.round(rect.width / data[0].length));
    const cellH = Math.max(1, Math.round(rect.height / data.length));
    const { width, height, rgba } = gridToImageData(data, cellW, cellH);
    cv.width = width;
    cv.height = height;
    // jsdom's canvas has no 2d context (see vitest.setup.ts), so this is a
    // deliberate no-op there rather than a throw.
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  }, [data]);
  return <canvas ref={ref} className={styles.canvas} />;
}
