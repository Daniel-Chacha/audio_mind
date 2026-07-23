"use client";
import { useEffect, useRef, useState } from "react";
import { SpectrogramCanvas } from "@/components/SpectrogramCanvas";
import { ConfidenceBars } from "@/components/ConfidenceBars";
import { DISPLAY } from "@/lib/genres";
import type { ClassifyResponse } from "@/lib/types";
import styles from "./PredictionView.module.css";

interface PredictionViewProps {
  result: ClassifyResponse;
  /** The clip that was classified, so it can be played back. */
  clipUrl?: string | null;
  onReset: () => void;
}

export function PredictionView({ result, clipUrl, onReset }: PredictionViewProps) {
  const pct = (result.confidence * 100).toFixed(1);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  // Drive the playhead from rAF rather than `timeupdate`, which only fires a
  // few times a second and looks choppy across a 6-second clip.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const a = audioRef.current;
      if (a && a.duration > 0) setProgress(a.currentTime / a.duration);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  function toggle() {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      a.pause();
      return;
    }
    void a
      .play()
      .then(() => setPlaying(true))
      .catch(() => setPlaying(false));
  }

  return (
    <div className={styles.result}>
      <div className={styles.see}>
        <span className={styles.kicker}>what the model sees</span>
        <div className={styles.seeCanvasWrap}>
          <SpectrogramCanvas data={result.spectrogram.data} />
          {clipUrl && (
            <div
              className={styles.playhead}
              style={{ left: `${progress * 100}%`, opacity: playing ? 1 : 0 }}
            />
          )}
        </div>
        <div className={styles.seeControls}>
          {clipUrl && (
            <>
              <button
                type="button"
                className={styles.play}
                onClick={toggle}
                aria-pressed={playing}
                aria-label={playing ? "Pause the clip" : "Play the clip"}
              >
                <span className={styles.playIcon} aria-hidden="true">
                  {playing ? "❚❚" : "▶"}
                </span>
                {playing ? "pause" : "play clip"}
              </button>
              <audio
                ref={audioRef}
                src={clipUrl}
                preload="metadata"
                onEnded={() => {
                  setPlaying(false);
                  setProgress(0);
                }}
                onPause={() => setPlaying(false)}
              />
            </>
          )}
          <span className={styles.axis}>128 mel bands × 130 frames · log-power dB</span>
        </div>
      </div>

      <div className={styles.verdict}>
        <span className={styles.eyebrow}>prediction</span>
        <h2 className={styles.genre}>{DISPLAY[result.genre]}</h2>
        <div className={styles.confTop}>
          <b>{pct}%</b> confidence
        </div>
        <ConfidenceBars probabilities={result.probabilities} />
        <div className={styles.resetWrap}>
          <button className="btn btn-ghost" onClick={onReset}>
            ↺ Analyze another
          </button>
        </div>
      </div>
    </div>
  );
}
