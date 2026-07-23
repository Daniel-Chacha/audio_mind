"use client";
import { useEffect, useRef, useState } from "react";
import type { Sample } from "@/lib/samples";
import styles from "./SamplePicker.module.css";

/**
 * The bundled demo clips. Each chip carries two controls: a play/pause toggle to
 * hear the clip, and the genre label to classify it. They're separate buttons
 * because a button can't nest inside another button.
 */
export function SamplePicker({
  samples,
  onPick,
  disabled,
}: {
  samples: Sample[];
  onPick: (s: Sample) => void;
  disabled?: boolean;
}) {
  const [playing, setPlaying] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const audio = new Audio();
    audio.preload = "none";
    audioRef.current = audio;
    const clear = () => setPlaying(null);
    audio.addEventListener("ended", clear);
    audio.addEventListener("error", clear);
    return () => {
      audio.removeEventListener("ended", clear);
      audio.removeEventListener("error", clear);
      audio.pause();
      audioRef.current = null;
    };
  }, []);

  function stop() {
    audioRef.current?.pause();
    setPlaying(null);
  }

  function togglePlay(s: Sample) {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing === s.genre) {
      stop();
      return;
    }
    // Only one clip plays at a time.
    audio.pause();
    audio.src = s.url;
    audio.currentTime = 0;
    void audio
      .play()
      .then(() => setPlaying(s.genre))
      .catch(() => setPlaying(null));
  }

  function pick(s: Sample) {
    stop(); // don't leave a clip playing over the analysis
    onPick(s);
  }

  return (
    <div className={styles.wrap}>
      <span className={styles.label}>Or try a sample</span>
      <div className={styles.chips}>
        {samples.map((s) => {
          const isPlaying = playing === s.genre;
          return (
            <span
              key={s.genre}
              className={`${styles.chip} ${isPlaying ? styles.chipPlaying : ""}`}
            >
              <button
                type="button"
                className={`${styles.play} ${isPlaying ? styles.playActive : ""}`}
                disabled={disabled}
                aria-pressed={isPlaying}
                aria-label={`${isPlaying ? "Pause" : "Play"} the ${s.label} sample`}
                onClick={() => togglePlay(s)}
              >
                {isPlaying ? "❚❚" : "▶"}
              </button>
              <button
                type="button"
                className={styles.name}
                disabled={disabled}
                onClick={() => pick(s)}
              >
                {s.label}
              </button>
            </span>
          );
        })}
      </div>
    </div>
  );
}
