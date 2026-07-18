"use client";
import type { Sample } from "@/lib/samples";
import styles from "./SamplePicker.module.css";

export function SamplePicker({
  samples,
  onPick,
  disabled,
}: {
  samples: Sample[];
  onPick: (s: Sample) => void;
  disabled?: boolean;
}) {
  return (
    <div className={styles.wrap}>
      <span className={styles.label}>Or try a sample</span>
      <div className={styles.chips}>
        {samples.map((s) => (
          <button
            key={s.genre}
            type="button"
            className={styles.chip}
            disabled={disabled}
            onClick={() => onPick(s)}
          >
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}
