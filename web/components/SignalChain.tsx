"use client";
import styles from "./SignalChain.module.css";

export function SignalChain({ active }: { active: "input" | "analyze" | "result" }) {
  return (
    <nav className={styles.chain} aria-label="Signal chain">
      <span className={`${styles.stage} ${active === "input" ? styles.on : ""}`}>
        waveform
      </span>
      <span className={styles.arr} aria-hidden="true">→</span>
      <span className={`${styles.stage} ${active === "analyze" ? styles.on : ""}`}>
        mel spectrogram
      </span>
      <span className={styles.arr} aria-hidden="true">→</span>
      <span className={`${styles.stage} ${active === "analyze" ? styles.on : ""}`}>
        cnn
      </span>
      <span className={styles.arr} aria-hidden="true">→</span>
      <span className={`${styles.stage} ${active === "result" ? styles.on : ""}`}>
        genre
      </span>
    </nav>
  );
}
