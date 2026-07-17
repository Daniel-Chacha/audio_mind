"use client";
import { useRef } from "react";
import styles from "./Dropzone.module.css";

export function Dropzone({ onFile, onRecord, disabled }: {
  onFile: (f: File) => void; onRecord: () => void; disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div
      className={styles.dropzone}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
    >
      <h1 className={styles.title}>Drop a track to classify</h1>
      <p className={styles.sub}>Feed AudioMind a clip and it returns the genre from a 3-second window.</p>
      <div className={styles.actions}>
        <button className="btn btn-primary" disabled={disabled}
          onClick={() => input.current?.click()}>Upload audio</button>
        <span className="or">or</span>
        <button className="btn btn-ghost" disabled={disabled} onClick={onRecord}>Record 3s</button>
        <input ref={input} data-testid="file-input" type="file" accept="audio/*" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      </div>
      <div className={styles.foot}>wav · mp3 · flac — resampled to 22.05 kHz mono</div>
    </div>
  );
}
