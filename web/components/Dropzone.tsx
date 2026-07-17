"use client";
import { useRef, useState } from "react";
import styles from "./Dropzone.module.css";

export function Dropzone({ onFile, onRecord, disabled }: {
  onFile: (f: File) => void; onRecord: () => void; disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  return (
    <div
      className={dragging ? `${styles.dropzone} ${styles.drag}` : styles.dropzone}
      onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
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
