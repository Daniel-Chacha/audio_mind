import { SpectrogramCanvas } from "@/components/SpectrogramCanvas";
import { ConfidenceBars } from "@/components/ConfidenceBars";
import { DISPLAY } from "@/lib/genres";
import type { ClassifyResponse } from "@/lib/types";
import styles from "./PredictionView.module.css";

interface PredictionViewProps {
  result: ClassifyResponse;
  onReset: () => void;
}

export function PredictionView({ result, onReset }: PredictionViewProps) {
  const pct = (result.confidence * 100).toFixed(1);

  return (
    <div className={styles.result}>
      <div className={styles.see}>
        <span className={styles.kicker}>what the model sees</span>
        <div className={styles.seeCanvasWrap}>
          <SpectrogramCanvas data={result.spectrogram.data} />
        </div>
        <span className={styles.axis}>
          128 mel bands × 130 frames · log-power dB
        </span>
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
