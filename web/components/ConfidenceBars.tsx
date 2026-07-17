"use client";

import { useMemo } from "react";
import { rankProbabilities } from "@/lib/rank";
import { DISPLAY } from "@/lib/genres";
import styles from "./ConfidenceBars.module.css";

interface ConfidenceBarsProps {
  probabilities: { genre: string; prob: number }[];
}

export function ConfidenceBars({ probabilities }: ConfidenceBarsProps) {
  const ranked = useMemo(() => rankProbabilities(probabilities), [probabilities]);

  return (
    <div className={styles.bars}>
      <div className={styles.barsHead}>Confidence · All 10 Genres</div>
      {ranked.map((item, idx) => {
        const pct = item.prob * 100;
        const delayMs = 120 + idx * 70;

        return (
          <div
            key={item.genre}
            className={`${styles.row} ${item.win ? styles.win : ""}`}
            tabIndex={0}
          >
            <span className={styles.name}>{DISPLAY[item.genre]}</span>
            <span className={styles.track}>
              <i
                style={{
                  width: `${Math.max(2, pct)}%`,
                  transitionDelay: `${delayMs}ms`,
                }}
              />
            </span>
            <span className={styles.val}>{pct.toFixed(1)}</span>
          </div>
        );
      })}
    </div>
  );
}
