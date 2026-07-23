"use client";
import { useEffect, useReducer } from "react";
import { reducer, type State } from "@/lib/machine";
import { classify, warmUp } from "@/lib/classifier";
import { useRecorder } from "@/hooks/useRecorder";
import { SignalChain } from "@/components/SignalChain";
import { Dropzone } from "@/components/Dropzone";
import { SamplePicker } from "@/components/SamplePicker";
import { AnalyzingView } from "@/components/AnalyzingView";
import { PredictionView } from "@/components/PredictionView";
import { SAMPLES, type Sample } from "@/lib/samples";
import styles from "./AnalyzerConsole.module.css";

const STAGE_BY_STATUS: Record<State["status"], "input" | "analyze" | "result"> = {
  idle: "input",
  analyzing: "analyze",
  result: "result",
  error: "input",
};

export function AnalyzerConsole() {
  const [state, dispatch] = useReducer(reducer, { status: "idle" });

  // Fetch the model + DSP assets while the user is still looking at the
  // dropzone, so the first classification doesn't pay the download.
  useEffect(() => {
    void warmUp().catch(() => {});
  }, []);

  function handleFile(f: File | Blob, source: string) {
    dispatch({ type: "START", source });
    classify(f)
      .then((result) => dispatch({ type: "SUCCESS", result }))
      .catch((e: unknown) =>
        dispatch({ type: "FAIL", message: e instanceof Error ? e.message : "Something went wrong." })
      );
  }

  function handleSample(s: Sample) {
    dispatch({ type: "START", source: s.filename });
    fetch(s.url)
      .then((r) => {
        if (!r.ok) throw new Error("Couldn't load that sample clip.");
        return r.blob();
      })
      .then((blob) => classify(blob))
      .then((result) => dispatch({ type: "SUCCESS", result }))
      .catch((e: unknown) =>
        dispatch({ type: "FAIL", message: e instanceof Error ? e.message : "Something went wrong." })
      );
  }

  const recorder = useRecorder({
    onDone: (blob) => handleFile(blob, "mic-clip.wav"),
    onError: () =>
      dispatch({ type: "FAIL", message: "Microphone access was blocked. Allow mic access and try again." }),
  });

  return (
    <div className={styles.app}>
      <header className={styles.rail}>
        <div className={styles.brand}>
          <span className={styles.glyph} aria-hidden="true">◆</span>
          <span className="wordmark">AUDIOMIND</span>
          <span className={styles.tagline}>neural genre analysis</span>
        </div>
        <div className={styles.railMeta}>
          <span className={styles.chip}>GTZAN · 10 genres</span>
          <span className={styles.chip}>CNN · mel-spectrogram</span>
          <span className={styles.status}>
            <i className={styles.led} aria-hidden="true" /> model ready
          </span>
        </div>
      </header>

      <SignalChain active={STAGE_BY_STATUS[state.status]} />

      <main className={styles.console}>
        <div className={styles.stagePanel}>
          {state.status === "idle" && (
            <div className={styles.inputWrap}>
              <Dropzone
                onFile={(f) => handleFile(f, f.name)}
                onRecord={() => recorder.start()}
                disabled={recorder.recording}
              />
              {recorder.recording && (
                <div className={styles.recOverlay}>
                  <div className={styles.recBadge}>
                    <i className={styles.dot} aria-hidden="true" /> recording
                  </div>
                  <div className={styles.recCount}>{recorder.countdown}</div>
                </div>
              )}
              <SamplePicker samples={SAMPLES} onPick={handleSample} disabled={recorder.recording} />
            </div>
          )}

          {state.status === "analyzing" && <AnalyzingView sourceLabel={state.source} />}

          {state.status === "result" && (
            <PredictionView result={state.result} onReset={() => dispatch({ type: "RESET" })} />
          )}

          {state.status === "error" && (
            <div className={styles.errorPanel} role="alert" aria-live="assertive">
              <p className={styles.errorMsg}>{state.error}</p>
              <button className="btn btn-ghost" onClick={() => dispatch({ type: "RESET" })}>
                Try again
              </button>
            </div>
          )}
        </div>
      </main>

      <footer className={styles.ticker} aria-hidden="true">
        <div className={styles.tickerTrack}>
          <span>
            <b>GTZAN dataset</b> 1,000 clips · 10 genres × 100 <em>/</em> <b>segment</b> 3 s window <em>/</em>{" "}
            <b>sample rate</b> 22,050 Hz mono <em>/</em> <b>n_fft</b> 2048 <em>/</em> <b>hop</b> 512 <em>/</em>{" "}
            <b>n_mels</b> 128 <em>/</em> <b>input</b> 128 × 130 × 1 <em>/</em> <b>arch</b> 3× conv-block CNN → GAP →
            dense <em>/</em>{" "}
            <b>blues classical country disco hiphop jazz metal pop reggae rock</b> <em>/</em>
          </span>
        </div>
      </footer>
    </div>
  );
}
