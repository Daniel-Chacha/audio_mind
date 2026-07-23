"use client";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { reducer, type State } from "@/lib/machine";
import { classify, warmUp, type ClassifyPhase } from "@/lib/classifier";
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
  // Which real pipeline stage is running, so the analyzing view can say so
  // instead of looking frozen while the model downloads or runs.
  const [phase, setPhase] = useState<ClassifyPhase>("loading");
  // Keep the classified clip around so the result view can play it back.
  // Object URLs must be revoked or they leak for the life of the document.
  const clipUrlRef = useRef<string | null>(null);
  const [clipUrl, setClipUrl] = useState<string | null>(null);

  const setClip = useCallback((blob: Blob | null) => {
    if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
    const url = blob ? URL.createObjectURL(blob) : null;
    clipUrlRef.current = url;
    setClipUrl(url);
  }, []);

  useEffect(
    () => () => {
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
    },
    [],
  );

  // Fetch the model + DSP assets while the user is still looking at the
  // dropzone, so the first classification doesn't pay the download.
  useEffect(() => {
    void warmUp().catch(() => {});
  }, []);

  function handleFile(f: File | Blob, source: string) {
    setPhase("loading");
    setClip(f);
    dispatch({ type: "START", source });
    classify(f, { onPhase: setPhase })
      .then((result) => dispatch({ type: "SUCCESS", result }))
      .catch((e: unknown) =>
        dispatch({ type: "FAIL", message: e instanceof Error ? e.message : "Something went wrong." })
      );
  }

  function handleSample(s: Sample) {
    setPhase("loading");
    dispatch({ type: "START", source: s.filename });
    fetch(s.url)
      .then((r) => {
        if (!r.ok) throw new Error("Couldn't load that sample clip.");
        return r.blob();
      })
      .then((blob) => {
        setClip(blob);
        return classify(blob, { onPhase: setPhase });
      })
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

          {state.status === "analyzing" && <AnalyzingView sourceLabel={state.source} phase={phase} />}

          {state.status === "result" && (
            <PredictionView
              result={state.result}
              clipUrl={clipUrl}
              onReset={() => {
                setClip(null);
                dispatch({ type: "RESET" });
              }}
            />
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
