"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const COUNTDOWN_SECONDS = 3;

export type MicErrorCode = "mic_denied";

export interface UseRecorderOptions {
  onDone: (blob: Blob) => void;
  onError: (code: MicErrorCode) => void;
}

/** Captures ~3s of mic audio via MediaRecorder with a 3→2→1 countdown. */
export function useRecorder({ onDone, onError }: UseRecorderOptions) {
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);

  const activeRef = useRef(false);
  const mountedRef = useRef(true);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const teardown = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
    activeRef.current = false;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      teardown();
    };
  }, [teardown]);

  const start = useCallback(() => {
    if (activeRef.current) return; // already recording — ignore double-start
    activeRef.current = true;
    chunksRef.current = [];

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        if (!mountedRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          activeRef.current = false;
          return;
        }

        streamRef.current = stream;
        const recorder = new MediaRecorder(stream);
        recorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunksRef.current.push(e.data);
        };

        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
          chunksRef.current = [];
          const wasMounted = mountedRef.current;
          teardown();
          if (wasMounted) {
            setRecording(false);
            setCountdown(0);
            onDone(blob);
          }
        };

        const reducedMotion =
          typeof window !== "undefined" &&
          typeof window.matchMedia === "function" &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        let remaining = COUNTDOWN_SECONDS;
        setRecording(true);
        setCountdown(reducedMotion ? 0 : remaining);
        recorder.start();

        timerRef.current = setInterval(() => {
          remaining -= 1;
          if (remaining <= 0) {
            if (timerRef.current !== null) {
              clearInterval(timerRef.current);
              timerRef.current = null;
            }
            recorderRef.current?.stop();
          } else if (!reducedMotion) {
            setCountdown(remaining);
          }
        }, 1000);
      })
      .catch(() => {
        activeRef.current = false;
        if (mountedRef.current) onError("mic_denied");
      });
  }, [onDone, onError, teardown]);

  return { recording, countdown, start };
}
