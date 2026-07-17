import { render, screen, fireEvent } from "@testing-library/react";
import { PredictionView } from "../PredictionView";
import { describe, it, expect, vi } from "vitest";
import type { ClassifyResponse } from "@/lib/types";

const PROBABILITIES = [
  { genre: "blues", prob: 0.01 },
  { genre: "classical", prob: 0.01 },
  { genre: "country", prob: 0.01 },
  { genre: "disco", prob: 0.02 },
  { genre: "hiphop", prob: 0.87 },
  { genre: "jazz", prob: 0.01 },
  { genre: "metal", prob: 0.02 },
  { genre: "pop", prob: 0.02 },
  { genre: "reggae", prob: 0.01 },
  { genre: "rock", prob: 0.02 },
];

function makeGrid(bands: number, frames: number): number[][] {
  return Array.from({ length: bands }, () =>
    Array.from({ length: frames }, () => Math.random() * 255)
  );
}

const FIXTURE: ClassifyResponse = {
  genre: "hiphop",
  confidence: 0.87,
  probabilities: PROBABILITIES,
  spectrogram: { bands: 128, frames: 130, data: makeGrid(128, 130) },
  meta: { segments: 1, duration_s: 3, sample_rate: 22050 },
};

describe("PredictionView", () => {
  it("renders the verdict, confidence, and spectrogram caption", () => {
    render(<PredictionView result={FIXTURE} onReset={() => {}} />);
    expect(screen.getByRole("heading", { name: "HIP-HOP" })).toBeInTheDocument();
    expect(screen.getByText("87.0%")).toBeInTheDocument();
    expect(screen.getByText("what the model sees")).toBeInTheDocument();
    expect(
      screen.getByText("128 mel bands × 130 frames · log-power dB")
    ).toBeInTheDocument();
  });

  it("calls onReset when Analyze another is clicked", () => {
    const onReset = vi.fn();
    render(<PredictionView result={FIXTURE} onReset={onReset} />);
    fireEvent.click(screen.getByRole("button", { name: /analyze another/i }));
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
