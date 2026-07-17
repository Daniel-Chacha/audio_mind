import { render, screen } from "@testing-library/react";
import { AnalyzingView } from "../AnalyzingView";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("AnalyzingView", () => {
  it("renders the mono readout and source label", () => {
    render(<AnalyzingView sourceLabel="song.wav" />);
    expect(screen.getByText("extracting")).toBeInTheDocument();
    expect(screen.getByText("mel spectrogram")).toBeInTheDocument();
    expect(screen.getByText(/song\.wav/)).toBeInTheDocument();
  });

  it("reduced motion: renders the full readout immediately, no rAF sweep", () => {
    // jsdom lacks matchMedia; stub it to report reduced motion.
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({ matches: true })
    );
    const rafSpy = vi.spyOn(window, "requestAnimationFrame");

    render(<AnalyzingView sourceLabel="song.wav" />);

    expect(screen.getByText(/frame 130 \/ 130/)).toBeInTheDocument();
    expect(rafSpy).not.toHaveBeenCalled();
  });

  it("cleans up the rAF loop on unmount", () => {
    const cafSpy = vi.spyOn(window, "cancelAnimationFrame");
    const { unmount } = render(<AnalyzingView sourceLabel="song.wav" />);
    unmount();
    expect(cafSpy).toHaveBeenCalled();
  });
});
