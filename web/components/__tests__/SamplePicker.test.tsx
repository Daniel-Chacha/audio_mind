import { render, screen, fireEvent } from "@testing-library/react";
import { SamplePicker } from "../SamplePicker";
import type { Sample } from "@/lib/samples";

const SAMPLES: Sample[] = [
  { genre: "metal", label: "METAL", filename: "metal.wav", url: "/samples/metal.wav" },
  { genre: "jazz", label: "JAZZ", filename: "jazz.wav", url: "/samples/jazz.wav" },
];

// jsdom implements no media playback, so the <audio> element's transport is stubbed.
const play = vi.fn<() => Promise<void>>(() => Promise.resolve());
const pause = vi.fn<() => void>(() => {});

beforeEach(() => {
  play.mockClear();
  pause.mockClear();
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(play);
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(pause);
});
afterEach(() => vi.restoreAllMocks());

test("renders a classify button per sample and emits the picked one", () => {
  const onPick = vi.fn();
  render(<SamplePicker samples={SAMPLES} onPick={onPick} />);
  fireEvent.click(screen.getByRole("button", { name: "METAL" }));
  expect(onPick).toHaveBeenCalledWith(SAMPLES[0]);
});

test("disables the buttons when disabled", () => {
  render(<SamplePicker samples={SAMPLES} onPick={() => {}} disabled />);
  expect(screen.getByRole("button", { name: "JAZZ" })).toBeDisabled();
  expect(screen.getByRole("button", { name: /play the metal sample/i })).toBeDisabled();
});

test("play button starts the clip and flips to pause", async () => {
  render(<SamplePicker samples={SAMPLES} onPick={() => {}} />);
  const playBtn = screen.getByRole("button", { name: /play the metal sample/i });
  expect(playBtn).toHaveAttribute("aria-pressed", "false");

  fireEvent.click(playBtn);
  expect(play).toHaveBeenCalled();

  // Once playing, the same control offers pause.
  const pauseBtn = await screen.findByRole("button", { name: /pause the metal sample/i });
  expect(pauseBtn).toHaveAttribute("aria-pressed", "true");

  fireEvent.click(pauseBtn);
  expect(pause).toHaveBeenCalled();
  expect(
    await screen.findByRole("button", { name: /play the metal sample/i }),
  ).toBeInTheDocument();
});

test("picking a sample stops playback so it doesn't run over the analysis", async () => {
  const onPick = vi.fn();
  render(<SamplePicker samples={SAMPLES} onPick={onPick} />);
  fireEvent.click(screen.getByRole("button", { name: /play the metal sample/i }));
  await screen.findByRole("button", { name: /pause the metal sample/i });

  fireEvent.click(screen.getByRole("button", { name: "METAL" }));
  expect(pause).toHaveBeenCalled();
  expect(onPick).toHaveBeenCalledWith(SAMPLES[0]);
});
