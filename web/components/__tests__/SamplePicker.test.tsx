import { render, screen, fireEvent } from "@testing-library/react";
import { SamplePicker } from "../SamplePicker";
import type { Sample } from "@/lib/samples";

const SAMPLES: Sample[] = [
  { genre: "metal", label: "METAL", filename: "metal.wav", url: "/samples/metal.wav" },
  { genre: "jazz", label: "JAZZ", filename: "jazz.wav", url: "/samples/jazz.wav" },
];

test("renders a button per sample and emits the picked one", () => {
  const onPick = vi.fn();
  render(<SamplePicker samples={SAMPLES} onPick={onPick} />);
  fireEvent.click(screen.getByRole("button", { name: "METAL" }));
  expect(onPick).toHaveBeenCalledWith(SAMPLES[0]);
});

test("disables the buttons when disabled", () => {
  render(<SamplePicker samples={SAMPLES} onPick={() => {}} disabled />);
  expect(screen.getByRole("button", { name: "JAZZ" })).toBeDisabled();
});
