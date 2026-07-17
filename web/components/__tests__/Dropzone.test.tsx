import { render, screen, fireEvent } from "@testing-library/react";
import { Dropzone } from "../Dropzone";
import { describe, it, expect, vi } from "vitest";

describe("Dropzone", () => {
  it("emits the chosen file", () => {
    const onFile = vi.fn();
    render(<Dropzone onFile={onFile} onRecord={() => {}} />);
    const input = screen.getByTestId("file-input") as HTMLInputElement;
    const file = new File(["x"], "song.wav", { type: "audio/wav" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("calls onRecord when Record pressed", () => {
    const onRecord = vi.fn();
    render(<Dropzone onFile={() => {}} onRecord={onRecord} />);
    fireEvent.click(screen.getByRole("button", { name: /record/i }));
    expect(onRecord).toHaveBeenCalled();
  });
});
