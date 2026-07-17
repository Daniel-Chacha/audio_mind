/// <reference types="vitest" />
import { gridToImageData } from "../spectro";

test("produces rgba buffer sized by cells", () => {
  const grid = [[0, 255], [128, 64]];               // 2x2
  const out = gridToImageData(grid, 3, 3);           // 3px per cell → 6x6
  expect(out.width).toBe(6);
  expect(out.height).toBe(6);
  expect(out.rgba.length).toBe(6 * 6 * 4);
  expect(out.rgba[3]).toBe(255);                      // alpha of first pixel
});
