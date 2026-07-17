import { magma } from "./magma";

export function gridToImageData(
  data: number[][],
  cellW: number,
  cellH: number
) {
  const rows = data.length,
    cols = data[0].length;
  const width = cols * cellW,
    height = rows * cellH;
  const rgba = new Uint8ClampedArray(width * height * 4);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const [rr, gg, bb] = magma(data[r][c] / 255);
      for (let dy = 0; dy < cellH; dy++) {
        for (let dx = 0; dx < cellW; dx++) {
          const x = c * cellW + dx,
            y = r * cellH + dy,
            i = (y * width + x) * 4;
          rgba[i] = rr;
          rgba[i + 1] = gg;
          rgba[i + 2] = bb;
          rgba[i + 3] = 255;
        }
      }
    }
  }

  return { width, height, rgba };
}
