import { magma } from "../magma";

test("clamps out of range", () => {
  expect(magma(-1)).toEqual([0, 0, 4]);          // first anchor
  expect(magma(2)).toEqual([252, 253, 191]);     // last anchor
});

test("returns integer rgb in range", () => {
  const [r, g, b] = magma(0.5);
  for (const c of [r, g, b]) {
    expect(c).toBeGreaterThanOrEqual(0);
    expect(c).toBeLessThanOrEqual(255);
    expect(Number.isInteger(c)).toBe(true);
  }
});
