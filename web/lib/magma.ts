const MAG: [number,number,number][] = [
  [0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
  [229,80,100],[251,135,97],[254,194,135],[252,253,191],
];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function magma(t: number): [number, number, number] {
  if (t < 0) t = 0; if (t > 1) t = 1;
  const s = t * (MAG.length - 1), i = Math.floor(s), f = s - i;
  const a = MAG[i], b = MAG[Math.min(i + 1, MAG.length - 1)];
  return [lerp(a[0],b[0],f)|0, lerp(a[1],b[1],f)|0, lerp(a[2],b[2],f)|0];
}
