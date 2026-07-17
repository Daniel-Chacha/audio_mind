export function rankProbabilities(probs: { genre: string; prob: number }[]) {
  const sorted = [...probs].sort((a, b) => b.prob - a.prob);
  const max = sorted.length ? sorted[0].prob : 0;
  return sorted.map((p, i) => ({ ...p, win: i === 0 && p.prob === max }));
}
