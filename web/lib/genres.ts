export const GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"];
export const DISPLAY: Record<string,string> = Object.fromEntries(
  GENRES.map(g => [g, g === "hiphop" ? "HIP-HOP" : g.toUpperCase()])
);
