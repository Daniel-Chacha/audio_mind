import { GENRES, DISPLAY } from "./genres";

export interface Sample {
  genre: string;
  label: string;
  filename: string;
  /** Served from web/public/samples/ at the site root. */
  url: string;
}

// One bundled GTZAN clip per genre, served as a static asset. Clicking one
// runs the exact same classify flow as an uploaded file.
export const SAMPLES: Sample[] = GENRES.map((g) => ({
  genre: g,
  label: DISPLAY[g],
  filename: `${g}.wav`,
  url: `/samples/${g}.wav`,
}));
