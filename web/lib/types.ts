export interface ClassifyResponse {
  genre: string;
  confidence: number;
  probabilities: { genre: string; prob: number }[];
  spectrogram: { bands: number; frames: number; data: number[][] };
  meta: { segments: number; duration_s: number; sample_rate: number };
}
