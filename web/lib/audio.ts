/**
 * Browser audio decoding. The Web Audio API decodes wav/mp3/flac natively and
 * resamples for us, so there is no server-side ffmpeg in this app any more.
 */

type AudioCtor = typeof AudioContext;

function audioContextCtor(): AudioCtor {
  const w = window as unknown as { AudioContext?: AudioCtor; webkitAudioContext?: AudioCtor };
  const Ctor = w.AudioContext ?? w.webkitAudioContext;
  if (!Ctor) throw new Error("Web Audio is not available in this browser.");
  return Ctor;
}

/**
 * Decode any browser-supported audio file to mono at `targetSr`.
 *
 * Resampling is done by an OfflineAudioContext rendering at the target rate,
 * which also downmixes to a single channel.
 */
export async function decodeToMono(blob: Blob, targetSr: number): Promise<Float32Array> {
  const bytes = await blob.arrayBuffer();

  const Ctor = audioContextCtor();
  const ctx = new Ctor();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(bytes);
  } finally {
    void ctx.close();
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * targetSr));
  const offline = new OfflineAudioContext(1, frames, targetSr);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}
