import Link from "next/link";
import styles from "./about.module.css";

export const metadata = {
  title: "AudioMind — About",
  description:
    "How AudioMind classifies music genres: the in-browser pipeline, the CNN, the GTZAN dataset, and its limits.",
};

const STATS: { value: string; label: string }[] = [
  { value: "10", label: "genres" },
  { value: "85%", label: "song-level accuracy" },
  { value: "0.42 MB", label: "model download" },
  { value: "100%", label: "runs in your browser" },
  { value: "0", label: "bytes of audio uploaded" },
];

const GENRES = [
  "blues", "classical", "country", "disco", "hiphop",
  "jazz", "metal", "pop", "reggae", "rock",
];

const PIPELINE: { step: string; title: string; body: string }[] = [
  {
    step: "01",
    title: "Decode & resample",
    body: "Your file is decoded and resampled to mono at 22,050 Hz using the browser's Web Audio API — the same sample rate the model was trained on. wav, mp3 and flac all work; nothing is uploaded.",
  },
  {
    step: "02",
    title: "Segment",
    body: "The clip is cut into non-overlapping 3-second windows (66,150 samples each). Each window is classified on its own, and the results are averaged — a whole song is easier to place than any single moment of it.",
  },
  {
    step: "03",
    title: "Log-mel spectrogram",
    body: "Each window becomes a picture the model can read: a short-time Fourier transform (n_fft 2048, hop 512), mapped onto 128 perceptual mel bands, then converted to decibels. The result is a 128 × 130 image of how energy is spread across pitch and time.",
  },
  {
    step: "04",
    title: "CNN → genre",
    body: "A small convolutional neural network reads that spectrogram like an image and outputs a probability for each of the 10 genres. Averaging those probabilities across the clip's windows gives the final call.",
  },
];

const CONFUSIONS: { pair: string; why: string }[] = [
  { pair: "rock ↔ metal", why: "shared distorted guitars and driving drums" },
  { pair: "disco ↔ pop", why: "similar four-on-the-floor production and tempo" },
  { pair: "reggae ↔ hiphop", why: "prominent bass and off-beat, sparse arrangements" },
  { pair: "blues ↔ country", why: "overlapping acoustic instrumentation and phrasing" },
];

const GLOSSARY: { term: string; def: string }[] = [
  {
    term: "Spectrogram",
    def: "A 2-D picture of sound: time runs left-to-right, frequency bottom-to-top, and brightness is how much energy is at that pitch and moment.",
  },
  {
    term: "Mel scale",
    def: "A warping of frequency that matches how humans hear — we tell low pitches apart far better than high ones. Mel bands give the model perceptually meaningful features.",
  },
  {
    term: "STFT",
    def: "Short-Time Fourier Transform. Slide a small window along the audio and, for each position, measure how much of each frequency is present. Stacking those columns builds the spectrogram.",
  },
  {
    term: "CNN",
    def: "Convolutional Neural Network. The kind of model that excels at images; since a spectrogram is an image, genre classification becomes image classification.",
  },
];

export default function About() {
  return (
    <div className={styles.page}>
      <header className={styles.rail}>
        <Link href="/" className={styles.brand}>
          <span className={styles.glyph} aria-hidden="true">◆</span>
          <span className="wordmark">AUDIOMIND</span>
        </Link>
        <Link href="/" className={styles.back}>
          ← analyzer
        </Link>
      </header>

      <main className={styles.main}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>about</p>
          <h1 className={styles.title}>
            A music-genre classifier that runs entirely in your browser.
          </h1>
          <p className={styles.lede}>
            Drop in a clip and AudioMind turns it into the same kind of picture
            its neural network trained on — a mel spectrogram — and predicts one
            of 10 genres. The model is downloaded once and runs on-device with
            TensorFlow.js, so there’s no server, nothing to wait on, and your
            audio never leaves your machine.
          </p>
        </section>

        <section className={styles.stats} aria-label="Key facts">
          {STATS.map((s) => (
            <div key={s.label} className={styles.stat}>
              <span className={styles.statValue}>{s.value}</span>
              <span className={styles.statLabel}>{s.label}</span>
            </div>
          ))}
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>How it works</h2>
          <p className={styles.p}>
            The trick is that <strong>a spectrogram is a 2-D image</strong>, so
            classifying a genre becomes classifying an image — a problem CNNs are
            very good at. The new work is at the front: turning sound into that
            image, exactly the way the training pipeline did.
          </p>
          <ol className={styles.pipeline}>
            {PIPELINE.map((s) => (
              <li key={s.step} className={styles.pipeStep}>
                <span className={styles.pipeNum}>{s.step}</span>
                <div>
                  <h3 className={styles.pipeTitle}>{s.title}</h3>
                  <p className={styles.pipeBody}>{s.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>The model</h2>
          <p className={styles.p}>
            A compact convolutional network — three convolution blocks
            (32 → 64 → 128 filters, each with batch-norm and max-pooling), global
            average pooling, then a dense layer into a 10-way softmax. Its input
            is a single 128 × 130 × 1 spectrogram; its output is 10 probabilities
            that sum to 1.
          </p>
          <div className={styles.cards}>
            <div className={styles.card}>
              <span className={styles.cardK}>Accuracy</span>
              <p className={styles.cardV}>
                <strong>85.3%</strong> at the song level (averaging a clip’s
                windows), <strong>78.3%</strong> on a single 3-second window.
                Chance is 10%.
              </p>
            </div>
            <div className={styles.card}>
              <span className={styles.cardK}>Where it slips</span>
              <p className={styles.cardV}>
                Mistakes cluster between genres that genuinely sound alike:
              </p>
              <ul className={styles.confList}>
                {CONFUSIONS.map((c) => (
                  <li key={c.pair}>
                    <code>{c.pair}</code> — {c.why}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>Everything runs on-device</h2>
          <p className={styles.p}>
            The Keras model was converted to a TensorFlow.js graph model — about
            0.42 MB — and runs in the page. Two things make that trustworthy:
          </p>
          <ul className={styles.checks}>
            <li>
              <strong>Privacy.</strong> Audio is decoded and analyzed locally.
              Nothing is uploaded; there’s no backend at all.
            </li>
            <li>
              <strong>Fidelity.</strong> The browser must build the <em>exact</em>{" "}
              spectrogram the model trained on, or predictions drift — so
              librosa’s mel filterbank is exported verbatim rather than
              re-derived in JavaScript. The result is verified two ways: the
              in-browser spectrogram matches librosa to ~1.7×10⁻⁴ dB, and the
              converted model reproduces the original Keras predictions.
            </li>
          </ul>
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>The data it learned from</h2>
          <p className={styles.p}>
            AudioMind was trained on <strong>GTZAN</strong>, the standard
            benchmark for this task: 1,000 clips, 30 seconds each, 100 per genre,
            across the 10 genres below.
          </p>
          <div className={styles.genreRow}>
            {GENRES.map((g) => (
              <span key={g} className={styles.genreChip}>{g}</span>
            ))}
          </div>
          <p className={styles.note}>
            GTZAN is a product of its time (assembled in the early 2000s) and is
            well known in the research literature to contain repeated tracks,
            some mislabelled clips, and a few corrupted files. It skews toward
            Western, mostly 20th-century recordings. It’s excellent for learning
            and benchmarking — and a fair reminder that a model is only ever as
            broad as the data behind it.
          </p>
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>What it can’t do</h2>
          <ul className={styles.limits}>
            <li>
              It knows <strong>these 10 genres and nothing else</strong> — hand
              it lo-fi, afrobeats, or a podcast and it will still confidently
              pick one of the ten.
            </li>
            <li>
              It judges <strong>timbre and texture over ~3 seconds</strong>, not
              lyrics, structure, or artist. A genre-blending track can land
              anywhere.
            </li>
            <li>
              It reflects GTZAN’s era and taste, so modern or non-Western music
              may be classified less reliably.
            </li>
            <li>
              It’s a focused demonstration of the audio → spectrogram → CNN
              pipeline, not a production music-tagging service.
            </li>
          </ul>
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>Glossary</h2>
          <dl className={styles.glossary}>
            {GLOSSARY.map((g) => (
              <div key={g.term} className={styles.gItem}>
                <dt className={styles.gTerm}>{g.term}</dt>
                <dd className={styles.gDef}>{g.def}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section className={styles.block}>
          <h2 className={styles.h2}>Built with</h2>
          <p className={styles.p}>
            Keras / TensorFlow and librosa for training and feature extraction;
            TensorFlow.js for in-browser inference; Next.js, TypeScript and the
            Web Audio API for the app. The spectrogram pipeline is checked
            against librosa and the original model in the test suite, so accuracy
            claims stay honest as the code changes.
          </p>
          <Link href="/" className={styles.cta}>
            Try it →
          </Link>
        </section>
      </main>

      <footer className={styles.footer}>
        <span>AudioMind · neural genre analysis · GTZAN · 10 genres · in-browser CNN</span>
      </footer>
    </div>
  );
}
