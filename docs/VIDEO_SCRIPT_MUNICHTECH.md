# DriftSense — 2–3 min demo video script

Target: 2:45. Screen recording + voiceover. Shots you already have live:
`demo/app.py` (web UI), `python -m demo.cli`, and the montages in
`outputs/figures/`.

---

**[0:00–0:20] The hook — show, don't tell.**
Open on the `demo_pair16.png` montage.
> "This is a scanning-electron-microscope image of a chip. Somewhere in it is
> this reference pattern. Finding it sounds trivial — until you notice the whole
> wafer is made of *identical copies*. Classical template matching picks this one
> [point to orange box] — wrong, by 967 pixels. Our system, DriftSense, picks
> this one [green box] — the real one, to a fraction of a pixel."

**[0:20–0:45] The problem & why it matters.**
> "Re-finding a known pattern inside a large SEM image is a core step in
> semiconductor metrology and defect review. On periodic structures — memory,
> standard cells — hundreds of copies match equally well, and under real rotation,
> scale drift and noise, the classical method picks the wrong copy 82% of the time
> on hard cases. In a fab, a wrong match means a corrupted measurement and lost
> yield."

**[0:45–1:20] The idea — two questions, two physics.**
Show the middle (spectrum) then right (LER) panels.
> "DriftSense splits the problem. First, *where could a copy be?* The pattern is a
> lattice, so its FFT gives us the reciprocal lattice — we enumerate only the
> structurally valid positions, cutting a million candidates to about 1,500 while
> keeping the true one 97.5% of the time. Second, *which copy is real?* Every
> etched edge has a unique, fixed roughness — line-edge roughness — a physical
> fingerprint of that exact spot on the wafer. We match that fingerprint to pick
> the one true copy."

**[1:20–1:55] The live demo.**
Screen-record the web app: pick hard / pair 16 → Locate → result appears.
Then drag a different pair. Then (optional) upload custom images.
> "Here it is running. Pick any SEM pair… classical NCC in orange, DriftSense in
> green, ground truth dotted. Located — error under a pixel. It's CPU-only: no
> GPU, no cloud, no trained model."

**[1:55–2:20] The numbers.**
Cut to `benchmark_chart.png`.
> "Across a sealed test set: on clean images both are near-perfect, but as
> distortion rises the gap explodes. On the hard tier DriftSense triples classical
> accuracy — 17.5% to 57.5% — and recall is 97.5% everywhere. And it's honest:
> we report where it still falls short."

**[2:20–2:45] Why Europe — the close.**
> "Metrology and inspection is where Europe leads — ASML, Zeiss, imec — and where
> the Chips Act is investing. DriftSense strengthens exactly that layer, and it's
> sovereign by design: explainable physics, and not one byte of fab data ever
> leaves the machine. That's technology built in Europe, for Europe. Thanks for
> watching."

---
**Recording tips**
- Run `uvicorn demo.app:app --port 8000` beforehand; pre-load the page.
- The pipeline takes ~6 s/image — either let it run (shows it's real) or cut.
- Keep the montage on screen while narrating; it carries the story.
