# Drift-Sense — narrated video script & shot list

A ready-to-produce **~4-minute** explainer. Generate the narration in
**ElevenLabs**, drop each clip onto the matching visual in any editor
(CapCut, DaVinci Resolve, Premiere, or even PowerPoint → export video), and cut
on the beats noted below.

- **Runtime:** ~3 min 55 s (≈ 620 narration words @ ~160 wpm)
- **Aspect ratio:** 16:9, 1920×1080
- **Tone:** confident, technical but plain-spoken — you're explaining to smart judges, not reading a paper
- **Motif:** dark navy background, teal = "where/recall", coral = "which/precision", amber = geometry

> Full-screen branded cards live in `docs/video_assets/`. Content figures live in
> `docs/images/`. Content figures are not 16:9 — in your editor, place them
> centered on a solid navy (`#12203A`) 1920×1080 background with a short title
> caption above. Cards are already 16:9 and full-bleed.

---

## ElevenLabs setup

1. **Voice:** a calm, clear narrator — e.g. *Adam*, *Daniel*, or *Rachel*. Pick one and keep it for the whole video.
2. **Model:** *Eleven Multilingual v2* (or *Turbo v2.5* for speed).
3. **Settings:** Stability **45–55%**, Similarity **70–80%**, Style **0–15%**, Speaker boost **on**.
4. **Workflow:** paste **each scene's narration block separately**, export each as its own `.mp3` named `scene_01.mp3`, `scene_02.mp3`, … — this makes syncing to visuals trivial.
5. The lines already include pacing. Keep the em-dashes and periods; they control the breaths. Avoid adding SSML unless you want a specific pause (`<break time="0.4s"/>`).

---

## Timeline overview

| # | Time | Visual (file) | Beat |
|---|------|---------------|------|
| 1 | 0:00–0:18 | `video_assets/01_title.png` | Title + the problem in one sentence |
| 2 | 0:18–0:42 | `images/structure_gallery.png` | Why it's hard — periodic wafers |
| 3 | 0:42–1:00 | `images/failure_analysis_nominal_NCC.png` | Classical matching fails — 100% aliases |
| 4 | 1:00–1:18 | `video_assets/05_hook_553.png` | The hook: 553 vs 2 |
| 5 | 1:18–1:40 | `images/ambiguity_demo.png` | The insight: LER breaks the tie |
| 6 | 1:40–2:00 | `video_assets/07_two_questions.png` | The reframing: WHERE vs WHICH |
| 7 | 2:00–2:22 | `images/data_engine.png` | The data engine (our own dataset) |
| 8 | 2:22–2:44 | `video_assets/06_stat_0to100.png` | Stage 2B — LER discrimination result |
| 9 | 2:44–3:04 | `video_assets/08_stat_recall.png` | Stage 2A — spectral recall result |
| 10 | 3:04–3:24 | `images/candidate_budget.png` | Stage 2D — the provable boundary |
| 11 | 3:24–3:42 | `images/results_bars.png` | Results + the honesty control |
| 12 | 3:42–3:55 | `video_assets/12_outro.png` | Close |

---

## Scene-by-scene

### Scene 1 — Title  (0:00–0:18)
**Visual:** `video_assets/01_title.png` (hold; optional slow zoom-in 3%)
**On-screen caption:** *(none — the card has it)*

> **Narration:**
> This is Drift-Sense. The task sounds simple — find a small reference image
> inside a much larger scan of a semiconductor wafer, and return its exact
> coordinates. But on a real wafer, that turns out to be one of the hardest kinds
> of search there is.

---

### Scene 2 — Why it's hard  (0:18–0:42)
**Visual:** `images/structure_gallery.png` on navy bg. Caption above: **"Real wafers are periodic."**

> **Narration:**
> Wafers are built from densely repeating structures — DRAM memory grids, and
> FinFET transistor arrays. The pattern repeats every few pixels, hundreds of
> times across the image. So the reference doesn't match one location — it
> matches hundreds of them, almost perfectly. And Applied Materials calls this
> out directly as the case where standard template matching breaks.

---

### Scene 3 — Classical matching fails  (0:42–1:00)
**Visual:** `images/failure_analysis_nominal_NCC.png` on navy bg. Caption: **"Classical matching: 0% correct."**
**Cut:** on the word "wrong copy," flash the red bars in the figure.

> **Narration:**
> And it does. We ran the classical methods — cross-correlation, phase
> correlation, multi-scale matching. On periodic structures, they score zero
> percent. But here's the key finding: they don't fail randomly. A hundred
> percent of the time, they lock onto the wrong copy — off by an exact multiple
> of the pattern spacing. The failure is structured. Which means it can be
> understood.

---

### Scene 4 — The hook  (1:00–1:18)
**Visual:** `video_assets/05_hook_553.png`. Animate: "553" appears first, then the arrow, then "2" pops in teal.

> **Narration:**
> So we asked a sharper question. In a perfectly repeating image, how many
> locations look identical to the target? Five hundred and fifty-three. It's
> genuinely impossible — every copy is an equally valid answer. But add back the
> real physics of a wafer, and that number drops to two.

---

### Scene 5 — The insight  (1:18–1:40)
**Visual:** `images/ambiguity_demo.png` on navy bg. Caption: **"The tie-breaker: line-edge roughness."**

> **Narration:**
> The tie-breaker is something called line-edge roughness. Every etched line has
> a tiny, random waviness — and that waviness is frozen into the silicon. So two
> separate scans of the same spot see the *same* roughness, just with different
> noise on top. That makes roughness a fingerprint for position. The periodic
> pattern hides the location; the roughness reveals it.

---

### Scene 6 — The reframing  (1:40–2:00)
**Visual:** `video_assets/07_two_questions.png`

> **Narration:**
> This splits the problem cleanly in two. First — where could a copy possibly be?
> That's a frequency question, answered by the structure's own geometry. Second —
> which copy is the real one? That's a physics question, answered by the
> roughness fingerprint. Frequency finds the candidates. Roughness picks the
> truth.

---

### Scene 7 — The data engine  (2:00–2:22)
**Visual:** `images/data_engine.png` on navy bg. Caption: **"No dataset given — so we built the simulator."**

> **Narration:**
> There's a catch: no dataset is provided. So we built the simulator ourselves —
> a physics-based model of a scanning electron microscope. DRAM and FinFET
> structures, roughness, line-width drift, defects, and nine separate imaging
> effects, each grounded in the literature. Every image comes with exact,
> self-audited ground truth. That means dataset realism is something we control
> and can defend.

---

### Scene 8 — Stage 2B result  (2:22–2:44)
**Visual:** `video_assets/06_stat_0to100.png`

> **Narration:**
> Does the roughness fingerprint actually work? We ran a controlled experiment:
> switch on roughness, and nothing else. Accuracy jumps from chance to a hundred
> percent, with a separation score around four to five — where a value above one
> is already a clear signal. The extractor recovers the true roughness with
> ninety-nine-point-seven percent correlation, and it survives even the noisiest
> imaging.

---

### Scene 9 — Stage 2A result  (2:44–3:04)
**Visual:** `video_assets/08_stat_recall.png`

> **Narration:**
> The frequency side does its job too. By reading the structure's lattice
> directly from a Fourier transform, we generate candidate locations that keep
> the true site nearly every time. On the hardest setting, recall climbed from
> seventy-seven percent to a hundred — after we traced a stubborn failure down to
> a single rounding limit in the frequency analysis, and fixed it.

---

### Scene 10 — The provable boundary  (3:04–3:24)
**Visual:** `images/candidate_budget.png` on navy bg. Caption: **"We measured the limit — and proved it."**

> **Narration:**
> We didn't just build — we mapped the limits. This curve shows accuracy against
> the number of competing candidates, and it proves that shrinking that set is
> what raises accuracy. It also let us prove something negative but important: no
> cheap shortcut can replace the roughness step. On this problem, the roughness
> fingerprint isn't just useful — it's necessary.

---

### Scene 11 — Results + control  (3:24–3:42)
**Visual:** `images/results_bars.png` on navy bg. Caption: **"Honest, tier by tier."**
**Cut:** highlight the `ambiguous` 0% bar with the coral arrow.

> **Narration:**
> Here's the full picture, across four difficulty levels — including one we can't
> solve on purpose. When we strip the physics out, the image becomes truly
> random-looking periodic, and Drift-Sense correctly stays at zero. That control
> proves our results come from real signal, not from a quirk of our own
> simulator.

---

### Scene 12 — Close  (3:42–3:55)
**Visual:** `video_assets/12_outro.png` (replace `<you>` with your GitHub handle first)

> **Narration:**
> We decomposed semiconductor navigation the way an algorithm team would — with
> physics, signal processing, and the controls to back it up. No neural network
> required. That's Drift-Sense.

---

## Assembly checklist

- [ ] Generate 12 narration clips in ElevenLabs (`scene_01.mp3` … `scene_12.mp3`).
- [ ] For each **content figure** (scenes 2,3,5,7,10,11): drop it centered on a navy `#12203A` 1920×1080 background, add the caption line, and set the clip length to the scene's narration length.
- [ ] The **cards** (scenes 1,4,6,8,9,12) are already 16:9 — use full-bleed.
- [ ] Add a light motion: a slow 3–5% zoom (Ken Burns) on each still keeps it alive.
- [ ] On scenes 4, 8, 9 let the big number "pop" (scale-in over ~0.3 s) as the narrator says it.
- [ ] Optional background music: a soft ambient/tech bed at **−22 dB** under the voice. Duck it during narration.
- [ ] Export 1080p, H.264, 24 or 30 fps.
- [ ] Replace `<you>` in the outro card and README with your GitHub handle.

## Optional: 60-second cut

For a teaser, keep scenes **1 → 4 → 5 → 6 → 8 → 12** (title, 553-vs-2 hook,
insight, two-questions, the 0→100 result, close). That's the whole story in a
minute.
