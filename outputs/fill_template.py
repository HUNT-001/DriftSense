"""
Fill the Hackathon-2026 idea-submission template with DriftSense content (v3).

v3 does real layout surgery: for each content block it resizes the CONTAINER
rectangle, the LABEL, and the BODY text box together, so text never spills
outside its panel, and reserves a clean right-hand column for figures.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from PIL import Image
from pathlib import Path

ROOT = Path("/sessions/awesome-affectionate-maxwell/mnt/DriftSense")
IMGD = ROOT/"docs/images"
SRC  = ROOT/"Idea-Submission-Template_Hackathon-2026-1.pptx"
OUT  = ROOT/"DriftSense_PS2.pptx"
E = 914400
prs = Presentation(str(SRC))
S = prs.slides

WHITE = RGBColor(0xFF,0xFF,0xFF)
BODY  = RGBColor(0xE6,0xEE,0xF8)

# ---------------- primitives ----------------
def tb(slide, needle, exact=False):
    for sh in slide.shapes:
        if not sh.has_text_frame: continue
        t = sh.text_frame.text.strip()
        if (t == needle) if exact else (needle in t):
            return sh
    return None

def all_tb(slide, needle):
    return [sh for sh in slide.shapes
            if sh.has_text_frame and needle in sh.text_frame.text]

def font_of(tf):
    for p in tf.paragraphs:
        for r in p.runs: return r.font
    return None

def put(shape, text, size=None, color=None, bold=None):
    """Single-line replace preserving template styling."""
    tf = shape.text_frame; tf.word_wrap = True
    p0 = tf.paragraphs[0]
    if p0.runs:
        r0 = p0.runs[0]; r0.text = text
        for r in p0.runs[1:]: r.text = ""
    else:
        r0 = p0.add_run(); r0.text = text
    if size:  r0.font.size = Pt(size)
    if bold is not None: r0.font.bold = bold
    if color:
        try: r0.font.color.rgb = color
        except Exception: pass
    for extra in list(tf.paragraphs[1:]): extra._p.getparent().remove(extra._p)

def bullets(shape, lines, size=11, color=BODY, bullet="•  ", space=3):
    tf = shape.text_frame
    tf.word_wrap = True
    try: tf.auto_size = MSO_AUTO_SIZE.NONE
    except Exception: pass
    try: tf.vertical_anchor = MSO_ANCHOR.TOP
    except Exception: pass
    f = font_of(tf); fname = f.name if f else "Calibri"
    for p in list(tf.paragraphs): p._p.getparent().remove(p._p)
    for ln in lines:
        p = tf.add_paragraph(); p.space_after = Pt(space); p.line_spacing = 1.0
        r = p.add_run(); r.text = (bullet + ln) if bullet else ln
        r.font.name = fname; r.font.size = Pt(size); r.font.bold = False
        try: r.font.color.rgb = color
        except Exception: pass

def geo(shape, L=None, T=None, W=None, H=None):
    if L is not None: shape.left   = Emu(int(L*E))
    if T is not None: shape.top    = Emu(int(T*E))
    if W is not None: shape.width  = Emu(int(W*E))
    if H is not None: shape.height = Emu(int(H*E))

def container_for(slide, label_shape, max_w=11.6):
    """Find the rounded-rect panel that visually contains a label."""
    lx, ly = label_shape.left, label_shape.top
    best, best_area = None, None
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip(): continue
        if sh.left is None or sh.width is None: continue
        if sh.width/E > max_w or sh.height/E > 4.5: continue
        if sh.width/E < 1.0 or sh.height/E < 0.4: continue
        if sh.left <= lx and sh.top <= ly and \
           sh.left+sh.width >= lx and sh.top+sh.height >= ly:
            a = sh.width*sh.height
            if best is None or a < best_area: best, best_area = sh, a
    return best

def block(slide, label_text, lines, L, T, W, H, size=11, lab_dx=0.26, lab_dy=0.25):
    """Reposition container+label+body as one unit and fill the body."""
    lab = tb(slide, label_text)
    if lab is None: return
    # body = the text box just below the label (nearest below, same column)
    cands = [sh for sh in slide.shapes
             if sh.has_text_frame and sh is not lab
             and sh.top > lab.top and abs(sh.left-lab.left) < 0.6*E
             and sh.text_frame.text.strip()]
    body = min(cands, key=lambda s: s.top) if cands else None
    cont = container_for(slide, lab)
    # Carry any small decorative icons that sit inside the container, so they
    # do not get orphaned when the panel moves.
    riders = []
    if cont is not None:
        c0L, c0T = cont.left, cont.top
        for sh in slide.shapes:
            if sh is cont or sh is lab or sh is body: continue
            if sh.left is None or sh.width is None: continue
            if sh.width/E > 0.42 or sh.height/E > 0.42: continue
            if (cont.left <= sh.left and cont.top <= sh.top and
                sh.left+sh.width <= cont.left+cont.width and
                sh.top+sh.height <= cont.top+cont.height):
                riders.append((sh, (sh.left-c0L)/E, (sh.top-c0T)/E))
        geo(cont, L, T, W, H)
        for sh, dx, dy in riders:
            geo(sh, L+dx, T+dy)
    geo(lab, L+lab_dx, T+lab_dy, min(W-2*lab_dx, 4.0), 0.21)
    if body is not None:
        geo(body, L+lab_dx, T+lab_dy+0.30, W-2*lab_dx, H-lab_dy-0.38)
        bullets(body, lines, size=size)

def picture(slide, path, L, T, W, H):
    im = Image.open(path); ar = im.width/im.height
    w = W; h = w/ar
    if h > H: h = H; w = h*ar
    slide.shapes.add_picture(str(path), Inches(L+(W-w)/2), Inches(T+(H-h)/2),
                             Inches(w), Inches(h))


def retitle(slide, prefix, tail, size=None):
    """Replace the grey instructional tail of a template title with real content,
    keeping run 0 (the white heading) and its styling intact."""
    for sh in slide.shapes:
        if not sh.has_text_frame: continue
        if prefix not in sh.text_frame.text: continue
        for pa in sh.text_frame.paragraphs:
            runs = pa.runs
            if len(runs) >= 2 and prefix in runs[0].text:
                runs[1].text = tail
                if size: runs[1].font.size = Pt(size)
                for r in runs[2:]: r.text = ""
                # widen the title box so the line never wraps
                sh.width = Emu(int(12.2*E)); sh.left = Emu(int(0.92*E))
                sh.text_frame.word_wrap = True
                return sh
    return None

# =============== SLIDE 2 · TEAM ===============
s = S[1]
put(tb(s,"Enter Team Name Here...",exact=True), "DriftSense", color=WHITE)

names = sorted([x for x in s.shapes if x.has_text_frame and
                x.text_frame.text.strip()=="{Enter Name}"], key=lambda z: z.top)
years = sorted([x for x in s.shapes if x.has_text_frame and
                x.text_frame.text.strip()=="{Enter Year}"], key=lambda z: z.top)
# widen so nothing wraps: NAME col 4.46 -> 9.2 ; YEAR col 9.29 -> 12.4
for sh in names: geo(sh, W=4.70)
for sh in years: geo(sh, W=3.00)
put(names[0], "Tanush Pavan Vakkalagadda", color=WHITE)
put(names[1], "LKM Premchand Korukonda",  color=WHITE)
put(years[0], "3rd Year — B.Tech (EEE)",  color=WHITE)
put(years[1], "Final Year — B.Tech (ELC)",color=WHITE)
for sh in names[2:] + years[2:]: put(sh, "")
for lab in ["Member 2","Member 3","3","4"]:
    sh = tb(s, lab, exact=True)
    if sh: put(sh, "")

col = tb(s,"{Enter Full College Name}");  geo(col, W=10.4); put(col,"Amrita Vishwa Vidyapeetham, Coimbatore", color=WHITE)
ph  = tb(s,"{+91 XXXXX XXXXX}");          geo(ph,  W=5.20); put(ph, "+91 92905 70949", color=WHITE)
em  = tb(s,"{email@example.com}");        geo(em,  W=4.90); put(em, "tanushpavan2007@gmail.com", color=WHITE)

# =============== SLIDE 3 · PROBLEM ===============
s = S[2]
h = tb(s,"Selected the problem statement")
geo(h, L=1.85, T=2.66, W=5.35, H=0.62)
put(h, "PS2 · Applied Materials — locating a reference pattern in periodic SEM wafer images", size=13)
block(s, "DESCRIPTION / DETAILS", [
    "Find a small reference inside a 1000×1000 SEM image → output centre (x, y).",
    "Wafers are dense periodic arrays (DRAM, FinFET) — hundreds of sites look identical.",
    "Classical template matching locks onto the wrong copy: 0% accuracy, 100% period aliases.",
    "No dataset is provided — realistic data generation is part of the challenge.",
], L=1.05, T=3.55, W=6.15, H=2.35, size=11)
picture(s, IMGD/"ambiguity_demo.png", L=7.55, T=2.45, W=4.85, H=4.45)

# =============== SLIDE 4 · IDEA ===============
s = S[3]
retitle(s, "Idea Description", "Two questions instead of one blind search", size=22)
h = tb(s,"Provide a brief summary of your idea")
geo(h, L=1.85, T=2.72, W=5.35, H=0.50)
put(h, "Reframe the search as two different information problems.", size=13)
block(s, "KEY CONCEPT & APPROACH", [
    "WHERE could a copy be?  →  spectral lattice from the FFT.",
    "WHICH copy is the real one?  →  line-edge-roughness (LER) fingerprint.",
    "Classical CV + signal processing — no neural network.",
], L=1.05, T=3.35, W=6.15, H=1.55, size=11)
block(s, "SOLUTION OVERVIEW", [
    "LER is frozen into the etched silicon, so two scans see the same roughness — a positional fingerprint that breaks periodic ambiguity.",
], L=1.05, T=5.10, W=6.15, H=1.35, size=11)
picture(s, IMGD/"two_questions.png", L=7.55, T=2.55, W=4.85, H=4.20)

# =============== SLIDE 5 · PROPOSED SOLUTION ===============
s = S[4]
retitle(s, "Proposed Solution", " From a raw SEM scan to an exact (x, y)", size=22)
h = tb(s,"Describe your idea in detail")
geo(h, L=1.85, T=2.72, W=10.4, H=0.45)
put(h, "Two stages, each solving a different half of the problem.", size=14)
block(s, "SOLUTION DETAILS", [
    "Stage 2A — spectral lattice → ~1500 structurally valid candidates (recall 77.5% → 100%).",
    "Stage 2B — LER fingerprint ranks the candidates (separation d′ ≈ 4–5).",
    "Stage 2C/2E — safe rotation + scale de-warp; max-fallback can never regress.",
    "Sub-pixel refinement → final (x, y) coordinate.",
], L=1.05, T=3.38, W=5.85, H=2.15, size=11)
picture(s, IMGD/"pipeline_architecture.png", L=7.15, T=3.30, W=5.25, H=3.30)

# =============== SLIDE 6 · INNOVATION ===============
s = S[5]
h = tb(s,"Highlight what makes your idea unique")
geo(h, L=1.85, T=2.72, W=10.4, H=0.45)
put(h, "A new decomposition — and a physical signal used where models fail.", size=14)
block(s, "KEY INNOVATION", [
    "Separate recall (frequency) from discrimination (LER physics).",
    "Use line-edge roughness as a positional fingerprint — signal, not model size.",
    "Built on our own physics-based SEM data engine.",
], L=1.05, T=3.38, W=5.49, H=1.62, size=11)
block(s, "COMPETITIVE ADVANTAGE", [
    "Solves the exact aliasing case where template matching scores 0%.",
    "Interpretable · no training data or GPU required.",
    "Includes an impossibility control proving the gains are real.",
], L=6.79, T=3.38, W=5.49, H=1.62, size=11)
picture(s, IMGD/"results_bars.png", L=3.75, T=5.15, W=5.85, H=1.80)

# =============== SLIDE 7 · IMPACT ===============
s = S[6]
h = tb(s,"Explain how your solution will make an impact")
geo(h, L=1.85, T=2.72, W=10.4, H=0.45)
put(h, "Recovering navigation accuracy where classical inspection tools fail.", size=14)
block(s, "Primary Impact", [
    "Reliable navigation and overlay recovery on repetitive wafers.",
    "Turns a 0%-accuracy failure mode into a solved one.",
    "No training data, no GPU — deployable alongside existing tools.",
], L=1.05, T=3.38, W=5.49, H=1.72, size=11, lab_dx=0.55)
block(s, "Quantifiable Outcomes", [
    "Clean-tier accuracy  0% → 100%",
    "Hard-tier recall  77.5% → 100%",
    "Discrimination  d′ ≈ 4–5  (vs 1.1–1.5 for NCC)",
    "Ambiguity collapse  553 → 2 equal peaks",
], L=6.79, T=3.38, W=5.49, H=1.72, size=11, lab_dx=0.58)
picture(s, IMGD/"results_journey.png", L=1.15, T=5.25, W=11.0, H=1.70)

# =============== SLIDE 8 · TECHNOLOGY ===============
s = S[7]
h = tb(s,"Describe the technologies, methodologies")
geo(h, L=1.93, T=2.55, W=10.2, H=0.45)
put(h, "Classical CV + signal processing on a physics-based SEM data engine.", size=14)
block(s, "IMPLEMENTATION STRATEGY", [
    "Software: Python · NumPy / SciPy · OpenCV · Matplotlib (PyTorch-ready).",
    "Data engine: 9 cited SEM effects, analytically inverted ground truth (100% audited).",
    "Algorithms: FFT lattice · sub-pixel peak fit · per-line LER extraction · correlation ranking.",
    "Feasibility: runs on CPU; 31 automated tests; frozen params; sealed held-out set.",
], L=1.93, T=3.18, W=5.95, H=2.20, size=10.5)
picture(s, IMGD/"data_engine.png", L=8.15, T=3.22, W=4.10, H=2.15)
# bottom three cards get real content
for lab, txt in [("Software Architecture","Two-stage pipeline:\nspectral lattice → LER ranking"),
                 ("Hardware Components","CPU-only; no GPU or\nspecialised hardware required"),
                 ("Development Tools","Python · OpenCV · pytest\nGit · Matplotlib")]:
    sh = tb(s, lab, exact=True)
    if sh:
        geo(sh, W=3.2, H=0.24)
        put(sh, lab, size=11, color=RGBColor(0xAE,0xFF,0x82), bold=True)

# =============== SLIDE 9 · LINKS ===============
s = S[8]
for sh in all_tb(s,"{Paste your GitHub"):
    put(sh, "https://github.com/HUNT-001/DriftSense", color=WHITE)
for sh in all_tb(s,"{Paste your Video"):
    put(sh, "Demo video — paste your uploaded link here", color=WHITE)

# =============== SLIDE 10 · RESEARCH ===============
s = S[9]
rb = all_tb(s,"{Detail your research findings")
if rb:
    bullets(rb[0], [
        "Periodic ambiguity is information-theoretic; line-edge roughness is a stable physical property that resolves it.",
        "SEM imaging modelled from first principles: secondary-electron edge effect, Poisson shot noise, beam PSF, scan drift.",
        "Validated with an impossibility control: with physics removed, accuracy correctly stays at 0%.",
    ], size=11)
for n, v in [("{Ref 1","Bunday et al., 'Optimal parameters for CD-SEM line-edge roughness', Proc. SPIE 6152 (2006)."),
             ("{Ref 2","Constantoudis et al., 'Line-edge roughness and CD variation', J. Micro/Nanolith. MEMS MOEMS 3(3) (2004)."),
             ("{Ref 3","Kuglin & Hines, 'The phase correlation image alignment method', Proc. IEEE (1975).")]:
    for sh in all_tb(s, n): put(sh, v, size=11, color=BODY)

# drop the instructions slide
lst = prs.slides._sldIdLst; lst.remove(list(lst)[0])
prs.save(str(OUT))
print("SAVED", OUT.name, "| slides:", len(prs.slides._sldIdLst))
