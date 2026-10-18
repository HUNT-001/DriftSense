# VirtualSEM — the digital-twin instrument

`driftsense.sim.VirtualSEM` is a stateful, fully deterministic virtual scanning
electron microscope used to develop and evaluate the alignment agent. This
document records the **modelling assumptions** behind it. It intentionally
contains no measured numbers — quantitative results live in `results/` and are
rendered into the report by the docs pipeline, never typed here.

## Why a twin, and not the data engine

The synthetic data engine (`data_gen`) emits independent *(reference, search)*
pairs: each pair is a fresh scene, so two acquisitions never observe the same
physical wafer. That is the right abstraction for benchmarking a localizer in
isolation, but it is the wrong one for an *agent* that drives a single tool over
a single wafer and must reason across repeated acquisitions.

VirtualSEM closes that gap. It reuses the frozen `data_gen` physics unchanged
and wraps it in an instrument that holds persistent state, so the agent faces
the decisions a real operator faces: where to point, how long to dwell, when to
move the stage, and how to spend a finite dose and acquisition budget.

## Assumptions

### One wafer, generated once
A single clean wafer site is synthesised at construction and kept for the
instrument's lifetime. Every acquisition is a crop of that one site. This is the
defining twin property: re-imaging a location returns the *same* structure.

### Roughness is a permanent property, noise is not
Line-edge roughness, critical-dimension variation and defects are baked into the
persistent wafer, so they recur identically on every visit to a site. They are
treated as stable physical properties of the etched silicon. Detector and
counting noise, by contrast, are drawn independently for each acquisition. This
separation — persistent structure plus independent noise — is exactly what makes
a location recoverable from its roughness fingerprint rather than from the
commanded coordinate. See the references in `config.py` and `data_gen`.

### Dose, dwell and shot noise
Electron dose per pixel is taken to be proportional to dwell time, and shot
noise is the Poisson counting statistics of that dose. The modelling consequence
the agent must exploit is that acquisition noise falls as the inverse square
root of dwell: longer dwell buys a cleaner image at a proportionally higher dose
cost. Dose is accounted cumulatively across the session and is capped.

### The stage does not go where it is told
Stage positioning is modelled as the sum of three effects: a constant
power-on **bias**, a slow thermal **random walk** that advances with each
acquisition, and a per-move **positioning error** incurred whenever the stage is
commanded to move, growing with the move distance. The imaged field is therefore
centred on the commanded coordinate *plus* this error. Recovering that offset is
the agent's job.

### Scan rotation
A small scan rotation is applied geometrically to the imaged field before the
acquisition chain, modelling tool-to-tool and session-to-session rotation of the
raster relative to the wafer frame.

### Determinism
Given its seed, the instrument is fully reproducible: the wafer, the stage
physics and every acquisition's noise are drawn from independent but
seed-derived random streams. Identical construction arguments and an identical
sequence of calls reproduce identical images, metadata and ground truth.

## Ground truth is evaluator-only

The true stage error is the quantity the agent is being asked to estimate, so it
must never be observable from the acquisition path. The `(png_bytes, meta)`
returned by `acquire` carries only operator-visible information — the *commanded*
centre, field of view, dwell, rotation, pixel size, image dimensions and dose.
The true error is exposed solely through `verify()`, which exists for a grading
harness. An agent restricted to `acquire` has no path to the answer; this
boundary is asserted in the test suite.

## Scope and non-goals

VirtualSEM is a development and evaluation fixture, not a physical-accuracy
claim about any specific tool. It deliberately reuses — and does not re-derive —
the acquisition physics that the data engine already validates, so that the
instrument and the benchmark agree by construction. It does not read or depend
on any sealed data.
