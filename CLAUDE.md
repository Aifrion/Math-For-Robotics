# CSE 276C interactive notes site: brief for Claude Code

This file is the brief for building and growing a website of lecture notes for **CSE 276C, Mathematics for Robotics** (UCSD, Fall 2026, Prof. Henrik Christensen). Each lecture becomes a page of clear notes with **interactive visualizations** woven into the text.

The owner (Brian) will hand you lecture slides (PDF) and sometimes his own notes. Your job, for each lecture, is to turn them into a page following the patterns below.

The folder `reference-demos/` holds 9 working demos from earlier lectures. **Open them, run them, and copy their patterns.** They are the house style. `reference-figures/` holds static SVG figures and the Python that generated them.

---

## 1. Who the reader is and how to pitch it

- Brian is a CS master's student. He learns best through **visuals and worked examples**, and wants **as many visuals as possible** (animations welcome).
- Prefer **high-level intuition and "why this matters for robots"** over proofs. Use the math, but always say what it *means* in plain words right next to it.
- Every abstract idea gets a **concrete robot example** (a 2-link arm, a camera, a sensor, a trajectory) and a **small worked example with real numbers**.
- Tone: friendly and direct. Short sentences. No filler.

---

## 2. Workflow for each new lecture

1. **Read the slides fully** (every page, including the figures). List the topics in order.
2. **Plan the page**: one section per topic, and for each, decide: worked example? static figure? interactive demo? (Aim for 2–4 demos per lecture, plus 3–6 static figures.)
3. **Verify all the math in Python/NumPy before writing it down.** Every number that appears on the page (worked examples, claims like "30 → 9 terms", eigenvalues, error values) must be computed, not recalled. Slides contain typos; never copy a formula from the slides without checking it.
4. **Write the page** using the template in §3.
5. **Build the demos** as self-contained HTML (see §5) and the static figures as SVG.
6. **Test everything headlessly** with Playwright: load each demo, check there are no console errors, take a screenshot, and *look at it* for overlapping labels, clipped text and arrows pointing the wrong way. Fix and re-check.
7. **Collect slide typos** into the page's "Typos in the slides" box, saying what's wrong and the correct version.

---

## 3. Page template (keep this order)

1. **Summary callout**: "The one-sentence version: …" plus the lecture number, professor and readings.
2. **Table of contents.**
3. **The big picture**: a short paragraph on how this lecture connects to the previous one, a small flow diagram (Mermaid is fine), and a table of *robotics problems this solves* (columns like "problem / what you have / what you need / method").
4. **Numbered sections, one per topic.** Each one has:
   - the idea in 1–2 plain sentences;
   - the key formula (KaTeX), followed by a **table explaining each symbol** in plain words;
   - a **worked example with real numbers** in a collapsible `<details>` block ("Worked example (click to open)");
   - a static figure and/or an interactive demo introduced with "🕹️ Try it: …", one line saying what to do;
   - callouts where useful: 💡 insight, ⚠️ gotcha, ✏️ "fix to my notes / fine print".
5. **Summary**: bullets plus a comparison table ("Method / one-line idea / use it when").
6. **⚠️ Typos in the slides to watch out for** (collapsible).
7. **🧠 Self-check questions** (collapsible, each answer in its own nested collapsible).

---

## 4. Site architecture (suggested; adjust if you have a strong reason)

- **Static site deployed to GitHub Pages.** No backend.
- Recommended stack: **Astro + MDX + KaTeX** (remark-math / rehype-katex), or anything equivalent that gives Markdown pages with math, collapsible blocks and embedded components.
- **Each interactive demo is its own standalone HTML file** in `public/demos/<lecture>/<name>.html`. Pages embed it with an `<iframe>`. Reasons:
  - each demo keeps working on its own and can be embedded elsewhere (Brian also embeds them in Notion by URL);
  - demos can't break each other (no global variable clashes).
- Make iframes **auto-size**: the demo posts its height (`parent.postMessage({demoHeight: document.body.scrollHeight}, '*')` on load and resize), and the page listens and sets the iframe height. No inner scrollbars.
- Static figures are **SVG files** in `public/figures/<lecture>/`.
- Navigation: a sidebar listing lectures in order. Each page also links to its neighbors.
- Both **light and dark mode** must look right (follow `prefers-color-scheme`).

---

## 5. How to build the demos

### 5.1 Rules
- **One HTML file per demo, no build step, no libraries.** Plain JS + inline SVG (or `<canvas>` for dense curves and heatmaps). Everything in the reference demos is under ~6 KB.
- **One idea per demo.** If it needs a paragraph to explain the controls, it's two demos.
- **Show the mechanism, not a picture *of* it.** A null space demo shows the arm's joints moving while the hand holds still, not a box labelled "null space".
- **Live numbers next to the picture.** Always show the current values (x, b, error, det, …) and a **check line** that proves it ("A·x = (0.50, 0.00) = b ✓"). Seeing the numbers change is half the learning.
- **Presets for the canonical cases.** One button per case: the lecture's exact example, a "good" case, and every failure case (singular, parallel, Runge, near-singular). The page text refers to these by name.
- **State-dependent captions.** A caption under the demo explains *what you're currently looking at* and changes with the preset or state.
- **Use "same thing, two verdicts" comparisons** whenever a concept is about rules or choices rather than physics. Example: two identical arms side by side, one graded against two rules and one against one rule.
- Default state must already teach something on load: no blank canvases.

### 5.2 Visual style
- Colors carry meaning; keep them consistent across the whole site:

| Color | Hex | Use |
|---|---|---|
| Blue | `#378ADD` / `#2563eb` | row space, first item, "column 1", main curve |
| Pink / red | `#D4537E` / `#dc2626` | second item, null space, error, "column 2" |
| Amber | `#BA7517` | the goal or target: b, the hand, the query point |
| Green | `#16a34a` / `#1D9E75` | a good result: solution x, best fit, "✓" |
| Gray | `#6b7280` / `#888780` | structure: axes, robot base, true function |
| Purple | `#534AB7` / `#7c3aed` | extra series when needed |

- Theme through CSS variables on `:root` with a `prefers-color-scheme: dark` override. Copy the variable block from `reference-demos/lec2_null_space_arm.html`. Never hard-code black or white text.
- Flat style: no gradients, shadows or glows. Thin borders (0.5–1px), rounded corners (8px) on buttons.
- Fonts: system sans; 13–14px for labels, 11–12px for secondary text, never below 11px. Use a real minus sign "−" in displayed numbers (`n.toFixed(2).replace('-','−')`).
- Buttons are a row of small outlined pills; the active preset or tab gets an accent background.
- Must work at phone width: `width="100%"` with a fixed `viewBox`, `touch-action:none` on draggable SVGs, pointer events (not mouse events).

### 5.3 Code patterns (all in the reference demos)
- **World↔screen mapping**: `X = v => L + (v - xmin)/(xmax - xmin)*W`, `Y = v => T + (ymax - v)/(ymax - ymin)*H`, plus the inverses for dragging.
- **Render by rebuilding the SVG string each frame** (`svg.innerHTML = s`). Small demos redraw instantly; this keeps the code tiny and stateless.
- **Dragging**: on `pointerdown`, convert client→viewBox coordinates using `getBoundingClientRect()` and the viewBox size, hit-test handles within ~14–16 px, `setPointerCapture`, update on `pointermove`, clamp to the plot range.
- **Snapping** to special states, so users can actually hit them: e.g. if two columns are within ~3° of parallel, snap to exactly parallel.
- **Animation**: `requestAnimationFrame` with a Pause/Play button; cancel any running animation when the user drags or picks a preset.
- **Arrow helper**: draw the shaft plus two short lines for the head, computed from the direction; skip arrows shorter than ~4 px.
- **`clipPath`** on the plot area so lines that leave the plot don't draw over labels.
- **Numerics**:
  - interpolation uses the barycentric Lagrange formula (stable);
  - splines use the tridiagonal second-derivative solve;
  - small dense solves use Gaussian elimination with partial pivoting;
  - **any robot motion re-solves with a fresh Jacobian every frame**: one linear step drifts off.
- **Dark-mode redraw**: listen to `matchMedia('(prefers-color-scheme: dark)')` changes and redraw (canvas demos read CSS variables at draw time).

### 5.4 Static figures
- SVG. Either matplotlib (`svg.fonttype='none'`, then strip metadata, round coordinates to 1 decimal or integers, simplify polylines) or the tiny hand-rolled plotter in `reference-figures/svgplot.py`, which outputs compact SVG directly.
- Keep each figure under ~15 KB, with a title that states the takeaway (e.g. "Same function, same degree. Only WHERE you sample changed"), and a one-line gray caption under it.
- Always render and look at the result before using it.

---

## 6. Catalog of what already exists (reuse and extend)

### Files in `reference-demos/`

| File | Lecture | What it teaches | Controls |
|---|---|---|---|
| `lec2_null_space_arm.html` | 2: linear systems | Joint-speed plane with the row space (blue) and null space (red) lines at right angles; a draggable x split into x_row + x_null; a 2-link arm showing the resulting hand motion | 3 tabs (bent arm with x and y rules / height rule only / straight singular arm), drag x, "slide along null space" animation, "drop to row space" |
| `lec2_same_physics_two_tasks.html` | 2 | Two identical arms: Task A (hand x = 0 and y = 0) is frozen, Task B (only y = 0) slides sideways at constant height. Rules are a checklist; they don't change the physics | Mode toggle: what each task allows / same motion on both; ✓/✗ checklist |
| `lec2_column_independence.html` | 2 | Column view: b = mix of the columns, with a skewed grid of all mixes. Dependent columns collapse the grid onto a line; unreachable b gets a least-squares point and a "cancelling mix" readout | Drag both column tips and b; presets; parallel snapping |
| `lec3_interpolation_explorer.html` | 3: interpolation | Linear vs one big Lagrange polynomial vs cubic spline; Runge wiggles appear as points are added | Drag, add and delete points; toggles; "Load Runge example" |
| `lec3_bilinear_explorer.html` | 3 | Bilinear weights = area of the opposite rectangle; heatmap of the cell | Drag the query point; sliders for the 4 corner values; live weight table |
| `lec3_kriging_explorer.html` | 3 | Gaussian process / kriging mean and ±2σ band; effect of bump width s and noise λ | Add, drag and delete points; sliders s, λ |
| `lec4_minimax_explorer.html` | 4: functional approximation | Best line for eˣ; the error equioscillates (+, −, +) at 0.2788; the lower bound 0.2715 is shown as a band | Sliders a, b; presets: Taylor / least squares / minimax |
| `lec4_chebyshev_explorer.html` | 4 | Evenly spaced vs Chebyshev nodes (half-circle construction); max error readout | Slider N; node-type toggle; function picker |
| `lec4_truncation_explorer.html` | 4 | Truncating a power series vs a Chebyshev series for 1/(2 + x); log-error plot; terms needed to reach 10⁻⁹ | Slider K; interval toggle [0, 1] / [−1, 1] |

### Demos and figures that were made in chat (no file here; rebuild them from these specs when doing Lecture 2's page)

- **m vs n arms** (static, 3 panels): a 3-joint arm reaching a 2D target in 2 poses (m < n), a 2-joint arm with one pose (m = n), and a 1-joint arm missing the target, with a red least-squares gap (m > n).
- **Jacobian anatomy** (static): a 2-link arm at shoulder 90°, elbow −90°. Arrows at the hand show what each joint does (column 1 = (−1, 1), column 2 = (0, 1)) and the goal b = (0.5, 0). Beside it, the matrix equation with the columns labelled shoulder/elbow and the rows labelled hand x / hand y. Solution x = (−0.5, 0.5).
- **Row view of the arm system** (interactive): axes θ̇₁, θ̇₂. Row 1 (−θ̇₁ = b₁) is a vertical blue line, row 2 (θ̇₁ + θ̇₂ = b₂) a slanted pink line, and the crossing point is x. Sliders for b₁ and b₂ slide the lines without turning them. Presets: move right / up / up-left / stay still. A check panel shows A·x = b.
- **Hand x / hand y and dropped rules** (static): the hand's velocity arrow split into a sideways part and an up-down part. Below, three panels: 2 rules → one allowed arrow; 1 rule (height) → a whole line of allowed sideways arrows; 0 rules → arrows in every direction.
- **Dependent columns cancel** (static): a straight arm. Shoulder +1 pushes the hand up 2 and elbow −2 pushes it down 2, so the hand stays still; 1·(0, 2) − 2·(0, 1) = 0, so (1, −2) is in the null space.
- **Row independence with two lines** (interactive): line 1 x + 2y = 3 is fixed; tilt and shift sliders move line 2. Row vectors are drawn as arrows at right angles to their lines; readouts show det and status (independent / same line / parallel). Presets: 3x − y = 2 (crosses at (1, 1)), 2x + 4y = 6, 2x + 4y = 10.
- **Eigen-coordinates** (interactive): A = [2 1; 1 2], eigenvectors v₁ = (1, 1) (×3) and v₂ = (1, −1) (×1). Two panels: drag b on the left and it splits into c₁v₁ + c₂v₂; on the right, x = (c₁/3)v₁ + (c₂/1)v₂. Check that A·x = b.
- **Time scale / linearization table**: one linear joint step vs the true arm motion, for step sizes 0.0005 / 0.05 / 0.25 / 0.5 rad. The error grows from ~0 to 12 cm.

---

## 7. Teaching insights from Brian's study sessions (important)

These are the exact points where Brian got confused. Make the pages and demos address them head-on.

1. **Row view**: lines are drawn in **x-space**. The crossing point is **x (the solution)**, not b. b only *slides* the lines; A sets their tilt.
2. **In the robot-arm system J·θ̇ = v**, the columns of A are **joints** (what each joint does to the hand) and the rows are **hand directions** (one rule each). People expect rows = joints; say clearly that it's the other way around.
3. **x is "how much of each column to mix"**: the "weights" intuition is correct, and it's literally the column view.
4. **b holds only the hand speeds you wrote a rule for.** b = 0 means "every *listed* speed is zero", not "nothing moves". With only a height rule, the hand still slides sideways; that direction just isn't in b.
5. **Rules don't change the physics.** The same joint motion moves the arm the same way regardless of A's rows; the rows only decide which motions *pass*. Demos should show "same motion, different verdict".
6. **The null space belongs to the task *and* the pose.** With the same two rules, a bent arm has no null space but a straight arm does. A singularity = the pose quietly adding a null space and removing a hand direction.
7. **No built-in time scale.** J maps rad → m (or rad/s → m/s) equally well. What matters is *step size*: the linear model is only exact for small motions, so controllers re-solve every tick (100–1000 Hz).
8. **m counts rules, not dimensions of space.** One rigid hand has up to 6 (3 position + 3 orientation); more bodies means more rows. In estimation problems (calibration, fitting, SLAM), m ≫ n is normal.
9. **Independence**: rows → lines not parallel; columns → arrows not on one line; for square matrices both happen exactly when det ≠ 0. n independent vectors are needed to reach everything in n-D (2 independent vectors in 3D only span a plane).
10. **Dependent columns → a nonzero mix cancels to zero → that mix is a null-space vector.** Show this physically (two joints doing the same job, one undoing the other).
11. **LDU**: it's plain Gauss elimination (below the diagonal only), not Gauss–Jordan. P records swaps, L records multipliers, D holds the pivots, and U's 1s come from dividing out the pivots. The point is reuse: factor once (~n³/3), then ~n² per new b.
12. **Eigen-decomposition is not a cheaper solver** (LU is cheaper). Its value is insight: in eigen-coordinates, solving = dividing each piece by its λ; λ = 0 means a null direction; tiny λ means noise sensitivity. It needs n independent eigenvectors (full rank isn't the condition; e.g. [1 1; 0 1] fails). It's the building block for SVD, PCA/subspace methods, stability (e^(λt)) and uncertainty / manipulability ellipses.
13. Double-check simple facts before stating them (e.g. A = [2 1; 1 2] has eigenvalues 3 and 1; trace and determinant checks: 3 + 1 = 4, 3 × 1 = 3).

---

## 8. Quality checklist before calling a lecture done

- [ ] Every topic in the slides is covered, in order.
- [ ] Every number on the page was computed in Python (keep the script in `scripts/<lecture>/`).
- [ ] Each section has an intuition sentence, a formula with a symbol table, a worked example, and a visual.
- [ ] 2–4 interactive demos, each loading with zero console errors and teaching something before any interaction.
- [ ] Every demo and figure screenshot-checked in **light and dark** mode and at **phone width**: no overlaps or clipping.
- [ ] Presets cover the lecture's own example and every failure case mentioned.
- [ ] The slide-typo list and self-check questions are present.
- [ ] Page links to the previous and next lectures; the sidebar is updated.
