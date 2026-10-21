# Math for Robotics — CSE 276C interactive notes

Interactive lecture notes for **CSE 276C (Mathematics for Robotics, UCSD, Fall 2026)**:
one page per lecture with plain-words explanations, worked examples (every number
verified in Python), and self-contained interactive demos.

## Viewing the site

The site is a zero-build static site in [`docs/`](docs/). Three ways to view it:

1. **GitHub Pages** (recommended): repo **Settings → Pages → Deploy from a branch**,
   pick `main` (after merging) and folder `/docs`. The site then lives at
   `https://<user>.github.io/Math-For-Robotics/`.
2. **Locally**: `python3 -m http.server -d docs` and open <http://localhost:8000>.
3. Any single demo in `docs/demos/` is a standalone HTML file — open it directly,
   or embed it anywhere (e.g. Notion) by URL.

## What's here

| Page | Covers |
|---|---|
| `docs/hw2.html` | HW2 companion: trajectory reconstruction by interpolation + cos-table design |
| `docs/lec5.html` | Root finding: bisection → secant → Brent → Newton, Sturm, deflation, Müller |
| `docs/lec6.html` | Optimization: golden section, simplex, Powell, CG, annealing, DP |
| `docs/lec7.html` | Integration & ODEs: trapezoid → Romberg, RK4, Bulirsch–Stoer, stiffness |
| `docs/lec8.html` | Subspace methods: PCA, LDA/Fisherfaces, CCA, ICA |

- `docs/demos/<lecture>/` — standalone interactive demos (no build step, no libraries)
- `docs/figures/<lecture>/` — compact SVG figures
- `scripts/<lecture>/verify.py` — the Python that produced **every number quoted on the pages**
- `scripts/<lecture>/make_figs.py` — figure generators (house plotter in `scripts/svgplot.py`)
- `reference-demos/`, `reference-figures/` — the original house-style reference set (lectures 2–4)
- `slides/`, `hw2/` — course materials
- `CLAUDE.md` — the build brief and house style guide

Lectures 2–4 pages are not built yet (no slide PDFs in the repo); their reference
demos live on in `reference-demos/` and the interpolation explorer is embedded in
the HW2 page.

## Checking changes

Headless screenshot/console check for any page, demo, or SVG:

```bash
node scripts/shot.cjs docs/lec5.html /tmp/out.png 1000 light   # or 390 dark
```
