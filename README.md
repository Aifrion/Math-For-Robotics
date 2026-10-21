# Math for Robotics — CSE 276C interactive notes

Interactive lecture notes for **CSE 276C (Mathematics for Robotics, UCSD, Fall 2026)**:
one page per lecture with plain-words explanations, worked examples (every number
verified in Python), and self-contained interactive demos.

## Viewing the site

**Live site: <https://aifrion.github.io/Math-For-Robotics/>** — deployed from
[`docs/`](docs/) by GitHub Actions ([`.github/workflows/pages.yml`](.github/workflows/pages.yml))
on every push to `main`.

Other ways to view it:

1. **Locally**: `python3 -m http.server -d docs` and open <http://localhost:8000>.
2. Any single demo in `docs/demos/` is a standalone HTML file — open it directly,
   or embed it anywhere (e.g. Notion) by URL, such as
   `https://aifrion.github.io/Math-For-Robotics/demos/lec4/minimax_explorer.html`.

## What's here

| Page | Covers |
|---|---|
| `docs/lec1.html` | Transformations: rotations, Euler angles & gimbal lock, quaternions, homogeneous T |
| `docs/lec2.html` | Linear systems: row/column views, null space, Gauss/LDU, eigen, QR, SVD |
| `docs/lec3.html` | Interpolation: Lagrange, Vandermonde, cubic splines, bilinear, kriging |
| `docs/lec4.html` | Function approximation: minimax, equioscillation, Chebyshev, truncation |
| `docs/lec5.html` | Root finding: bisection → secant → Brent → Newton, Sturm, deflation, Müller |
| `docs/lec6.html` | Optimization: golden section, simplex, Powell, CG, annealing, DP |
| `docs/lec7.html` | Integration & ODEs: trapezoid → Romberg, RK4, Bulirsch–Stoer, stiffness |
| `docs/lec8.html` | Subspace methods: PCA, LDA/Fisherfaces, CCA, ICA |
| `docs/hw2.html` | HW2 companion: trajectory reconstruction by interpolation + cos-table design |

- `docs/demos/<lecture>/` — standalone interactive demos (no build step, no libraries)
- `docs/figures/<lecture>/` — compact SVG figures
- `scripts/<lecture>/verify.py` — the Python that produced **every number quoted on the pages**
- `scripts/<lecture>/make_figs.py` — figure generators (house plotter in `scripts/svgplot.py`)
- `reference-demos/`, `reference-figures/` — the original house-style reference set (lectures 2–4)
- `slides/`, `hw2/` — course materials
- `CLAUDE.md` — the build brief and house style guide

## Checking changes

Headless screenshot/console check for any page, demo, or SVG:

```bash
node scripts/shot.cjs docs/lec5.html /tmp/out.png 1000 light   # or 390 dark
```
