"""Lecture 4 figures (from the reference kit, repointed at docs/figures/lec4)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from numpy.polynomial import chebyshev as C
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec4/"
os.makedirs(OUT, exist_ok=True)
e = np.e
b_mm = (e - 1 / e) / 2
x1 = np.log(b_mm)
a_mm = (e - b_mm * x1) / 2
E = 0.2788
xs = np.linspace(-1, 1, 160)

# ---------- 01 best line for e^x: curve + error ----------
f = Fig(860, 345)
p = Panel(f, 60, 45, 340, 230, (-1, 1), (-0.2, 3), "Three straight-line guesses for eˣ on [−1, 1]",
          xticks=[-1, -0.5, 0, 0.5, 1], yticks=[0, 1, 2, 3], xlabel="x")
p.line(xs, np.exp(xs), BLACK, 2.6)
p.line(xs, 1 + xs, ORANGE, 2, "6 4")
p.line(xs, np.sinh(1) + 3 / e * xs, PURPLE, 2, "2 3")
p.line(xs, a_mm + b_mm * xs, GREEN, 2.6)
p.legend([("eˣ", BLACK, None), ("Taylor at 0:  1 + x", ORANGE, "6 4"), ("least squares", PURPLE, "2 3"),
          ("minimax:  1.2643 + 1.1752x", GREEN, None)])
q = Panel(f, 490, 45, 340, 230, (-1, 1), (-0.8, 0.8), "Error  e(x) = eˣ − p(x)",
          xticks=[-1, -0.5, 0, 0.5, 1], yticks=[-0.6, -0.3, 0, 0.3, 0.6], xlabel="x")
q.band(xs, -E * np.ones_like(xs), E * np.ones_like(xs), GREEN, 0.08)
q.hline(E, GREEN); q.hline(-E, GREEN)
q.line(xs, np.exp(xs) - 1 - xs, ORANGE, 2, "6 4")
q.line(xs, np.exp(xs) - np.sinh(1) - 3 / e * xs, PURPLE, 2, "2 3")
q.line(xs, np.exp(xs) - a_mm - b_mm * xs, GREEN, 2.6)
q.dots([-1, x1, 1], [E, -E, E], GREEN, 5.5)
q.label(-1, E, "+0.2788", GREEN, 11, dx=8, dy=-8, weight="bold")
q.label(x1, -E, "−0.2788", GREEN, 11, anchor="middle", dy=20, weight="bold")
q.label(1, E, "+0.2788", GREEN, 11, anchor="end", dx=-8, dy=-8, weight="bold")
q.label(0.98, 0.72, "Taylor: 0.718 at x=1", ORANGE, 10.5, anchor="end")
f.text(430, 336, "Minimax = the line whose worst error is smallest. Its error touches ±0.2788 at 3 points with alternating signs.", 12, "middle", GRAY)
print("01", f.save(OUT + "01_minimax_line.svg"))

# ---------- 02 lower bound picture ----------
f = Fig(560, 350)
p = Panel(f, 55, 40, 470, 235, (-1.15, 1.15), (0.2, 2.9), "Why no line can beat 0.2715",
          xticks=[-1, 0, 1], yticks=[0.5, 1, 1.5, 2, 2.5], xlabel="x")
xx = np.linspace(-1.15, 1.15, 120)
p.line(xx, np.exp(xx), BLACK, 2.4)
chord = lambda x: np.cosh(1) + np.sinh(1) * x
p.line(xx, chord(xx), GRAY, 1.6, "5 4")
gap = np.cosh(1) - 1
mid = lambda x: chord(x) - gap / 2
p.line(xx, mid(xx), GREEN, 2.4)
p.dots([-1, 0, 1], [np.exp(-1), 1, e], RED, 5.5)
p.vseg(0, 1, np.cosh(1), RED, 1.8, None)
p.label(0, (1 + np.cosh(1)) / 2, f"gap = {gap:.4f}", RED, 11.5, dx=8, weight="bold")
p.label(-0.75, chord(-0.75), "chord through the outer points", GRAY, 10.5, dy=-12)
p.label(0.45, mid(0.45), "best you can do: split the gap", GREEN, 10.5, dy=24)
f.text(280, 322, "A line can't bend, so it must miss at least one of the 3 points by ≥ gap / 2 = 0.2715.", 12, "middle", GRAY)
f.text(280, 340, "That's exactly the slide's lower bound (f(−1) − 2f(0) + f(1)) / 4.", 12, "middle", GRAY)
print("02", f.save(OUT + "02_lower_bound.svg"))

# ---------- 03 chebyshev: circle + T_n ----------
f = Fig(900, 320)
N = 8
p = Panel(f, 40, 45, 340, 200, (-1.1, 1.1), (-0.15, 1.15), f"Chebyshev nodes = shadows of evenly spaced points (N = {N})",
          xticks=[-1, 0, 1], yticks=None, grid=False)
th = np.linspace(0, np.pi, 100)
p.line(np.cos(th), np.sin(th), GRAY, 1.4)
k = np.arange(N)
ang = np.pi * (k + 0.5) / N
for a in ang:
    p.vseg(np.cos(a), np.sin(a), 0, BLUE, 1, "3 3")
p.dots(np.cos(ang), np.sin(ang), BLUE, 4.5, True)
p.dots(np.cos(ang), np.zeros(N), RED, 5)
p.label(0, 1.07, "equal angles on the circle", BLUE, 11, "middle")
p.label(0, 0.08, "→ crowded near ±1 on the line", RED, 11, "middle")
cols = [GRAY, ORANGE, GREEN, RED, PURPLE]
q = Panel(f, 490, 45, 340, 200, (-1, 1), (-1.1, 1.1), "Tₙ(x) = cos(n·arccos x) always stays between −1 and 1",
          xticks=[-1, -0.5, 0, 0.5, 1], yticks=[-1, 0, 1])
for n in range(5):
    q.line(xs, np.cos(n * np.arccos(xs)), cols[n], 2.1)
q.legend([(f"T{s}", cols[i], None) for i, s in enumerate("₀₁₂₃₄")], x=q.x + q.w + 8, y=q.y + 20)
f.text(430, 295, "Same wiggle height everywhere → errors get spread evenly across [−1, 1] instead of piling up at the ends.", 12, "middle", GRAY)
print("03", f.save(OUT + "03_chebyshev.svg"))

# ---------- 04 equispaced vs chebyshev nodes ----------
runge = lambda x: 1 / (1 + 25 * x ** 2)
n = 11
xe = np.linspace(-1, 1, n)
xc = np.cos(np.pi * (np.arange(n) + 0.5) / n)
xx = np.linspace(-1, 1, 260)
pe = np.polyval(np.polyfit(xe, runge(xe), n - 1), xx)
pc = np.polyval(np.polyfit(xc, runge(xc), n - 1), xx)
f = Fig(860, 305)
p = Panel(f, 50, 45, 340, 200, (-1, 1), (-0.4, 2.0), "11 evenly spaced points", xticks=[-1, 0, 1], yticks=[0, 1, 2])
p.line(xx, runge(xx), BLACK, 2)
p.line(xx, pe, RED, 2.3)
p.dots(xe, runge(xe), BLUE, 4.5)
p.label(0, 1.85, f"max error {np.max(abs(pe - runge(xx))):.2f}", RED, 12, "middle", weight="bold")
q = Panel(f, 490, 45, 340, 200, (-1, 1), (-0.4, 2.0), "11 Chebyshev points", xticks=[-1, 0, 1], yticks=[0, 1, 2])
q.line(xx, runge(xx), BLACK, 2)
q.line(xx, pc, GREEN, 2.3)
q.dots(xc, runge(xc), BLUE, 4.5)
q.label(0, 1.85, f"max error {np.max(abs(pc - runge(xx))):.2f}", GREEN, 12, "middle", weight="bold")
f.text(430, 285, "Same function, same degree-10 polynomial. Only WHERE you sample changed — Lecture 3's Runge wiggles mostly disappear.", 12, "middle", GRAY)
print("04", f.save(OUT + "04_nodes.svg"), np.max(abs(pe - runge(xx))), np.max(abs(pc - runge(xx))))

# ---------- 05 truncation: coefficient decay ----------
K = 31
pw = np.array([0.5 ** (k + 1) for k in range(K)])
g = lambda y: 1 / (2 + (y + 1) / 2)
cc = np.abs(C.chebinterpolate(g, 40))[:K]
f = Fig(760, 340)
p = Panel(f, 70, 45, 640, 210, (-0.8, K - 0.2), (-12, 0), "Size of each term for f(x) = 1/(2 + x) on [0, 1]  (log scale)",
          xticks=[0, 5, 10, 15, 20, 25, 30], yticks=[0, -3, -6, -9, -12], xlabel="term number k", ylabel="|coefficient|", logy=True)
for k in range(K):
    p.bar(k - 0.2, np.log10(pw[k]), 0.38, ORANGE)
    if cc[k] > 1e-13:
        p.bar(k + 0.2, np.log10(cc[k]), 0.38, GREEN)
p.hline(-9, RED, 1.5)
p.legend([("power series  1/2, x/4, x²/8 …   (halves each term)", ORANGE, None), ("Chebyshev cₖ   (÷ ~10 each term)", GREEN, None), ("target error 10⁻⁹", RED, "4 3")], x=380, y=p.y + 20)
f.text(390, 330, "Power series needs ~30 terms to get under 10⁻⁹; Chebyshev gets there in ~9. Drop everything below the red line.", 12, "middle", GRAY)
print("05", f.save(OUT + "05_truncation.svg"))
