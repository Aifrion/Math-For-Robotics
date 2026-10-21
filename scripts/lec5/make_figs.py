"""Lecture 5 figures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec5/"
os.makedirs(OUT, exist_ok=True)

# ---------- 01 convergence race ----------
f = lambda x: x**3 - 2*x - 5
fp = lambda x: 3*x**2 - 2
root = [r.real for r in np.roots([1, 0, -2, -5]) if abs(r.imag) < 1e-9][0]
def run(n):
    a, b = 2.0, 3.0; bi = []
    for _ in range(n):
        c = (a+b)/2; bi.append(abs(c-root))
        if f(a)*f(c) < 0: b = c
        else: a = c
    a, b = 2.0, 3.0; rf = []
    for _ in range(n):
        c = b - f(b)*(b-a)/(f(b)-f(a)); rf.append(abs(c-root))
        if f(a)*f(c) < 0: b = c
        else: a = c
    x0, x1 = 2.0, 3.0; se = []
    for _ in range(n):
        if f(x1) == f(x0): se.append(se[-1]); continue
        x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0)); se.append(abs(x2-root)); x0, x1 = x1, x2
    x = 3.0; nw = []
    for _ in range(n):
        x = x - f(x)/fp(x); nw.append(abs(x-root))
    return bi, rf, se, nw
bi, rf, se, nw = run(13)
fig = Fig(760, 360)
p = Panel(fig, 70, 45, 620, 250, (0, 13), (-16, 0.5), "Error vs iteration on x³ − 2x − 5, bracket [2, 3]",
          xticks=[0, 2, 4, 6, 8, 10, 12], yticks=[0, -4, -8, -12, -16], xlabel="iteration", ylabel="log10 |error|", logy=True)
L = lambda e: [max(-16, np.log10(v)) if v > 0 else -16 for v in e]
ks = list(range(1, 14))
p.line(ks, L(bi), GRAY, 2.2)
p.line(ks, L(rf), PURPLE, 2.2)
p.line(ks, L(se), BLUE, 2.2)
p.line(ks, L(nw), GREEN, 2.2)
p.dots(ks, L(bi), GRAY, 3); p.dots(ks, L(rf), PURPLE, 3); p.dots(ks, L(se), BLUE, 3); p.dots(ks, L(nw), GREEN, 3)
p.legend([("bisection — slope fixed: ×½ per step", GRAY, None), ("regula falsi — superlinear-ish", PURPLE, None),
          ("secant — order 1.618", BLUE, None), ("Newton — order 2: digits double", GREEN, None)], x=96, y=200)
fig.text(380, 345, "Linear convergence = straight line. Superlinear = curve that dives. Newton reaches 10⁻¹⁵ in 4 steps; bisection needs ~33 for 10⁻¹⁰.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_convergence_race.svg"))

# ---------- 02 Sturm number line ----------
p0 = np.poly1d([1, 5, 0, -20, -10, 2])
roots = sorted(np.roots(p0.coeffs).real)
fig = Fig(820, 240)
pts = [-5, -4, -3, -2, -1, 0, 1, 2, 5]
var = {-5: 5, -4: 5, -3: 4, -2: 3, -1: 3, 0: 2, 1: 1, 2: 0, 5: 0}
p = Panel(fig, 40, 70, 740, 90, (-5.5, 5.5), (0, 1), "Sturm's theorem on x⁵ + 5x⁴ − 20x² − 10x + 2: sign changes count roots",
          xticks=pts, yticks=None, grid=False)
fig.parts.append(f'<path d="M{p.X(-5.5):.0f} {p.Y(0.35):.0f}H{p.X(5.5):.0f}" stroke="{BLACK}" stroke-width="1.4"/>')
for x in pts:
    fig.text(p.X(x), p.Y(0.72), f"V = {var[x]}", 12, "middle", BLUE, weight="bold")
    fig.parts.append(f'<path d="M{p.X(x):.0f} {p.Y(0.3):.0f}V{p.Y(0.42):.0f}" stroke="{GRAY}" stroke-width="1"/>')
for r in roots:
    fig.parts.append(f'<circle cx="{p.X(r):.0f}" cy="{p.Y(0.35):.0f}" r="5.5" fill="{RED}"/>')
pairs = [(-4, -3), (-3, -2), (-1, 0), (0, 1), (1, 2)]
for a, b in pairs:
    fig.parts.append(f'<path d="M{p.X(a)+6:.0f} {p.Y(0.12):.0f}H{p.X(b)-6:.0f}" stroke="{GREEN}" stroke-width="3" stroke-linecap="round"/>')
fig.text(410, 205, "V(x) = sign changes down the Sturm sequence at x. V drops by 1 exactly when x passes a real root,", 12, "middle", GRAY)
fig.text(410, 222, "so V(a) − V(b) = number of real roots in (a, b). Green bars: the 5 intervals the table isolates; red dots: the actual roots.", 12, "middle", GRAY)
print("02", fig.save(OUT + "02_sturm_line.svg"))

# ---------- 03 cubic discriminant cases ----------
fig = Fig(860, 300)
cases = [
    ("y³ − 3y + 6:  Δ > 0", [1, 0, -3, 6], "one real root + complex pair", ORANGE),
    ("y³ − 3y + 2:  Δ = 0", [1, 0, -3, 2], "three real, two equal (touches zero)", PURPLE),
    ("y³ − 3y + 1:  Δ < 0", [1, 0, -3, 1], "three distinct real roots", GREEN),
]
xs = np.linspace(-2.8, 2.8, 160)
for i, (title, c, sub, col) in enumerate(cases):
    p = Panel(fig, 55 + i * 275, 45, 230, 190, (-2.8, 2.8), (-9, 11), title,
              xticks=[-2, 0, 2], yticks=[-5, 0, 5, 10] if i == 0 else [])
    ys = np.polyval(c, xs)
    fig.parts.append(f'<path d="M{p.X(-2.8):.0f} {p.Y(0):.0f}H{p.X(2.8):.0f}" stroke="{GRAY}" stroke-width="1" stroke-dasharray="4 3"/>')
    p.line(xs, ys, col, 2.4)
    rts = [z.real for z in np.roots(c) if abs(z.imag) < 1e-9]
    p.dots(rts, [0]*len(rts), RED, 5)
    fig.text(55 + i * 275 + 115, 272, sub, 11.5, "middle", GRAY)
fig.text(430, 293, "Same cubic family y³ − 3y + b: the discriminant Δ = b²/4 + a³/27 just asks whether the local hump dips across zero.", 11.5, "middle", GRAY)
print("03", fig.save(OUT + "03_cubic_cases.svg"))
