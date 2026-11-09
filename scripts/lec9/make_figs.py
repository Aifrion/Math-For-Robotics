"""Lecture 9 figures (calculus of variations)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec9/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"

x = np.linspace(0, 1, 200)
J = lambda y: np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

# ---------- 01: a functional eats a curve, returns a number ----------
fig = Fig(860, 345)
p = Panel(fig, 55, 45, 360, 240, (-0.02, 1.02), (-0.08, 1.35),
          "Each CURVE gets one number: its length L(y)",
          xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
curves = [(x, BLACK, 2.6, None, "L = 1.4142  (the straight line)"),
          (x + 0.15*np.sin(2*np.pi*x), ORANGE, 2.1, "6 4", "L = 1.5011"),
          (x + 0.30*np.sin(2*np.pi*x), RED, 2.1, "2 3", "L = 1.7983")]
for yy, col, w, dash, lab in curves:
    p.line(x, yy, col, w, dash)
p.dots([0, 1], [0, 1], AMBER, 5.5)
p.label(0.26, 0.62, "L = 1.4142", BLACK, 11.5, weight="bold", dy=-6)
p.label(0.21, 0.40, "1.5011", ORANGE, 11.5, weight="bold", dy=16)
p.label(0.30, 0.09, "1.7983", RED, 11.5, weight="bold", dy=6)
q = Panel(fig, 490, 45, 330, 240, (-0.36, 0.36), (1.40, 1.82),
          "Slice through function space: Φ(ε) = L(y* + ε·η)",
          xticks=[-0.3, 0, 0.3], yticks=[1.4, 1.6, 1.8], xlabel="ε")
eps = np.linspace(-0.35, 0.35, 120)
phi = [J(x + e*np.sin(2*np.pi*x)) for e in eps]
q.line(eps, phi, BLUE, 2.4)
q.dots([0], [np.sqrt(2)], GREEN, 6)
q.label(-0.335, 1.455, "Φ′(0) = 0: flat at the optimum", GREEN, 11, weight="bold")
q.vseg(0, 1.40, np.sqrt(2), GREEN, 1, "3 3")
fig.text(430, 330, "A functional maps a whole curve to one number — and at the best curve, Φ is flat along every slice y* + ε·η.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_functional.svg"))

# ---------- 02: isoperimetric — same area, different lengths ----------
from scipy.optimize import brentq
A = 0.30
def seg_area(t):
    R = 0.5/np.sin(t)
    return R*R*(t - np.sin(t)*np.cos(t))
t = brentq(lambda tt: seg_area(tt) - A, 1e-3, np.pi/2 + 1.2)
R = 0.5/np.sin(t); yc = -R*np.cos(t)
arc = yc + np.sqrt(np.maximum(R*R - (x-0.5)**2, 0))
para = 1.8*x*(1-x)                  # area exactly 0.3
tent = np.where(x < 0.5, 1.2*x, 1.2*(1-x))   # area 0.3
L_arc, L_para, L_tent = J(arc), J(para), J(tent)

fig = Fig(860, 360)
p = Panel(fig, 60, 45, 480, 250, (-0.03, 1.03), (-0.05, 0.68),
          f"Shortest curve holding area {A} below it (endpoints pinned)",
          xticks=[0, 0.5, 1], yticks=[0, 0.3, 0.6])
# shaded area under the arc
band = f'M{p.X(0):.1f} {p.Y(0):.1f}'
for a, b in zip(x, arc): band += f'L{p.X(a):.1f} {p.Y(b):.1f}'
band += f'L{p.X(1):.1f} {p.Y(0):.1f}Z'
p.f.parts.append(f'<path d="{band}" fill="{GREEN}" opacity="0.10"/>')
p.line(x, tent, GRAY, 1.8, "2 3")
p.line(x, para, ORANGE, 2.0, "6 4")
p.line(x, arc, GREEN, 2.8)
p.dots([0, 1], [0, 0], AMBER, 5.5)
p.label(0.5, 0.10, "area A = 0.3", GREEN, 11.5, "middle")
p.label(0.70, 0.52, f"tent  L = {L_tent:.4f}", GRAY, 11)
p.label(0.76, 0.40, f"parabola  L = {L_para:.4f}", ORANGE, 11)
p.label(0.54, 0.185, f"circle arc  L = {L_arc:.4f} ✓", GREEN, 11.5, weight="bold")
X0 = 572
fig.text(X0, 85, "Euler + Lagrange says:", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 110, "[F_y − d/dx F_y′]", 12, "start", BLUE)
fig.text(X0, 130, "  + λ₀[G_y − d/dx G_y′] = 0", 12, "start", BLUE)
fig.text(X0, 162, "solution: a circle arc with", 12, "start", BLACK)
fig.text(X0, 186, f"radius 1/λ₀ = {R:.3f}", 12.5, "start", GREEN, weight="bold")
fig.text(X0, 214, f"λ₀ recovered from the", 11.5, "start", GRAY)
fig.text(X0, 231, f"optimized curve: 1.962", 11.5, "start", GRAY)
fig.text(X0, 248, f"1/R = {1/R:.3f} ✓", 11.5, "start", GRAY)
fig.text(430, 338, "Three curves with the same area: the constraint multiplier λ₀ turns out to be the curve's curvature — tension against the constraint, made geometric.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_isoperimetric.svg"))
