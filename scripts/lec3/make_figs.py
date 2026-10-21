"""Lecture 3 figure: Vandermonde conditioning."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec3/"
os.makedirs(OUT, exist_ok=True)
ns = list(range(3, 21))
conds = [np.log10(np.linalg.cond(np.vander(np.linspace(0, 1, n), increasing=True))) for n in ns]
fig = Fig(760, 330)
p = Panel(fig, 70, 45, 620, 220, (2, 21), (0, 17), "Why nobody solves the Vandermonde system: cond(V) vs number of points",
          xticks=[5, 10, 15, 20], yticks=[0, 4, 8, 12, 16], xlabel="points n", ylabel="log10 cond(V)", logy=True)
p.line(ns, conds, RED, 2.4)
p.dots(ns, conds, RED, 3.5)
p.hline(16, GRAY, 1.2)
p.label(3.4, 16, "double precision holds ~16 digits — beyond this line the solve returns noise", GRAY, 11, dy=16)
fig.text(380, 310, "Fitting p(x) = a0 + a1x + ... through n points via V·a = y squares away one digit of accuracy every ~1.5 points.", 11.5, "middle", GRAY)
fig.text(380, 326, "The Lagrange form (and barycentric evaluation) sidesteps V entirely — same polynomial, stable arithmetic.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_vandermonde.svg"))
