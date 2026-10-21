"""Lecture 6 figures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec6/"
os.makedirs(OUT, exist_ok=True)
w = (3 - np.sqrt(5)) / 2

# ---------- 01 golden section construction ----------
fig = Fig(760, 320)
y1, y2 = 90, 190
# top interval a..b..c, new point x in larger half
a, c = 60, 700
b = a + w * (c - a)
x = b + w * (c - b)
def seg(y, x0, x1, col, wd=3):
    fig.parts.append(f'<path d="M{x0:.0f} {y}H{x1:.0f}" stroke="{col}" stroke-width="{wd}" stroke-linecap="round"/>')
def tick(xp, y, lab, col=BLACK, dy=-10):
    fig.parts.append(f'<path d="M{xp:.0f} {y-7}V{y+7}" stroke="{col}" stroke-width="2"/>')
    fig.text(xp, y + dy - 4, lab, 13, "middle", col, weight="bold")
fig.text(380, 30, "Golden section: place the probe so both outcomes shrink the bracket the same way", 13, "middle", BLACK, weight="bold")
seg(y1, a, c, GRAY, 2)
tick(a, y1, "a"); tick(b, y1, "b", BLUE); tick(x, y1, "x", ORANGE); tick(c, y1, "c")
fig.parts.append(f'<path d="M{a} {y1+18}H{b}" stroke="{BLUE}" stroke-width="2"/>')
fig.text((a+b)/2, y1 + 34, "w = 0.382", 12, "middle", BLUE)
fig.parts.append(f'<path d="M{b} {y1+18}H{c}" stroke="{GRAY}" stroke-width="2"/>')
fig.text((b+c)/2, y1 + 34, "1 − w = 0.618", 12, "middle", GRAY)
# two outcomes
fig.text(180, y2 - 28, "if f(x) > f(b): keep [a, x]", 12.5, "middle", GREEN, weight="bold")
seg(y2, a, x, GREEN, 3)
tick(a, y2, "a", BLACK); tick(b, y2, "b", BLUE); tick(x, y2, "x", ORANGE)
fig.text((a+x)/2, y2 + 30, "new bracket = 0.618 of the old", 11.5, "middle", GRAY)
y3 = y2 + 88
fig.text(180, y3 - 28, "if f(x) < f(b): keep [b, c]", 12.5, "middle", GREEN, weight="bold")
seg(y3, b, c, GREEN, 3)
tick(b, y3, "b", BLUE); tick(x, y3, "x", ORANGE); tick(c, y3, "c", BLACK)
fig.text((b+c)/2, y3 + 30, "also 0.618 of the old — same ratios inside, so it repeats forever", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_golden_section.svg"))

# ---------- 02 simplex moves ----------
fig = Fig(860, 290)
tri = np.array([[0.15, 0.25], [0.5, 0.9], [0.95, 0.35]])  # b, m, w (best, mid, worst)
cent = (tri[0] + tri[1]) / 2
moves = [
    ("reflection", cent + 1.0 * (cent - tri[2]), "flip the worst point\nthrough the centroid", BLUE),
    ("expansion", cent + 2.0 * (cent - tri[2]), "reflection was great:\ngo twice as far", GREEN),
    ("contraction", cent + 0.5 * (tri[2] - cent), "reflection was bad:\npull inward instead", ORANGE),
    ("shrink", None, "nothing works: shrink\neverything toward the best", RED),
]
for i, (name, newp, desc, col) in enumerate(moves):
    ox, oy, s = 30 + i * 210, 60, 150
    P = lambda p: (ox + (p[0] + 0.75) / 1.9 * s, oy + (1 - p[1]) * s)
    pts = [P(p) for p in tri]
    fig.parts.append(f'<path d="M{pts[0][0]:.0f} {pts[0][1]:.0f}L{pts[1][0]:.0f} {pts[1][1]:.0f}L{pts[2][0]:.0f} {pts[2][1]:.0f}Z" fill="none" stroke="{GRAY}" stroke-width="1.6"/>')
    fig.text(*P(tri[0]), "", 1)
    bx, by = P(tri[0]); fig.text(bx - 8, by + 14, "best", 10.5, "middle", GRAY)
    wx, wy = P(tri[2]); fig.text(wx + 2, wy + 16, "worst", 10.5, "middle", GRAY)
    if name == "shrink":
        for q in tri[1:]:
            nq = tri[0] + 0.5 * (q - tri[0])
            x1, yy1 = P(q); x2, yy2 = P(nq)
            fig.parts.append(f'<path d="M{x1:.0f} {yy1:.0f}L{x2:.0f} {yy2:.0f}" stroke="{col}" stroke-width="1.2" stroke-dasharray="3 3"/>')
            fig.parts.append(f'<circle cx="{x2:.0f}" cy="{yy2:.0f}" r="4.5" fill="{col}"/>')
        n1, n2 = tri[0] + 0.5*(tri[1]-tri[0]), tri[0] + 0.5*(tri[2]-tri[0])
        q1, q2, q0 = P(n1), P(n2), P(tri[0])
        fig.parts.append(f'<path d="M{q0[0]:.0f} {q0[1]:.0f}L{q1[0]:.0f} {q1[1]:.0f}L{q2[0]:.0f} {q2[1]:.0f}Z" fill="{col}" fill-opacity="0.12" stroke="{col}" stroke-width="1.6"/>')
    else:
        cx, cy = P(cent)
        nx, ny = P(newp)
        fig.parts.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="3" fill="{GRAY}"/>')
        fig.parts.append(f'<path d="M{wx:.0f} {wy:.0f}L{nx:.0f} {ny:.0f}" stroke="{col}" stroke-width="1.2" stroke-dasharray="3 3"/>')
        fig.parts.append(f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="5" fill="{col}"/>')
        p0, p1 = P(tri[0]), P(tri[1])
        fig.parts.append(f'<path d="M{p0[0]:.0f} {p0[1]:.0f}L{nx:.0f} {ny:.0f}L{p1[0]:.0f} {p1[1]:.0f}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    fig.text(ox + s/2 + 10, 40, name, 13, "middle", col, weight="bold")
    for j, ln in enumerate(desc.split("\n")):
        fig.text(ox + s/2 + 10, 236 + j * 15, ln, 11, "middle", GRAY)
fig.text(430, 284, "One Nelder–Mead step = try these in order. The triangle oozes downhill, stretching along valleys and shrinking at the bottom.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_simplex_moves.svg"))

# ---------- 03 DP stage graph ----------
fig = Fig(760, 352)
nodes = {'S': (80, 165), 'A1': (280, 85), 'A2': (280, 245), 'B1': (500, 55), 'B2': (500, 165), 'B3': (500, 275), 'G': (690, 165)}
E = {('S','A1'):1, ('S','A2'):4, ('A1','B1'):4, ('A1','B2'):5, ('A2','B2'):1, ('A2','B3'):2,
     ('B1','G'):5, ('B2','G'):1, ('B3','G'):4}
ctg = {'G': 0, 'B1': 5, 'B2': 1, 'B3': 4, 'A1': min(4+5, 5+1), 'A2': min(1+1, 2+4), 'S': None}
ctg['S'] = min(1 + ctg['A1'], 4 + ctg['A2'])
opt = [('S','A2'), ('A2','B2'), ('B2','G')]
greedy = [('S','A1'), ('A1','B1'), ('B1','G')]
fig.text(380, 28, "Dynamic programming: label every node with its cost-to-go, then just walk downhill", 13, "middle", BLACK, weight="bold")
for (u, v), cst in E.items():
    x1, yy1 = nodes[u]; x2, yy2 = nodes[v]
    col, wd = GRAY, 1.3
    if (u, v) in opt: col, wd = GREEN, 3.5
    elif (u, v) in greedy: col, wd = RED, 2
    dash = ' stroke-dasharray="5 4"' if (u, v) in greedy else ''
    fig.parts.append(f'<path d="M{x1} {yy1}L{x2} {yy2}" stroke="{col}" stroke-width="{wd}"{dash}/>')
    mx, my = (x1+x2)/2, (yy1+yy2)/2 - 7
    fig.text(mx, my, str(cst), 12, "middle", col if col != GRAY else BLACK)
for n, (x, y) in nodes.items():
    fig.parts.append(f'<circle cx="{x}" cy="{y}" r="22" fill="#fff" stroke="{BLACK}" stroke-width="1.5"/>')
    fig.text(x, y - 2, n, 13, "middle", BLACK, weight="bold")
    fig.text(x, y + 13, str(ctg[n]), 11.5, "middle", BLUE, weight="bold")
fig.text(380, 322, "Blue number = cheapest cost from that node to G, filled right-to-left: cost-to-go(n) = min over edges (edge + cost-to-go(next)).", 11.5, "middle", GRAY)
fig.text(380, 340, "Green = optimal path (6). Red dashed = greedy 'always take the cheapest next edge' (10). Greedy can't see past one stage; DP sees to the end.", 11.5, "middle", GRAY)
print("03", fig.save(OUT + "03_dp_graph.svg"))
