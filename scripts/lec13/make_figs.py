"""Lecture 13 figures (planning algorithms)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import heapq
from collections import deque
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec13/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"

# ---------- shared search world (same as verify.py) ----------
N = 60
blocked = np.zeros((N, N), bool)
blocked[15:45, 28:32] = True
mud = np.zeros((N, N), bool)
mud[45:58, 20:40] = True
start, goal = (30, 5), (30, 55)
def cellcost(c): return 5.0 if mud[c] else 1.0
def nbrs(c):
    for d in ((1,0),(-1,0),(0,1),(0,-1)):
        nb = (c[0]+d[0], c[1]+d[1])
        if 0 <= nb[0] < N and 0 <= nb[1] < N and not blocked[nb]:
            yield nb
def search(h_weight, greedy=False):
    gcost = {start: 0.0}; pq = [(0.0, start)]; done = set(); prev = {}
    while pq:
        _, c = heapq.heappop(pq)
        if c in done: continue
        done.add(c)
        if c == goal: break
        for nb in nbrs(c):
            ng = gcost[c] + cellcost(nb)
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng; prev[nb] = c
                h = np.hypot(nb[0]-goal[0], nb[1]-goal[1])
                f = h if greedy else ng + h_weight*h
                heapq.heappush(pq, (f, nb))
    path = [goal]; cost = 0.0
    while path[-1] != start:
        cost += cellcost(path[-1]); path.append(prev[path[-1]])
    return done, path[::-1], cost

fig = Fig(860, 380)
cases = [(search(0.0), "Dijkstra", "3149 expanded · cost 82 ✓"),
         (search(1.0), "A* (Euclidean h)", "1825 expanded · cost 82 ✓"),
         (search(0.0, True), "greedy best-first", "152 expanded · cost 132 ✗")]
for k, ((done, path, cost), title, lab) in enumerate(cases):
    p = Panel(fig, 42 + k*278, 45, 250, 250, (0, N), (0, N), title, xticks=None, yticks=None, grid=False)
    cell = 250/N
    def runs(predicate, col, opacity):
        # merge consecutive cells per row into one rect each
        d = ""
        for i in range(N):
            j = 0
            while j < N:
                if predicate(i, j):
                    j0 = j
                    while j < N and predicate(i, j): j += 1
                    d += f"M{p.X(j0):.0f} {p.Y(i)-cell:.1f}h{(j-j0)*cell:.1f}v{cell:.1f}h-{(j-j0)*cell:.1f}Z"
                else:
                    j += 1
        fig.parts.append(f'<path d="{d}" fill="{col}" opacity="{opacity}"/>')
    runs(lambda i, j: mud[i, j], AMBER, 0.25)
    runs(lambda i, j: (i, j) in done, BLUE, 0.3)
    runs(lambda i, j: blocked[i, j], BLACK, 0.8)
    # path
    dp = ""
    for t, (i, j) in enumerate(path):
        dp += ("M" if t == 0 else "L") + f"{p.X(j)+cell/2:.0f} {p.Y(i)-cell/2:.0f}"
    fig.parts.append(f'<path d="{dp}" stroke="{"#16a34a" if cost < 100 else RED}" stroke-width="2.4" fill="none"/>')
    fig.parts.append(f'<circle cx="{p.X(start[1])+cell/2:.0f}" cy="{p.Y(start[0])-cell/2:.0f}" r="4" fill="{BLACK}"/>')
    fig.parts.append(f'<circle cx="{p.X(goal[1])+cell/2:.0f}" cy="{p.Y(goal[0])-cell/2:.0f}" r="4" fill="{AMBER}"/>')
    fig.text(42 + k*278 + 125, 318, lab, 11.5, "middle", GREEN if cost < 100 else RED, weight="bold")
fig.text(430, 352, "Blue = expanded cells. A* focuses Dijkstra without losing optimality; greedy ignores cost-so-far and pays 132 in the mud (amber).", 11, "middle", GRAY)
print("01", fig.save(OUT + "01_search_race.svg"))

# ---------- 02 wavefront on a small grid ----------
W, H = 12, 8
wall = set()
for j in range(2, 7): wall.add((4, j))
for i in range(4, 7): wall.add((i, 6))
for j in range(8, 11): wall.add((2, j))
gl = (6, 10)
INF = 999
phi = {(i, j): INF for i in range(H) for j in range(W) if (i, j) not in wall}
phi[gl] = 0
dq = deque([gl])
while dq:
    c = dq.popleft()
    for d in ((1,0),(-1,0),(0,1),(0,-1)):
        nb = (c[0]+d[0], c[1]+d[1])
        if nb in phi and phi[nb] > phi[c] + 1:
            phi[nb] = phi[c] + 1; dq.append(nb)
fig = Fig(860, 360)
p = Panel(fig, 70, 45, 480, 270, (0, W), (0, H), "wavefront: breadth-first ripple out from the GOAL", xticks=None, yticks=None, grid=False)
cw, ch = 480/W, 270/H
for i in range(H):
    for j in range(W):
        x, y = p.X(j), p.Y(i) - ch
        if (i, j) in wall:
            fig.parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cw:.0f}" height="{ch:.0f}" fill="{BLACK}" opacity="0.8"/>')
        else:
            v = phi[(i, j)]
            o = max(0.04, 0.5 - 0.03*v)
            fig.parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cw:.0f}" height="{ch:.0f}" fill="{GREEN}" opacity="{o:.2f}" stroke="#e5e7eb" stroke-width="0.5"/>')
            fig.text(x + cw/2, y + ch/2 + 4, str(v), 11.5, "middle", BLACK)
fig.parts.append(f'<circle cx="{p.X(gl[1])+cw/2:.0f}" cy="{p.Y(gl[0])-ch/2:.0f}" r="14" fill="none" stroke="{AMBER}" stroke-width="2.5"/>')
# descent path from start (1,1)
c = (1, 1); dp = f"M{p.X(c[1])+cw/2:.0f} {p.Y(c[0])-ch/2:.0f}"
while phi[c] > 0:
    c = min((nb for nb in [(c[0]+d[0], c[1]+d[1]) for d in ((1,0),(-1,0),(0,1),(0,-1))] if nb in phi), key=lambda n: phi[n])
    dp += f"L{p.X(c[1])+cw/2:.0f} {p.Y(c[0])-ch/2:.0f}"
fig.parts.append(f'<path d="{dp}" stroke="{RED}" stroke-width="2" opacity="0.65" fill="none"/>')
X0 = 596
fig.text(X0, 90, "cost-to-go as a navigation", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 108, "FUNCTION (a policy):", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 136, "from ANY cell, step to any", 11.5, "start", GRAY)
fig.text(X0, 153, "neighbor with smaller φ.", 11.5, "start", GRAY)
fig.text(X0, 181, "verified: no local minima —", 11.5, "start", GREEN)
fig.text(X0, 198, "greedy descent cannot stall", 11.5, "start", GREEN)
fig.text(X0, 226, "(a hand-built potential field", 11.5, "start", RED)
fig.text(X0, 243, "on the same idea got stuck", 11.5, "start", RED)
fig.text(X0, 260, "in a U-trap after 23 steps)", 11.5, "start", RED)
fig.text(430, 342, "BFS from the goal turns the grid into a controller: the red path is pure greedy descent of φ.", 11, "middle", GRAY)
print("02", fig.save(OUT + "02_wavefront.svg"))

# ---------- 03 dispersion: random vs grid ----------
rng = np.random.default_rng(13)
Prand = rng.random((196, 2))
gp = (np.arange(14) + 0.5)/14
Pgrid = np.array([(a, b) for a in gp for b in gp])
def worst_gap(P):
    xs = np.linspace(0, 1, 70); best = (0, 0, 0)
    for x in xs:
        for y in xs:
            d = np.min(np.hypot(P[:, 0]-x, P[:, 1]-y))
            if d > best[0]: best = (d, x, y)
    return best
fig = Fig(860, 340)
for k, (P, title, col) in enumerate([(Prand, "196 pseudorandom samples", RED), (Pgrid, "196-point Sukharev grid", GREEN)]):
    p = Panel(fig, 90 + k*400, 45, 240, 240, (0, 1), (0, 1), title, xticks=None, yticks=None, grid=False)
    d, x, y = worst_gap(P)
    fig.parts.append(f'<circle cx="{p.X(x):.1f}" cy="{p.Y(y):.1f}" r="{d*240:.1f}" fill="{col}" opacity="0.15" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    dd = "".join(f"M{p.X(a):.0f} {p.Y(b):.0f}h.1" for a, b in P)
    fig.parts.append(f'<path d="{dd}" stroke="{BLUE}" stroke-width="3" stroke-linecap="round" fill="none"/>')
    fig.text(90 + k*400 + 120, 308, f"dispersion (largest empty ball) = {d:.3f}", 11.5, "middle", col, weight="bold")
fig.text(430, 330, "Random sampling leaves big holes (0.160); a low-dispersion grid caps the largest hole at 0.051.", 11, "middle", GRAY)
print("03", fig.save(OUT + "03_dispersion.svg"))
