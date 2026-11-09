"""Verify every number quoted on the Lecture 13 page (planning algorithms).

Run: python3 scripts/lec13/verify.py
"""
import numpy as np
import heapq
from collections import deque

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

rng = np.random.default_rng(13)

# ---------------------------------------------------------------
# 1. Resolution (in)completeness: a wall with a gap of width 0.08
#    at x=0.5. Coarse grid (10 cells, h=0.1) marks every wall cell
#    blocked -> no path. Fine grid (40 cells) finds one.
# ---------------------------------------------------------------
def build_grid(n):
    # world [0,1]^2, wall x in [0.48, 0.52], free gap y in [0.46, 0.54]
    g = np.zeros((n, n), bool)   # True = blocked
    for i in range(n):
        for j in range(n):
            x0, x1 = i/n, (i+1)/n
            y0, y1 = j/n, (j+1)/n
            overlap_wall = (x1 > 0.48) and (x0 < 0.52)
            inside_gap = (y0 >= 0.46) and (y1 <= 0.54)
            if overlap_wall and not inside_gap:
                g[i, j] = True   # cell partially blocked -> conservative block
    return g
def bfs_path(g, s, t):
    n = len(g);
    if g[s] or g[t]: return None
    prev = {s: None}; dq = deque([s])
    while dq:
        c = dq.popleft()
        if c == t: break
        for d in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (c[0]+d[0], c[1]+d[1])
            if 0 <= nb[0] < n and 0 <= nb[1] < n and not g[nb] and nb not in prev:
                prev[nb] = c; dq.append(nb)
    if t not in prev: return None
    path = [t]
    while path[-1] != s: path.append(prev[path[-1]])
    return path[::-1]
for n, expect in ((10, False), (40, True)):
    g = build_grid(n)
    p = bfs_path(g, (1, n//2), (n-2, n//2))
    print(f"   {n}x{n} grid: path {'found, length '+str(len(p)) if p else 'NOT found'}")
    check(f"{n}x{n} grid path exists = {expect}", float(p is not None), float(expect), 0)

# ---------------------------------------------------------------
# 2. Dijkstra vs greedy best-first vs A* on one weighted grid
# ---------------------------------------------------------------
N = 60
blocked = np.zeros((N, N), bool)
blocked[15:45, 28:32] = True          # a wall across the middle band
mud = np.zeros((N, N), bool)          # slow terrain below the wall
mud[45:58, 20:40] = True              # entering a mud cell costs 5
start, goal = (30, 5), (30, 55)
def cellcost(c):
    return 5.0 if mud[c] else 1.0
def nbrs(c):
    for d in ((1,0),(-1,0),(0,1),(0,-1)):
        nb = (c[0]+d[0], c[1]+d[1])
        if 0 <= nb[0] < N and 0 <= nb[1] < N and not blocked[nb]:
            yield nb
def path_cost(prev, gcost):
    return gcost[goal]
def search(h_weight, greedy=False):
    # Dijkstra: h_weight=0. A*: h_weight=1 (admissible Euclid). Greedy: f=h.
    gcost = {start: 0.0}
    pq = [(0.0, start)]; done = set(); expanded = 0
    prev = {}
    while pq:
        _, c = heapq.heappop(pq)
        if c in done: continue
        done.add(c); expanded += 1
        if c == goal: break
        for nb in nbrs(c):
            ng = gcost[c] + cellcost(nb)
            if nb not in gcost or ng < gcost[nb]:
                gcost[nb] = ng; prev[nb] = c
                h = np.hypot(nb[0]-goal[0], nb[1]-goal[1])
                f = h if greedy else ng + h_weight*h
                heapq.heappush(pq, (f, nb))
    # follow the tree actually returned to the caller
    if goal not in gcost: return expanded, None
    c = goal; cost = 0.0
    while c != start:
        cost += cellcost(c); c = prev[c]
    return expanded, cost
exp_d, cost_d = search(0.0)
exp_a, cost_a = search(1.0)
exp_g, cost_g = search(0.0, greedy=True)
exp_w, cost_w = search(3.0)
print(f"   Dijkstra: {exp_d} expansions, cost {cost_d}")
print(f"   A* (Euclid): {exp_a} expansions, cost {cost_a}")
print(f"   greedy best-first: {exp_g} expansions, cost {cost_g}")
print(f"   weighted A* (3h, inadmissible): {exp_w} expansions, cost {cost_w}")
check("A* cost == Dijkstra cost (optimal)", cost_a, cost_d, 0)
check("A* expands fewer than Dijkstra", float(exp_a < exp_d), 1.0, 0)
check("greedy expands fewest", float(exp_g < exp_a), 1.0, 0)
check("greedy path is suboptimal here", float(cost_g > cost_d), 1.0, 0)
# inadmissible weights VOID the guarantee but need not break optimality;
# here 3h still lands on 82 -- the point is it may not, while A* must.
check("weighted A* never beats the optimum", float(cost_w >= cost_d), 1.0, 0)
check("weighted A* much cheaper than plain A*", float(exp_w < 0.5*exp_a), 1.0, 0)

# ---------------------------------------------------------------
# 3. Navigation function (wavefront from goal): no local minima;
#    potential field with repulsive term: traps in a U-shape.
# ---------------------------------------------------------------
# wavefront
INF = 1 << 30
phi = np.full((N, N), INF)
phi[goal] = 0
dq = deque([goal])
while dq:
    c = dq.popleft()
    for nb in nbrs(c):
        if phi[nb] > phi[c] + 1:
            phi[nb] = phi[c] + 1; dq.append(nb)
minima = 0
for i in range(N):
    for j in range(N):
        if blocked[i, j] or (i, j) == goal or phi[i, j] >= INF: continue
        if all(phi[nb] >= phi[i, j] for nb in nbrs((i, j))):
            minima += 1
check("wavefront navigation function has NO local minima", minima, 0, 0)
check("phi[start] = unit-cost shortest step count (80; mud-aware cost is 82)",
      phi[start], 80.0, 0)
# potential field with a U-trap
Ublocked = np.zeros((N, N), bool)
Ublocked[20:40, 30] = True; Ublocked[20, 30:45] = True; Ublocked[39, 30:45] = True
gl = (30, 55); st = (30, 5)
def potential(c):
    att = np.hypot(c[0]-gl[0], c[1]-gl[1])**2 / 100
    rep = 0.0
    for i in range(max(0,c[0]-6), min(N,c[0]+7)):
        for j in range(max(0,c[1]-6), min(N,c[1]+7)):
            if Ublocked[i, j]:
                d = max(np.hypot(c[0]-i, c[1]-j), 0.5)
                if d < 6: rep += 2.0*(1/d - 1/6)**2
    return att + rep
c = st; stuck = False
for step in range(2000):
    if c == gl: break
    cands = [nb for nb in [(c[0]+d[0], c[1]+d[1]) for d in ((1,0),(-1,0),(0,1),(0,-1))]
             if 0 <= nb[0] < N and 0 <= nb[1] < N and not Ublocked[nb]]
    best = min(cands, key=potential)
    if potential(best) >= potential(c):
        stuck = True; break
    c = best
print(f"   potential-field descent: {'STUCK at '+str(c)+' after '+str(step)+' steps' if stuck else 'reached goal'}")
check("potential field gets trapped in the U (local minimum)", float(stuck), 1.0, 0)
check("the trap is NOT at the goal", float(c != gl), 1.0, 0)

# ---------------------------------------------------------------
# 4. Uniform random rotations: the slide's quaternion formula vs
#    naive uniform Euler angles. For uniform rotations, applying to
#    a fixed vector gives points uniform on the sphere: z ~ U[-1,1].
# ---------------------------------------------------------------
M = 60000
u1, u2, u3 = rng.random(M), rng.random(M), rng.random(M)
h = np.stack([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2),
              np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)], 1)
check("quaternions are unit (max |1-norm|)", np.max(np.abs(1 - np.linalg.norm(h, axis=1))), 0.0, 1e-12)
def rotate_z(q):   # rotate e_z by quaternion (w,x,y,z) -- slide order is (x?,..): treat as (a,b,c,d) generic
    w, x, y, z = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    # R e_z third column z-component: 1 - 2(x^2 + y^2)
    return 1 - 2*(x*x + y*y)
zq = rotate_z(h)
# Euler naive: yaw,pitch,roll ~ U; rotated e_z z-component = cos(pitch)... compute properly
yaw = rng.uniform(-np.pi, np.pi, M); pitch = rng.uniform(-np.pi/2, np.pi/2, M); roll = rng.uniform(-np.pi, np.pi, M)
# R = Rz(yaw)Ry(pitch)Rx(roll); (R e_z)_z = cos(pitch) cos(roll)
ze = np.cos(pitch)*np.cos(roll)
# uniform-on-sphere z has mean 0, var 1/3, and |z| mean 1/2
print(f"   quaternion method: E[z] = {zq.mean():+.4f}, var = {zq.var():.4f} (uniform: 0, 0.3333)")
print(f"   naive Euler:       E[z] = {ze.mean():+.4f}, var = {ze.var():.4f} -> biased")
check("quaternion z-mean ~ 0", zq.mean(), 0.0, 0.01)
check("quaternion z-var ~ 1/3", zq.var(), 1/3, 0.01)
check("Euler sampling is biased (var far from 1/3)", float(abs(ze.var() - 1/3) > 0.05), 1.0, 0)

# ---------------------------------------------------------------
# 5. Dispersion: 196 random points vs a 14x14 Sukharev grid
#    (largest empty-ball radius, approximated on a fine probe grid)
# ---------------------------------------------------------------
def dispersion(P):
    xs = np.linspace(0, 1, 60)
    best = 0.0
    for x in xs:
        for y in xs:
            d = np.min(np.hypot(P[:, 0]-x, P[:, 1]-y))
            best = max(best, d)
    return best
Prand = rng.random((196, 2))
gridpts = (np.arange(14) + 0.5)/14
Pgrid = np.array([(a, b) for a in gridpts for b in gridpts])
dr, dg = dispersion(Prand), dispersion(Pgrid)
print(f"   dispersion, 196 points: random {dr:.4f} vs Sukharev grid {dg:.4f}")
check("grid dispersion lower than random", float(dg < dr), 1.0, 0)
check("Sukharev dispersion ~ half cell diagonal", dg, np.hypot(0.5/14, 0.5/14), 0.01)

# ---------------------------------------------------------------
# 6. RRT: finds a path through a gap; tree growth numbers
# ---------------------------------------------------------------
OBS = [(0.45, 0.55, 0.0, 0.42), (0.45, 0.55, 0.58, 1.0)]   # wall with gap
def free(p):
    return not any(a <= p[0] <= b and c <= p[1] <= d for a, b, c, d in OBS)
def seg_free(p, q, k=20):
    return all(free(p + t*(q-p)) for t in np.linspace(0, 1, k))
def rrt(seed, step=0.06, iters=4000, goal=np.array([0.9, 0.5]), r=0.05):
    rr = np.random.default_rng(seed)
    V = [np.array([0.1, 0.5])]; parent = {0: None}
    for it in range(iters):
        q = goal if rr.random() < 0.05 else rr.random(2)
        i = int(np.argmin([np.linalg.norm(v - q) for v in V]))
        d = q - V[i]; n = np.linalg.norm(d)
        if n < 1e-9: continue
        new = V[i] + d/n*min(step, n)
        if free(new) and seg_free(V[i], new):
            parent[len(V)] = i; V.append(new)
            if np.linalg.norm(new - goal) < r:
                return it+1, len(V), True
    return iters, len(V), False
results = [rrt(s) for s in range(10)]
iters = [r[0] for r in results]; succ = sum(r[2] for r in results)
print(f"   RRT through the gap: {succ}/10 runs succeed, iterations median {int(np.median(iters))}")
check("RRT succeeds in all 10 seeded runs", succ, 10, 0)

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
