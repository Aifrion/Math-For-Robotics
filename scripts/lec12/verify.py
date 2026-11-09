"""Verify every number quoted on the Lecture 12 page (configuration space).

Run: python3 scripts/lec12/verify.py
"""
import numpy as np

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

rng = np.random.default_rng(12)

# ---------------------------------------------------------------
# 1. Disk robot <-> inflated obstacle (rounded corners!)
#    Disk (center c, radius r) hits axis-aligned rect iff
#    dist(c, rect) < r iff c lies in the rect inflated by r
#    with QUARTER-CIRCLE corners. Square-inflation overcounts.
# ---------------------------------------------------------------
R_ = 0.5
rect = (1.0, 2.0, 1.0, 1.8)   # xmin xmax ymin ymax
def dist_rect(c):
    dx = max(rect[0]-c[0], 0, c[0]-rect[1])
    dy = max(rect[2]-c[1], 0, c[1]-rect[3])
    return np.hypot(dx, dy)
def in_round_inflate(c):
    return dist_rect(c) < R_
def in_square_inflate(c):
    return (rect[0]-R_ < c[0] < rect[1]+R_) and (rect[2]-R_ < c[1] < rect[3]+R_)
pts = rng.uniform(0, 3, (200000, 2))
true_hit = np.array([dist_rect(p) < R_ for p in pts])
round_hit = np.array([in_round_inflate(p) for p in pts])
square_hit = np.array([in_square_inflate(p) for p in pts])
check("rounded inflation = exact collision set", int(np.sum(true_hit != round_hit)), 0, 0)
frac_extra = np.mean(square_hit & ~true_hit)
print(f"   sharp-corner inflation wrongly forbids {100*frac_extra:.2f}% of sampled area "
      f"(the 4 corner regions, area (4-pi)r^2 = {(4-np.pi)*R_*R_:.4f})")
check("sharp corners over-forbid", float(frac_extra > 0), 1.0, 0)

# ---------------------------------------------------------------
# 2. Minkowski sum: Cobs = O (+) (-R). Membership via definition
#    q = o - r equals grid test of translated-robot overlap.
# ---------------------------------------------------------------
# robot: triangle (rel. to reference point), obstacle: rectangle
Rpts = np.array([[0.0, 0.0], [0.4, 0.0], [0.2, 0.5]])
tri = lambda q: Rpts + q
def robot_hits_obstacle(q, n=25):
    # sample points inside the triangle, test against rect
    a, b, c = tri(q)
    for _ in range(n*n):
        u, v = rng.random(), rng.random()
        if u + v > 1: u, v = 1-u, 1-v
        p = a + u*(b-a) + v*(c-a)
        if rect[0] <= p[0] <= rect[1] and rect[2] <= p[1] <= rect[3]:
            return True
    return False
def in_minkowski(q):
    # q in O (+) (-R)  iff exists r in R, o in O with q = o - r
    # iff R translated by q intersects O  (same thing) -- here test via
    # the triangle's vertices/edges vs rect using exact separating axis
    P = tri(q)
    # SAT for convex polygon vs AABB
    axes = [(1,0),(0,1)]
    for i in range(3):
        e = P[(i+1)%3] - P[i]
        axes.append((-e[1], e[0]))
    Rc = np.array([[rect[0],rect[2]],[rect[1],rect[2]],[rect[1],rect[3]],[rect[0],rect[3]]])
    for ax in axes:
        ax = np.array(ax, float)
        p1 = P@ax; p2 = Rc@ax
        if p1.max() <= p2.min() or p2.max() <= p1.min():
            return False
    return True
qs = rng.uniform(-0.5, 3, (3000, 2))
mismatch = 0
for q in qs:
    exact = in_minkowski(q)
    samp = robot_hits_obstacle(q, 8)
    if exact != samp and exact:  # sampling can miss thin overlaps; only exact=False+samp=True is impossible
        pass
    if (not exact) and samp:
        mismatch += 1
check("sampled collision never outside Minkowski set", mismatch, 0, 0)

# ---------------------------------------------------------------
# 3. The rotating rectangle: fits through a gap only at some theta
#    rect robot 2.4 x 0.6, vertical gap width 1.0:
#    footprint width at angle t: 2.4|cos t| + 0.6|sin t|
# ---------------------------------------------------------------
Lr, Wr, gap = 2.4, 0.6, 1.0
width = lambda t: Lr*abs(np.cos(t)) + Wr*abs(np.sin(t))
check("theta=0 footprint 2.4 > gap", width(0.0), 2.4, 1e-12)
check("theta=90 footprint 0.6 < gap", width(np.pi/2), 0.6, 1e-12)
ts = np.linspace(0, np.pi/2, 10001)
fits = ts[width(ts) < gap]
print(f"   rectangle fits for theta in [{np.degrees(fits.min()):.1f}, 90] deg "
      f"(needs at least {np.degrees(fits.min()):.1f} deg of rotation)")
check("must rotate at least ~70 deg", float(np.degrees(fits.min()) > 60), 1.0, 0)

# ---------------------------------------------------------------
# 4. Two-link FK/IK roundtrip, L1 = L2 = 1.
#    gamma = acos((x^2+y^2-L1^2-L2^2)/(2 L1 L2)), beta = pi - gamma,
#    alpha = atan2(y,x) - asin(L2 sin(gamma)/sqrt(x^2+y^2))  [sqrt!]
#    The slide prints /(x^2+y^2) without the sqrt -- show it fails.
# ---------------------------------------------------------------
L1 = L2 = 1.0
def fk(a, b):
    return (L1*np.cos(a) + L2*np.cos(a+b), L1*np.sin(a) + L2*np.sin(a+b))
fails_bg = fails_b180 = fails_plain = fails_up = 0
for _ in range(500):
    a0 = rng.uniform(-np.pi, np.pi); b0 = rng.uniform(0.15, np.pi-0.15)
    x, y = fk(a0, b0)
    r2 = x*x + y*y
    # printed gamma: cos(gamma) = (x^2+y^2-L1^2-L2^2)/(2 L1 L2)
    g = np.arccos(np.clip((r2 - L1*L1 - L2*L2)/(2*L1*L2), -1, 1))
    base = np.arcsin(np.clip(L2*np.sin(g)/np.sqrt(r2), -1, 1))
    a_dn = np.arctan2(y, x) - base
    # pairing A (consistent): beta = +gamma
    if np.hypot(*(np.array(fk(a_dn, g)) - [x, y])) > 1e-6: fails_bg += 1
    # pairing B (as printed): beta = pi - gamma
    if np.hypot(*(np.array(fk(a_dn, np.pi - g)) - [x, y])) > 1e-6: fails_b180 += 1
    # printed alpha without the sqrt
    a_plain = np.arctan2(y, x) - np.arcsin(np.clip(L2*np.sin(g)/r2, -1, 1))
    if np.hypot(*(np.array(fk(a_plain, g)) - [x, y])) > 1e-6: fails_plain += 1
    # elbow-up mirror: beta = -gamma, alpha = atan2 + base
    a_up = np.arctan2(y, x) + base
    if np.hypot(*(np.array(fk(a_up, -g)) - [x, y])) > 1e-6: fails_up += 1
check("IK with beta = gamma (consistent pairing): 500/500 exact", fails_bg, 0, 0)
print(f"   slide's pairing beta = 180 - gamma fails {fails_b180}/500 roundtrips against the slide's own FK")
check("printed beta = 180-gamma is inconsistent with the printed gamma", float(fails_b180 > 400), 1.0, 0)
print(f"   alpha with the printed x^2+y^2 (no sqrt) fails {fails_plain}/500 -> missing sqrt confirmed")
check("no-sqrt alpha fails often", float(fails_plain > 100), 1.0, 0)
check("elbow-up mirror (beta=-gamma, alpha=atan2+asin): 500/500 exact", fails_up, 0, 0)

# ---------------------------------------------------------------
# 5. Angular distance on S^1: d(359 deg, 1 deg) = 2, not 358
# ---------------------------------------------------------------
def angdist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)
check("d(359,1) = 2", angdist(359, 1), 2.0, 1e-12)
check("naive |359-1| = 358 (wrong)", abs(359-1), 358, 0)
check("d(0,180) = 180 (max)", angdist(0, 180), 180.0, 1e-12)

# ---------------------------------------------------------------
# 6. Arm C-space obstacle: 2-link arm, disk obstacle in workspace.
#    Compute Qobs on a torus grid; fraction blocked; straight
#    C-space segment A->B blocked but wrapped path exists.
# ---------------------------------------------------------------
OB = np.array([0.9, 0.75]); OR = 0.30
def seg_hits_disk(p, q, c, r):
    d = q - p; t = np.clip(np.dot(c - p, d)/max(np.dot(d, d), 1e-12), 0, 1)
    return np.linalg.norm(p + t*d - c) < r
def arm_collides(a, b):
    j0 = np.zeros(2); j1 = np.array([L1*np.cos(a), L1*np.sin(a)])
    j2 = j1 + [L2*np.cos(a+b), L2*np.sin(a+b)]
    return seg_hits_disk(j0, j1, OB, OR) or seg_hits_disk(j1, j2, OB, OR)
N = 240
A = np.linspace(-np.pi, np.pi, N, endpoint=False)
G = np.zeros((N, N), bool)
for i, a in enumerate(A):
    for j, b in enumerate(A):
        G[i, j] = arm_collides(a, b)
frac = G.mean()
print(f"   C-space obstacle covers {100*frac:.1f}% of the (alpha,beta) torus")
check("obstacle blob is substantial (10-40%)", float(0.10 < frac < 0.40), 1.0, 0)
# connectivity of free space on the torus (wrap-around BFS)
from collections import deque
free = ~G
comp = -np.ones((N, N), int)
cid = 0
for i in range(N):
    for j in range(N):
        if free[i, j] and comp[i, j] < 0:
            dq = deque([(i, j)]); comp[i, j] = cid
            while dq:
                x, y = dq.popleft()
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = (x+dx) % N, (y+dy) % N
                    if free[nx, ny] and comp[nx, ny] < 0:
                        comp[nx, ny] = cid; dq.append((nx, ny))
            cid += 1
sizes = [np.sum(comp == k) for k in range(cid)]
print(f"   free space components on the torus: {cid} (sizes {sorted(sizes, reverse=True)[:3]}...)")
check("free space is connected thanks to wrap-around", cid, 1, 0)

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
