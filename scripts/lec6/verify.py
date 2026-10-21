"""Lecture 6 (optimization) verification."""
import numpy as np

print("=== golden section ===")
w = (3 - np.sqrt(5)) / 2
print("w =", w, " check w^2-3w+1 =", w*w - 3*w + 1)
print("bracket shrink factor per evaluation: 1-w =", 1 - w, "(the golden ratio conjugate)")
# demonstrate on f(x)=x^2-4x-9 like the slide's parabola-ish curve (min at 2)
f = lambda x: (x - 2)**2 - 10
a, b, c = -2, 1, 10
for i in range(40):
    if c - b > b - a:
        x = b + w * (c - b)
        if f(x) < f(b): a, b = b, x
        else: c = x
    else:
        x = b - w * (b - a)
        if f(x) < f(b): c, b = b, x
        else: a = x
print("golden section 40 evals ->", b, " bracket", c - a)

print("\n=== parabolic interpolation step ===")
a, b, c = 0.0, 1.0, 4.0
num = (b-a)**2 * (f(b)-f(c)) - (b-c)**2 * (f(b)-f(a))
den = (b-a) * (f(b)-f(c)) - (b-c) * (f(b)-f(a))
x = b - 0.5 * num / den
print("one parabolic step from (0,1,4) on (x-2)^2-10:", x, "(exact min = 2: quadratic -> one shot)")

print("\n=== steepest descent vs conjugate gradient on quadratic ===")
# f = 1/2 x^T A x - b^T x, A spd with condition number kappa
def run_sd(A, bb, x0, iters):
    x = x0.copy(); errs = []
    xs = [x.copy()]
    xstar = np.linalg.solve(A, bb)
    for _ in range(iters):
        g = A @ x - bb
        lam = (g @ g) / (g @ A @ g)
        x = x - lam * g
        xs.append(x.copy()); errs.append(np.linalg.norm(x - xstar))
    return np.array(xs), errs
def run_cg(A, bb, x0, iters):
    x = x0.copy(); xstar = np.linalg.solve(A, bb)
    g = A @ x - bb; h = -g
    xs = [x.copy()]; errs = []
    for _ in range(iters):
        lam = -(g @ h) / (h @ A @ h)
        x = x + lam * h
        gn = A @ x - bb
        gamma = (gn @ gn) / (g @ g)
        h = -gn + gamma * h
        g = gn
        xs.append(x.copy()); errs.append(np.linalg.norm(x - xstar))
    return np.array(xs), errs
A = np.array([[1.0, 0], [0, 25.0]])  # kappa = 25
bb = np.zeros(2); x0 = np.array([10.0, 1.0])
_, e_sd = run_sd(A, bb, x0, 20)
_, e_cg = run_cg(A, bb, x0, 3)
print("steepest descent errors (kappa=25):", ["%.1e" % e for e in e_sd[:8]])
print("conjugate gradient errors        :", ["%.1e" % e for e in e_cg])
print("-> CG exact in n=2 steps on a 2-D quadratic; SD contracts by (k-1)/(k+1) =", (25-1)/(25+1), "per step")

print("\n=== Nelder-Mead on Rosenbrock ===")
def rosen(p): x, y = p; return (1-x)**2 + 100*(y-x*x)**2
def nelder_mead(f, x0, iters=200):
    n = len(x0)
    simplex = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float); p[i] += 0.5; simplex.append(p)
    evals = 0
    for k in range(iters):
        simplex.sort(key=f)
        x0c = np.mean(simplex[:-1], axis=0)
        xr = x0c + 1.0 * (x0c - simplex[-1])
        if f(xr) < f(simplex[0]):
            xe = x0c + 2.0 * (xr - x0c)
            simplex[-1] = xe if f(xe) < f(xr) else xr
        elif f(xr) < f(simplex[-2]):
            simplex[-1] = xr
        else:
            xc = x0c + 0.5 * (simplex[-1] - x0c)
            if f(xc) < f(simplex[-1]): simplex[-1] = xc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5*(p - simplex[0]) for p in simplex[1:]]
    simplex.sort(key=f)
    return simplex[0], f(simplex[0])
p, v = nelder_mead(rosen, [-1.2, 1.0])
print("NM on Rosenbrock from (-1.2,1):", p, "f =", v)

print("\n=== simulated annealing on a 12-city TSP ===")
rng = np.random.default_rng(3)
C = rng.random((12, 2))
def tour_len(t):
    return sum(np.linalg.norm(C[t[i]] - C[t[(i+1) % len(t)]]) for i in range(len(t)))
t = list(range(12)); L0 = tour_len(t)
best = L0; cur = L0
T = 1.0
for k in range(20000):
    T = 1.0 * (0.9995 ** k)
    i, j = sorted(rng.integers(1, 12, 2))
    if i == j: continue
    t2 = t[:i] + t[i:j+1][::-1] + t[j+1:]
    L2 = tour_len(t2)
    if L2 < cur or rng.random() < np.exp(-(L2 - cur) / max(T, 1e-12)):
        t, cur = t2, L2
        best = min(best, cur)
print(f"random order: {L0:.3f} -> annealed: {cur:.3f} (best {best:.3f})")

print("\n=== DP stage graph (own example, verified by brute force) ===")
# stages: S -> A1/A2 -> B1/B2/B3 -> G
E = {('S','A1'):1, ('S','A2'):4,
     ('A1','B1'):4, ('A1','B2'):5, ('A2','B2'):1, ('A2','B3'):2,
     ('B1','G'):5, ('B2','G'):1, ('B3','G'):4}
import itertools
paths = [['S','A1','B1','G'],['S','A1','B2','G'],['S','A2','B2','G'],['S','A2','B3','G']]
costs = {tuple(p): sum(E[(p[i],p[i+1])] for i in range(len(p)-1)) for p in paths}
for p, cst in costs.items(): print(" ", "->".join(p), "=", cst)
print("best:", min(costs.items(), key=lambda kv: kv[1]))
