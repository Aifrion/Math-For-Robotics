"""Verify every number quoted on the Lecture 4 page (function approximation).

Run: python3 scripts/lec4/verify.py
"""
import numpy as np

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

E = np.e

# ---------------------------------------------------------------
# 1. Minimax (best L-inf) LINE for f(x) = e^x on [-1, 1]
#    p(x) = a + b x. Equioscillation needs n+2 = 3 alternating
#    extreme points: x = -1, x1 (interior), x = +1.
#    Slope: error e(x)=f-p has e(-1) = e(1)  =>  b = (f(1)-f(-1))/2
# ---------------------------------------------------------------
b = (E - 1/E) / 2
check("slope b = (e - 1/e)/2", b, 1.1752011936438014, 1e-12)

# interior extremum: e'(x1) = e^{x1} - b = 0  =>  x1 = ln b
x1 = np.log(b)
check("x1 = ln b", x1, 0.16143936157119028, 1e-12)

# intercept a: error at -1 equals MINUS error at x1 (alternation)
#   f(-1) - (a - b) = -( f(x1) - (a + b*x1) )
#   => a = ( f(-1) + f(x1) + b - b*x1 ) / 2
a = (np.exp(-1) + np.exp(x1) + b - b*x1) / 2
check("intercept a", a, 1.2642790490197413, 1e-12)

# max error (should be attained, alternating, at -1, x1, +1)
xs = np.linspace(-1, 1, 2_000_001)
err = np.exp(xs) - (a + b*xs)
dmax = np.max(np.abs(err))
check("max |error| (minimax distance)", dmax, 0.2788015857955024, 1e-7)
e_m1 = np.exp(-1) - (a - b)
e_x1 = np.exp(x1) - (a + b*x1)
e_p1 = np.exp(1) - (a + b)
print(f"   equioscillation: e(-1)={e_m1:+.10f}  e(x1)={e_x1:+.10f}  e(1)={e_p1:+.10f}")
check("e(-1) = +dmax", e_m1, dmax, 1e-9)
check("e(x1) = -dmax", e_x1, -dmax, 1e-9)
check("e(+1) = +dmax", e_p1, dmax, 1e-9)

# rounded values quoted on the page
check("b rounded", round(b, 4), 1.1752, 0)
check("x1 rounded", round(x1, 5), 0.16144, 0)
check("a rounded", round(a, 4), 1.2643, 0)
check("dmax rounded", round(dmax, 4), 0.2788, 0)

# sanity: compare to numerical best line via Remez-style direct optimization
from scipy.optimize import minimize
res = minimize(lambda p: np.max(np.abs(np.exp(xs) - (p[0] + p[1]*xs))),
               [1.2, 1.2], method="Nelder-Mead", options={"xatol":1e-12,"fatol":1e-14})
print(f"   numeric best line: a={res.x[0]:.8f} b={res.x[1]:.8f} err={res.fun:.10f}")
check("numeric optimum matches a", res.x[0], a, 1e-5)
check("numeric optimum matches b", res.x[1], b, 1e-5)

# the two rival lines in Fig 1
check("Taylor line 1+x worst error = e-2", np.max(np.abs(np.exp(xs)-(1+xs))), E-2, 1e-7)
ls_a, ls_b = np.sinh(1), 3/E   # L2-best line on [-1,1] (Legendre projection)
check("least-squares line worst error", np.max(np.abs(np.exp(xs)-(ls_a+ls_b*xs))), 0.43944, 1e-5)

# ---------------------------------------------------------------
# 2. Lower bound via divided differences, nodes -1, 0, 1:
#    d(f, pi_1) >= |f[-1,0,1]| / sum 1/|w'(x_i)|
#    f[-1,0,1] = (f(-1) - 2 f(0) + f(1)) / 2 ; sum = 1/2+1+1/2 = 2
#    => bound = (f(-1) - 2 f(0) + f(1)) / 4
# ---------------------------------------------------------------
dd2 = (np.exp(-1) - 2*1 + np.exp(1)) / 2   # f[-1,0,1]
check("f[-1,0,1]", dd2, 0.5430806348152437, 1e-12)
w_abs = [abs((-1-0)*(-1-1)), abs((0+1)*(0-1)), abs((1+1)*(1-0))]  # |w'| at -1,0,1
sum_inv = sum(1/v for v in w_abs)
check("sum 1/|w'(x_i)|", sum_inv, 2.0, 1e-15)
bound = dd2 / sum_inv
check("lower bound (e - 2 + 1/e)/4", bound, 0.27154031740762185, 1e-12)
check("bound rounded", round(bound, 4), 0.2715, 0)
print(f"   bound {bound:.4f} <= true distance {dmax:.4f}: {bound <= dmax}")

# ---------------------------------------------------------------
# 3. Chebyshev polynomials
# ---------------------------------------------------------------
# recurrence vs definition on a grid
xg = np.linspace(-1, 1, 1001)
T = [np.ones_like(xg), xg.copy()]
for n in range(2, 9):
    T.append(2*xg*T[-1] - T[-2])
for n in range(9):
    direct = np.cos(n*np.arccos(xg))
    check(f"T_{n} recurrence == cos(n arccos x)", np.max(np.abs(T[n]-direct)), 0.0, 1e-10)

# zeros of T_N at cos(pi(k+1/2)/N)
N = 7
zeros = np.cos(np.pi*(np.arange(N)+0.5)/N)
check("T_7 at its zeros", np.max(np.abs(np.cos(N*np.arccos(zeros)))), 0.0, 1e-12)

# discrete orthogonality: sum over zeros of T_N of T_i T_j
for (i, j, want) in [(2, 5, 0.0), (3, 3, N/2), (0, 0, N)]:
    s = np.sum(np.cos(i*np.arccos(zeros))*np.cos(j*np.arccos(zeros)))
    check(f"discrete orthogonality T_{i}.T_{j}", s, want, 1e-10)

# continuous orthogonality with weight (1-x^2)^(-1/2) via theta substitution
from scipy.integrate import quad
for (i, j, want) in [(2, 4, 0.0), (3, 3, np.pi/2), (0, 0, np.pi)]:
    v, _ = quad(lambda th: np.cos(i*th)*np.cos(j*th), 0, np.pi)
    check(f"continuous orthogonality T_{i}.T_{j}", v, want, 1e-10)

# leading coefficient 2^(N-1), |T_N| <= 1
coef = np.polynomial.chebyshev.cheb2poly([0]*N + [1])
check("leading coeff of T_7 = 2^6", coef[-1], 64.0, 1e-9)

# ---------------------------------------------------------------
# 4. Chebyshev approximation of f(x) = e^x, degree 1 comparison
#    c_j = (2/N) sum f(x_k) T_j(x_k) over zeros of T_N
# ---------------------------------------------------------------
def cheb_coeffs(f, N, m):
    xk = np.cos(np.pi*(np.arange(N)+0.5)/N)
    return [(2/N)*np.sum(f(xk)*np.cos(j*np.arccos(xk))) for j in range(m)]

c = cheb_coeffs(np.exp, 40, 10)
# degree-1 Chebyshev approx: p = c0/2 + c1 x
p1 = c[0]/2 + c[1]*xg
err1 = np.max(np.abs(np.exp(xg) - p1))
print(f"   degree-1 Chebyshev line: a={c[0]/2:.4f} b={c[1]:.4f} maxerr={err1:.4f} (minimax {dmax:.4f})")
check("Chebyshev line near-minimax (within 16%)", float(err1 < 1.16*dmax), 1.0, 0)
check("Chebyshev line maxerr rounded", round(err1, 4), 0.3219, 0)

# ---------------------------------------------------------------
# 5. Truncated power series vs Chebyshev: 1/(2+x) on [0, 1], eps=1e-9
#    power series: 1/(2+x) = (1/2) sum (-x/2)^n; worst at x = 1,
#    remainder = (1/2)^n  =>  30 terms for 1e-9.
#    Chebyshev on the warped interval y in [-1,1], x = (y+1)/2.
# ---------------------------------------------------------------
f = lambda x: 1/(2+x)
def pow_err(n):
    xs2 = np.linspace(0, 1, 20001)
    acc = np.zeros_like(xs2)
    for k in range(n):
        acc += 0.5*(-xs2/2)**k
    return np.max(np.abs(f(xs2) - acc))
n_pow = next(n for n in range(1, 60) if pow_err(n) < 1e-9)
print(f"   power series terms for 1e-9 on [0,1]: {n_pow} (error with 29 terms: {pow_err(29):.2e})")
# remainder BOUND (1/2)^n crosses 1e-9 at n=30; measured error crosses one
# term sooner because the series alternates and partial sums overshoot less
check("remainder bound (1/2)^29 still > 1e-9", float(0.5**29 > 1e-9), 1.0, 0)
check("remainder bound (1/2)^30 < 1e-9", float(0.5**30 < 1e-9), 1.0, 0)
check("measured terms ~30", float(n_pow in (29, 30)), 1.0, 0)

# Chebyshev of the warped function g(y) = f((y+1)/2)
g = lambda y: f((y+1)/2)
cs = cheb_coeffs(g, 60, 30)
yk = np.linspace(-1, 1, 20001)
def cheb_err(m):
    approx = sum(cs[k]*np.cos(k*np.arccos(yk)) for k in range(m)) - cs[0]/2
    return np.max(np.abs(g(yk) - approx))
n_cheb = next(m for m in range(1, 30) if cheb_err(m) < 1e-9)
print(f"   Chebyshev terms for 1e-9: {n_cheb}  (err with {n_cheb-1}: {cheb_err(n_cheb-1):.2e}, with {n_cheb}: {cheb_err(n_cheb):.2e})")
check("Chebyshev needs ~9-10 terms", float(n_cheb in (9, 10)), 1.0, 0)

# decay ratio: g(y) = 2/(5+y), pole at y=-5; rho = 5 + sqrt(24) ~ 9.90 => ~10x/term
rho = 5 + np.sqrt(24)
ratio = abs(cs[10]/cs[9])
check("coeff decay ratio ~ 1/(5+sqrt24)", ratio, 1/rho, 2e-3)
print(f"   decay per term: {ratio:.5f} ~ 1/(5+sqrt24) = {1/rho:.5f} (vs 0.5 for power series)")

# ---------------------------------------------------------------
# 6. Domain warping: y = (x - (b+a)/2) / ((b-a)/2) maps [a,b]->[-1,1]
# ---------------------------------------------------------------
aa, bb = 2.0, 7.0
warp = lambda x: (x - (bb+aa)/2) / ((bb-aa)/2)
check("warp(a) = -1", warp(aa), -1.0, 1e-15)
check("warp(b) = +1", warp(bb), 1.0, 1e-15)
# the slide's printed version with minus in numerator does NOT map correctly:
bad = lambda x: (x - (bb-aa)/2) / ((bb-aa)/2)
print(f"   slide's printed formula maps a,b -> {bad(aa):.2f},{bad(bb):.2f} (not -1,1) — typo confirmed")

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
