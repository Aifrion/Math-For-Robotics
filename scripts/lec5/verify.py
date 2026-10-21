"""Lecture 5 (root finding) verification: every number on the page comes from here."""
import numpy as np

print("=== convergence race on f(x) = x^3 - 2x - 5 (root ~2.0946) ===")
f = lambda x: x**3 - 2*x - 5
fp = lambda x: 3*x**2 - 2
root = np.roots([1, 0, -2, -5])
root = root[np.isreal(root)].real[0]
print("true root:", root)

a, b = 2.0, 3.0
errs_bi = []
for i in range(12):
    c = (a + b) / 2
    errs_bi.append(abs(c - root))
    if f(a) * f(c) < 0: b = c
    else: a = c
x0, x1 = 2.0, 3.0
errs_sec = []
for i in range(9):
    if f(x1) == f(x0): break
    x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    errs_sec.append(abs(x2 - root))
    x0, x1 = x1, x2
a, b = 2.0, 3.0
errs_rf = []
for i in range(12):
    c = b - f(b) * (b - a) / (f(b) - f(a))
    errs_rf.append(abs(c - root))
    if f(a) * f(c) < 0: b = c
    else: a = c
x = 2.0
errs_nr = []
for i in range(6):
    x = x - f(x) / fp(x)
    errs_nr.append(abs(x - root))
print("bisection errors:", ["%.2e" % e for e in errs_bi[:8]])
print("regula falsi    :", ["%.2e" % e for e in errs_rf[:8]])
print("secant          :", ["%.2e" % e for e in errs_sec[:8]])
print("newton          :", ["%.2e" % e for e in errs_nr[:6]])
print("iterations to 1e-10: bisection", next(i for i,e in enumerate(errs_bi) if e<1e-10 or i==len(errs_bi)-1),
      "(cap)", " secant", next((i for i,e in enumerate(errs_sec) if e<1e-10), None),
      " newton", next((i for i,e in enumerate(errs_nr) if e<1e-10), None))
print("bisection evals for s0=1 to s=1e-10: log2(1e10) =", np.log2(1e10))

print("\n=== secant golden-ratio rate check ===")
e = errs_sec
for i in range(1, len(e)-1):
    if e[i+1] > 0 and e[i] > 0:
        print(f"  log e{i+1} / log e{i} = {np.log(e[i+1])/np.log(e[i]):.3f}")

print("\n=== Newton divergence example x^3 - 2x + 2 from x0=0 ===")
g = lambda x: x**3 - 2*x + 2
gp = lambda x: 3*x**2 - 2
x = 0.0
seq = [x]
for i in range(6):
    x = x - g(x)/gp(x)
    seq.append(x)
print("cycle:", seq)
print("real root of x^3-2x+2:", [r.real for r in np.roots([1,0,-2,2]) if abs(r.imag)<1e-9])

print("\n=== Descartes example p(x) = x^5 + 4x^4 - 3x^2 + x - 6 ===")
p = [1, 4, 0, -3, 1, -6]
def sign_changes(c):
    s = [np.sign(v) for v in c if v != 0]
    return sum(1 for i in range(len(s)-1) if s[i] != s[i+1])
print("sign changes p(x):", sign_changes(p), "-> 3 or 1 positive roots")
pm = [c * (-1)**(len(p)-1-i) for i, c in enumerate(p)]
print("p(-x) coeffs:", pm, "sign changes:", sign_changes(pm), "-> 2 or 0 negative roots")
r = np.roots(p)
print("actual roots:", np.round(r, 4))
print("actual: positive real:", sum(1 for z in r if abs(z.imag)<1e-9 and z.real>0),
      "negative real:", sum(1 for z in r if abs(z.imag)<1e-9 and z.real<0))

print("\n=== Sturm example x^5 + 5x^4 - 20x^2 - 10x + 2 ===")
p0 = np.poly1d([1, 5, 0, -20, -10, 2])
p1 = p0.deriv()
seq = [p0, p1]
while seq[-1].order > 0:
    _, rem = np.polydiv(seq[-2].coeffs, seq[-1].coeffs)
    rem = np.poly1d(-rem)
    rem = rem / abs(rem.coeffs[0])  # normalize like the slides
    seq.append(np.poly1d(rem))
for i, q in enumerate(seq):
    print(f"  f{i}: {np.round(q.coeffs, 4)}")
def var(x):
    vals = [q(x) for q in seq]
    s = [np.sign(v) for v in vals if v != 0]
    return sum(1 for i in range(len(s)-1) if s[i] != s[i+1])
pts = [-np.inf, -10, -5, -4, -3, -2, -1, 0, 1, 2, 5, 10, np.inf]
def vinf(sgn):
    vals = [np.sign(q.coeffs[0]) * (sgn ** (q.order % 2 if sgn<0 else 1)) for q in seq]
    # at +inf sign = sign of leading coeff; at -inf sign = sign(lead)*(-1)^degree
    vals = [np.sign(q.coeffs[0]) * ((-1)**q.order if sgn < 0 else 1) for q in seq]
    s = [v for v in vals if v != 0]
    return sum(1 for i in range(len(s)-1) if s[i] != s[i+1])
row = []
for x in pts:
    if np.isinf(x): row.append(vinf(-1 if x < 0 else 1))
    else: row.append(var(x))
print("  var at", pts, "=", row)
print("  roots of p0:", np.round(np.roots(p0.coeffs), 4))

print("\n=== Horner / deflation example p(x) = x^3 - 2x^2 - 4x + 3 ===")
c = [1, -2, -4, 3]
r = np.roots(c)
print("roots:", np.round(r, 6))
# Newton from x0=4 using Horner for p and p'
def horner(c, x):
    b = c[0]; d = 0
    for k in range(1, len(c)):
        d = d * x + b
        b = b * x + c[k]
    return b, d  # p(x), p'(x)
x = 4.0
for i in range(8):
    px, dpx = horner(c, x)
    x = x - px / dpx
print("newton from 4.0 ->", x)
# deflate by (x - x*)
b = [c[0]]
for k in range(1, len(c) - 1):
    b.append(c[k] + b[-1] * x)
print("deflated quadratic:", np.round(b, 6), " its roots:", np.round(np.roots(b), 6))

print("\n=== Cardano check on x^3 - 6x^2 + 11x - 6 (roots 1,2,3) ===")
pp, q, rr = -6, 11, -6
a = (3*q - pp**2) / 3
bb = (2*pp**3 - 9*pp*q + 27*rr) / 27
disc = bb**2/4 + a**3/27
print("depressed: y^3 + ay + b, a=", a, "b=", bb, "disc=", round(disc, 6), "(<0 -> 3 distinct real)")

print("\n=== Mueller on x^3 - 2x - 5 from 1, 2, 3 ===")
def mueller(f, x0, x1, x2, iters=8):
    out = []
    for _ in range(iters):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        h1, h2 = x1 - x0, x2 - x1
        d1, d2 = (f1 - f0)/h1, (f2 - f1)/h2
        aa = (d2 - d1) / (h2 + h1)
        bb = aa*h2 + d2
        cc = f2
        rad = np.sqrt(complex(bb*bb - 4*aa*cc))
        den = bb + rad if abs(bb + rad) > abs(bb - rad) else bb - rad
        dx = -2*cc / den
        x0, x1, x2 = x1, x2, x2 + dx
        out.append(x2)
    return out
seqm = mueller(f, 1, 2, 3)
print("mueller iterates:", [f"{abs(z-root):.2e}" for z in seqm[:6]])

print("\n=== 2-link arm IK as root finding (L1=L2=1, target x) ===")
# hand x for theta2 with theta1 chosen so hand is on the line y=0.3? keep 1D:
# f(t) = reach of elbow angle: solve |p(t)| = d  -> cos law. Use: f(t)=2+2cos t - d^2... simple:
d = 1.3
fik = lambda t: 2 + 2*np.cos(t) - d*d
tp = lambda t: -2*np.sin(t)
t = 2.5
for i in range(6):
    t = t - fik(t)/tp(t)
print("elbow angle for reach 1.3:", t, "check |p| =", np.sqrt(2+2*np.cos(t)))
print("closed form: acos((d^2-2)/2) =", np.arccos((d*d-2)/2))
