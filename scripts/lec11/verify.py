"""Verify every number quoted on the Lecture 11 page (regression & classification).

Run: python3 scripts/lec11/verify.py
"""
import numpy as np

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

rng = np.random.default_rng(11)

# ---------------------------------------------------------------
# 1. Bolts vs needles (slides 54-57): minimum-distance classifier
#    m_needle = [0.86, 2.34], m_bolt = [5.74, 5.85]
#    d_j(x) = m_j . x - 0.5 ||m_j||^2  (expanded from ||x - m_j||)
# ---------------------------------------------------------------
mn = np.array([0.86, 2.34]); mb = np.array([5.74, 5.85])
check("needle bias -0.5||m||^2", -0.5*mn@mn, -3.1076, 1e-4)
check("bolt bias", -0.5*mb@mb, -33.585, 1e-3)
# slide prints d_needle = 0.86x1+2.34x2-3.10 and d_bolt = ...-33.59: consistent
# boundary d_n - d_b:
diff_w = mn - mb; diff_b = -0.5*mn@mn + 0.5*mb@mb
check("boundary w1 = -4.88", diff_w[0], -4.88, 1e-9)
check("boundary w2 = -3.51", diff_w[1], -3.51, 1e-9)
check("boundary bias = 30.48 (slide rounds to 30.49)", diff_b, 30.47745, 1e-4)
# min-distance = this linear rule
x = np.array([3.0, 4.0])
d_lin = (mn - mb)@x + diff_b
d_direct = np.linalg.norm(x - mb) - np.linalg.norm(x - mn)
check("linear rule sign == distance-rule sign", np.sign(d_lin), np.sign(d_direct), 0)

# ---------------------------------------------------------------
# 2. Polynomial regression on Bishop-style data: overfitting
#    10 noisy samples of sin(2 pi x); M = 1, 3, 9; ridge fixes M=9
# ---------------------------------------------------------------
N = 10
xs = np.linspace(0, 1, N)
ts = np.sin(2*np.pi*xs) + 0.25*rng.standard_normal(N)
xt = np.linspace(0, 1, 200)
tt = np.sin(2*np.pi*xt)
def design(x, M): return np.vander(x, M+1, increasing=True)
def fit(M, lam=0.0):
    P = design(xs, M)
    w = np.linalg.solve(lam*np.eye(M+1) + P.T@P, P.T@ts)
    return w
def rms(w, x, t):
    return np.sqrt(np.mean((design(x, len(w)-1)@w - t)**2))
res = {}
for M in (1, 3, 9):
    w = fit(M)
    res[M] = (rms(w, xs, ts), rms(w, xt, tt))
    print(f"   M={M}: train rms {res[M][0]:.4f}, test rms {res[M][1]:.4f}")
check("M=9 interpolates (train rms ~ 0)", res[9][0], 0.0, 1e-3)
check("M=9 overfits (test rms worst)", float(res[9][1] > res[3][1]), 1.0, 0)
check("M=3 generalizes best of the three", float(res[3][1] < res[1][1]), 1.0, 0)
w9r = fit(9, 1e-4)
print(f"   M=9 + ridge 1e-4: train {rms(w9r, xs, ts):.4f}, test {rms(w9r, xt, tt):.4f}")
check("ridge rescues M=9", float(rms(w9r, xt, tt) < res[9][1]), 1.0, 0)
# normal equations == polyfit
w3 = fit(3)
check("normal equations match np.polyfit", np.max(np.abs(w3 - np.polyfit(xs, ts, 3)[::-1])), 0.0, 1e-8)

# ---------------------------------------------------------------
# 3. Line-fit normal equations (slide 20): the LHS should be
#    (mean(x y), mean(y)), not (mean(y^2), mean(y))
# ---------------------------------------------------------------
a_true, b_true = 1.3, -0.4
xl = rng.uniform(0, 2, 60)
yl = a_true*xl + b_true          # noise-free so the right answer is exact
A = np.array([[np.mean(xl**2), np.mean(xl)], [np.mean(xl), 1.0]])
good = np.linalg.solve(A, [np.mean(xl*yl), np.mean(yl)])
bad  = np.linalg.solve(A, [np.mean(yl**2), np.mean(yl)])
check("correct RHS (mean xy) recovers a", good[0], a_true, 1e-9)
check("correct RHS recovers b", good[1], b_true, 1e-9)
print(f"   slide's printed RHS mean(y^2): gives a = {bad[0]:.3f}, b = {bad[1]:.3f} -> typo confirmed")
check("slide RHS is wrong", float(abs(bad[0]-a_true) > 0.05), 1.0, 0)

# ---------------------------------------------------------------
# 4. LMS sign (slide 24): w <- w + eta (t - w.phi) phi converges;
#    the slide's printed minus diverges.
# ---------------------------------------------------------------
def lms(sign, eta=0.05, passes=40):
    w = np.zeros(2)
    for _ in range(passes):
        for xi, ti in zip(xl, yl):
            phi = np.array([xi, 1.0])
            w = w + sign*eta*(ti - w@phi)*phi
        if not np.all(np.isfinite(w)) or np.linalg.norm(w) > 1e6:
            return w, False
    return w, True
w_plus, ok_plus = lms(+1)
w_minus, ok_minus = lms(-1)
check("LMS with + sign converges to (a,b)", np.max(np.abs(w_plus - [a_true, b_true])), 0.0, 1e-3)
check("LMS with slide's - sign blows up", float(not ok_minus), 1.0, 0)

# ---------------------------------------------------------------
# 5. Bayesian linear regression (Bishop's running example):
#    y = w0 + w1 x, truth (-0.3, 0.5), noise sd 0.2 (beta = 25),
#    prior alpha = 2. Posterior after 1, 2, 20 points.
# ---------------------------------------------------------------
W_TRUE = np.array([-0.3, 0.5])
beta, alpha = 25.0, 2.0
xb = rng.uniform(-1, 1, 20)
tb = W_TRUE[0] + W_TRUE[1]*xb + 0.2*rng.standard_normal(20)
def posterior(n):
    P = np.column_stack([np.ones(n), xb[:n]])
    SNinv = alpha*np.eye(2) + beta*P.T@P
    mN = beta*np.linalg.solve(SNinv, P.T@tb[:n])
    return mN, np.linalg.inv(SNinv)
for n in (1, 2, 20):
    mN, SN = posterior(n)
    sd = np.sqrt(np.diag(SN))
    print(f"   after {n:2d} pts: mean w = ({mN[0]:+.3f}, {mN[1]:+.3f}), sd = ({sd[0]:.3f}, {sd[1]:.3f})")
mN, SN = posterior(20)
check("posterior mean near truth w0", mN[0], W_TRUE[0], 0.12)
check("posterior mean near truth w1", mN[1], W_TRUE[1], 0.12)
m1, S1 = posterior(1); m20, S20 = posterior(20)
check("posterior shrinks with data", float(np.trace(S20) < 0.2*np.trace(S1)), 1.0, 0)

# ---------------------------------------------------------------
# 6. LSQ classification vs outliers (slide 61)
# ---------------------------------------------------------------
n1 = rng.multivariate_normal([-1, 1.5], [[0.6, 0.2], [0.2, 0.4]], 60)
n2 = rng.multivariate_normal([1, -1], [[0.6, 0.2], [0.2, 0.4]], 60)
def lsq_boundary(c2):
    X = np.vstack([n1, c2]); T = np.hstack([np.ones(len(n1)), -np.ones(len(c2))])
    Xt = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.lstsq(Xt, T, rcond=None)[0]
    return w, np.degrees(np.arctan2(-w[1], w[2]))
w0_, ang0 = lsq_boundary(n2)
outl = rng.multivariate_normal([7.5, -6], [[0.3, 0], [0, 0.3]], 12)
w1_, ang1 = lsq_boundary(np.vstack([n2, outl]))
def err(w):
    f = lambda P: w[0] + P@w[1:]
    return (np.sum(f(n1) < 0) + np.sum(f(n2) > 0))/(len(n1)+len(n2))
print(f"   LSQ boundary tilt without outliers {ang0:.1f} deg (err {100*err(w0_):.1f}%), "
      f"with far outliers {ang1:.1f} deg (err {100*err(w1_):.1f}%)")
check("outliers move the LSQ boundary (>8 deg)", float(abs(ang1-ang0) > 8), 1.0, 0)
check("outliers worsen error on the SAME core data", float(err(w1_) > err(w0_)), 1.0, 0)

# ---------------------------------------------------------------
# 7. Fisher's discriminant: w = Sw^-1 (m2 - m1) beats mean-difference
#    on correlated clusters; matches top eigenvector of Sw^-1 Sb
# ---------------------------------------------------------------
C = np.array([[1.2, 1.05], [1.05, 1.0]])    # strongly correlated
f1 = rng.multivariate_normal([0, 0], C, 300)
f2 = rng.multivariate_normal([1.6, 0.4], C, 300)
m1_, m2_ = f1.mean(0), f2.mean(0)
Sw = (f1 - m1_).T@(f1 - m1_) + (f2 - m2_).T@(f2 - m2_)
w_fisher = np.linalg.solve(Sw, m2_ - m1_); w_fisher /= np.linalg.norm(w_fisher)
w_naive = (m2_ - m1_)/np.linalg.norm(m2_ - m1_)
def class_err(w):
    p1, p2 = f1@w, f2@w
    thr = (p1.mean() + p2.mean())/2
    s = 1 if p2.mean() > p1.mean() else -1
    return (np.sum(s*(p1 - thr) > 0) + np.sum(s*(p2 - thr) < 0))/600
e_f, e_n = class_err(w_fisher), class_err(w_naive)
print(f"   misclassification: naive mean-difference {100*e_n:.1f}%, Fisher {100*e_f:.1f}%")
check("Fisher beats naive projection", float(e_f < e_n), 1.0, 0)
Sb = np.outer(m2_ - m1_, m2_ - m1_)
evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
v = np.real(evecs[:, np.argmax(np.real(evals))]); v /= np.linalg.norm(v)
check("Fisher w == top eigenvector of Sw^-1 Sb", min(np.linalg.norm(v - w_fisher), np.linalg.norm(v + w_fisher)), 0.0, 1e-8)

# ---------------------------------------------------------------
# 8. Perceptron: converges when separable, never on XOR
# ---------------------------------------------------------------
def perceptron(X, T, max_passes=200):
    Xt = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xt.shape[1]); updates = 0
    for _ in range(max_passes):
        changed = False
        for xi, ti in zip(Xt, T):
            if ti*(w@xi) <= 0:
                w = w + xi*ti; updates += 1; changed = True
        if not changed:
            return w, updates, True
    return w, updates, False
sep1 = rng.multivariate_normal([-1.1, 1.1], 0.22*np.eye(2), 40)
sep2 = rng.multivariate_normal([1.1, -1.1], 0.22*np.eye(2), 40)
X = np.vstack([sep1, sep2]); T = np.hstack([np.ones(40), -np.ones(40)])
w, upd, conv = perceptron(X, T)
print(f"   separable data: converged after {upd} updates")
check("perceptron converges on separable data", float(conv), 1.0, 0)
Xx = np.array([[0,0],[1,1],[0,1],[1,0]], float); Tx = np.array([1,1,-1,-1])
_, _, convx = perceptron(Xx, Tx)
check("perceptron never converges on XOR", float(not convx), 1.0, 0)

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
