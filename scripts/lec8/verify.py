"""Lecture 8 (subspace methods) verification."""
import numpy as np
rng = np.random.default_rng(7)

print("=== PCA basics on a 2-D cloud ===")
n = 200
t = rng.normal(0, 2.0, n)
s = rng.normal(0, 0.5, n)
ang = np.deg2rad(30)
X = np.column_stack([t*np.cos(ang) - s*np.sin(ang), t*np.sin(ang) + s*np.cos(ang)])
mu = X.mean(0); Xc = X - mu
C = Xc.T @ Xc / n
lam, U = np.linalg.eigh(C)
lam, U = lam[::-1], U[:, ::-1]
print("eigenvalues:", lam.round(4), " (should be ~4 and ~0.25)")
print("principal axis angle:", np.rad2deg(np.arctan2(U[1,0], U[0,0])).round(2), "deg (should be ~30)")
# reconstruction MSE with 1 component == dropped eigenvalue
q = Xc @ U[:, :1]
Xrec = q @ U[:, :1].T
mse = ((Xc - Xrec)**2).sum(1).mean()
print("MSE with 1 PC:", mse.round(5), " = dropped eigenvalue:", lam[1].round(5), "✓" if abs(mse-lam[1])<1e-9 else "✗")

print("\n=== the small-matrix trick (n << m) ===")
m, n2 = 500, 8
Xi = rng.normal(0, 1, (m, n2))
Xi -= Xi.mean(1, keepdims=True)
Cbig = Xi @ Xi.T              # m x m -- expensive
Q = Xi.T @ Xi                 # n x n -- cheap
lq, Vq = np.linalg.eigh(Q); lq, Vq = lq[::-1], Vq[:, ::-1]
u0 = Xi @ Vq[:, 0] / np.sqrt(lq[0])
# compare with direct eigenvector of Cbig
lb, Ub = np.linalg.eigh(Cbig)
u_direct = Ub[:, -1]
align = abs(u0 @ u_direct)
print(f"eig of {m}x{m} via {n2}x{n2} matrix: |cos angle| = {align:.10f} (1 = identical)")

print("\n=== LDA vs PCA on two classes ===")
c1 = rng.normal(0, 1, (150, 2)) @ np.array([[2.4, 0], [0, 0.45]]) + np.array([0, 1.1])
c2 = rng.normal(0, 1, (150, 2)) @ np.array([[2.4, 0], [0, 0.45]]) + np.array([0, -1.1])
Xall = np.vstack([c1, c2]); mu_all = Xall.mean(0)
m1, m2 = c1.mean(0), c2.mean(0)
Sw = (c1 - m1).T @ (c1 - m1) + (c2 - m2).T @ (c2 - m2)
Sb = np.outer(m1 - mu_all, m1 - mu_all)*len(c1) + np.outer(m2 - mu_all, m2 - mu_all)*len(c2)
evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
w = np.real(evecs[:, np.argmax(np.real(evals))]); w /= np.linalg.norm(w)
Call = (Xall - mu_all).T @ (Xall - mu_all)
lp, Up = np.linalg.eigh(Call)
pca1 = Up[:, -1]
print("PCA direction:", pca1.round(3), "(points along the long, useless axis)")
print("LDA direction:", w.round(3), "(points along the class separation)")
def overlap(d):
    p1, p2 = c1 @ d, c2 @ d
    thr = (p1.mean() + p2.mean()) / 2
    sgn = 1 if p1.mean() > p2.mean() else -1
    err = ((sgn*(p1 - thr) < 0).sum() + (sgn*(p2 - thr) > 0).sum()) / 300
    return err
print(f"1-D classification error: project on PCA {100*overlap(pca1):.1f}%  vs LDA {100*overlap(w):.1f}%")

print("\n=== CCA tiny example ===")
z = rng.normal(0, 1, 500)
x = np.column_stack([z + 0.3*rng.normal(0,1,500), rng.normal(0,1,500)])
y = np.column_stack([rng.normal(0,1,500), -2*z + 0.5*rng.normal(0,1,500)])
xc = x - x.mean(0); yc = y - y.mean(0)
Cxx = xc.T@xc/500; Cyy = yc.T@yc/500; Cxy = xc.T@yc/500
isx = np.linalg.inv(np.linalg.cholesky(Cxx)).T
isy = np.linalg.inv(np.linalg.cholesky(Cyy)).T
A = isx.T @ Cxy @ isy  # Cxx^-1/2 Cxy Cyy^-1/2 (via cholesky)
Us, S, Vt = np.linalg.svd(A)
wx = isx @ Us[:, 0]; wy = isy @ Vt[0]
rho = (xc@wx) @ (yc@wy) / np.sqrt(((xc@wx)**2).sum() * ((yc@wy)**2).sum())
print("canonical correlations:", S.round(3), " check top rho:", rho.round(3))

print("\n=== ICA: unmix sine + sawtooth by kurtosis rotation ===")
tt = np.linspace(0, 8, 2000)
s1 = np.sin(2*np.pi*tt)
s2 = 2*((tt*1.37+0.2) % 1) - 1   # incommensurate period so sources are uncorrelated
S2 = np.vstack([s1, s2])
Amix = np.array([[0.7, 0.5], [0.3, -0.8]])
Xm = Amix @ S2
# whiten
Xm = Xm - Xm.mean(1, keepdims=True)
Cm = Xm @ Xm.T / Xm.shape[1]
d, E = np.linalg.eigh(Cm)
W = E @ np.diag(d**-0.5) @ E.T
Z = W @ Xm
kurt = lambda v: ((v**4).mean() / (v**2).mean()**2) - 3
best, bth = -1, 0
for th in np.linspace(0, np.pi/2, 361):
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    Y = R @ Z
    val = abs(kurt(Y[0])) + abs(kurt(Y[1]))
    if val > best: best, bth = val, th
R = np.array([[np.cos(bth), np.sin(bth)], [-np.sin(bth), np.cos(bth)]])
Y = R @ Z
c11 = max(abs(np.corrcoef(Y[0], s1)[0,1]), abs(np.corrcoef(Y[1], s1)[0,1]))
c22 = max(abs(np.corrcoef(Y[0], s2)[0,1]), abs(np.corrcoef(Y[1], s2)[0,1]))
print(f"best rotation {np.rad2deg(bth):.1f} deg; |corr(recovered, sine)| = {c11:.4f}, |corr(recovered, saw)| = {c22:.4f}")
print("kurtosis: sine", round(kurt(s1), 3), " saw", round(kurt(s2), 3), " mixture1", round(kurt(Z[0]), 3))
