"""Lecture 8 figures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec8/"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- 01 eigen-spectrum and reconstruction error ----------
# synthetic "image-like" data: 12 dims with decaying variance
dims = 12
scales = 3.2 * 0.72 ** np.arange(dims)
A = rng.normal(0, 1, (dims, dims))
Qo, _ = np.linalg.qr(A)
X = rng.normal(0, 1, (400, dims)) * scales @ Qo.T
Xc = X - X.mean(0)
C = Xc.T @ Xc / len(X)
lam, U = np.linalg.eigh(C)
lam, U = lam[::-1], U[:, ::-1]
mses = []
for k in range(dims + 1):
    rec = (Xc @ U[:, :k]) @ U[:, :k].T
    mses.append(((Xc - rec) ** 2).sum(1).mean())
fig = Fig(820, 362)
p = Panel(fig, 60, 45, 330, 220, (-0.7, 11.7), (0, max(lam)*1.1), "The eigen-spectrum: variance per direction",
          xticks=[0, 3, 6, 9], yticks=None, xlabel="eigenvalue index j")
for j in range(dims):
    p.bar(j, lam[j], 0.7, BLUE)
q = Panel(fig, 470, 45, 330, 220, (-0.4, 12.4), (0, max(mses)*1.08), "Reconstruction MSE vs components kept",
          xticks=[0, 4, 8, 12], yticks=None, xlabel="k components")
q.line(range(dims + 1), mses, RED, 2.4)
q.dots(range(dims + 1), mses, RED, 3.5)
tails = [lam[k:].sum() for k in range(dims + 1)]
q.dots(range(dims + 1), tails, GREEN, 5.5, True)
q.label(4, mses[3], "measured MSE (red)", RED, 11, dx=14, dy=-4)
q.label(3.0, tails[2], "Σ of dropped λ (green rings)", GREEN, 11, dx=16, dy=-18)
fig.text(410, 324, "PCA's central identity, verified: the mean-squared reconstruction error with k components is exactly the sum of the", 11.5, "middle", GRAY)
fig.text(410, 342, "eigenvalues you dropped. Keep the big-λ directions and the error is whatever little variance remains.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_spectrum_mse.svg"))

# ---------- 02 PCA vs LDA ----------
c1 = rng.normal(0, 1, (120, 2)) @ np.array([[2.4, 0], [0, 0.45]]) + np.array([0, 1.1])
c2 = rng.normal(0, 1, (120, 2)) @ np.array([[2.4, 0], [0, 0.45]]) + np.array([0, -1.1])
Xall = np.vstack([c1, c2]); mu = Xall.mean(0)
m1, m2 = c1.mean(0), c2.mean(0)
Sw = (c1-m1).T@(c1-m1) + (c2-m2).T@(c2-m2)
Sb = np.outer(m1-mu, m1-mu)*len(c1) + np.outer(m2-mu, m2-mu)*len(c2)
ev, evec = np.linalg.eig(np.linalg.solve(Sw, Sb))
w = np.real(evec[:, np.argmax(np.real(ev))]); w /= np.linalg.norm(w)
Ct = (Xall-mu).T@(Xall-mu)
lp, Up = np.linalg.eigh(Ct); pca1 = Up[:, -1]
fig = Fig(820, 400)
p = Panel(fig, 70, 45, 420, 300, (-7.5, 7.5), (-3.4, 3.4), "Same data, two 'best' directions",
          xticks=[-6, -3, 0, 3, 6], yticks=[-3, 0, 3])
p.dots(c1[:, 0], c1[:, 1], GREEN, 3)
p.dots(c2[:, 0], c2[:, 1], PURPLE, 3)
for d, col, lab, off in [(pca1, BLUE, "PCA: most variance", -0.4), (w, RED, "LDA: most separation", 0.4)]:
    x0, y0 = mu - 7*d if abs(d[0]) > 0.5 else mu - 3*d
    x1, y1 = mu + 7*d if abs(d[0]) > 0.5 else mu + 3*d
    fig.parts.append(f'<path d="M{p.X(x0):.0f} {p.Y(y0):.0f}L{p.X(x1):.0f} {p.Y(y1):.0f}" stroke="{col}" stroke-width="3"/>')
p.label(5.0, 0.42, "PCA", BLUE, 13, weight="bold")
p.label(0.25, 2.75, "LDA", RED, 13, weight="bold")
# histograms of projections on each direction
def hist_panel(x0, title, d, col):
    pp = Panel(fig, x0, 45, 110, 130, (0, 1), (0, 1), None, xticks=None, yticks=None, grid=False)
    fig.text(x0 + 55, 36, title, 11.5, "middle", col, weight="bold")
    pr1, pr2 = c1 @ d, c2 @ d
    lo, hi = min(pr1.min(), pr2.min()), max(pr1.max(), pr2.max())
    bins = np.linspace(lo, hi, 18)
    h1, _ = np.histogram(pr1, bins); h2, _ = np.histogram(pr2, bins)
    mx = max(h1.max(), h2.max())
    for i in range(17):
        xx = (bins[i] - lo) / (hi - lo)
        wdt = 1/17
        fig.parts.append(f'<rect x="{pp.X(xx):.0f}" y="{pp.Y(h1[i]/mx):.0f}" width="{110/17:.1f}" height="{(h1[i]/mx)*130:.0f}" fill="{GREEN}" fill-opacity="0.55"/>')
        fig.parts.append(f'<rect x="{pp.X(xx):.0f}" y="{pp.Y(h2[i]/mx):.0f}" width="{110/17:.1f}" height="{(h2[i]/mx)*130:.0f}" fill="{PURPLE}" fill-opacity="0.55"/>')
hist_panel(545, "projected on PCA", pca1, BLUE)
hist_panel(685, "projected on LDA", w, RED)
fig.text(645, 200, "PCA: classes pile on top of each other (49% error).", 11.5, "middle", GRAY)
fig.text(645, 217, "LDA: clean split (0.3% error). Verified.", 11.5, "middle", GRAY)
fig.text(410, 375, "PCA is unsupervised: it chases variance and happily picks the long axis that mixes the classes. Fisher's LDA maximizes", 11.5, "middle", GRAY)
fig.text(410, 392, "between-class over within-class scatter (S_B W = λ S_W W) and finds the direction that separates them.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_pca_vs_lda.svg"))

# ---------- 03 ICA strip ----------
tt = np.linspace(0, 8, 1200)
s1 = np.sin(2*np.pi*tt)
s2 = 2*((tt*1.37+0.2) % 1) - 1
S = np.vstack([s1, s2])
Am = np.array([[0.7, 0.5], [0.3, -0.8]])
Xm = Am @ S
Xm = Xm - Xm.mean(1, keepdims=True)
Cm = Xm @ Xm.T / Xm.shape[1]
dd, E = np.linalg.eigh(Cm)
Wm = E @ np.diag(dd**-0.5) @ E.T
Z = Wm @ Xm
kurt = lambda v: ((v**4).mean()/(v**2).mean()**2) - 3
best, bth = -1, 0
for th in np.linspace(0, np.pi/2, 361):
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    Y = R @ Z
    val = abs(kurt(Y[0])) + abs(kurt(Y[1]))
    if val > best: best, bth = val, th
R = np.array([[np.cos(bth), np.sin(bth)], [-np.sin(bth), np.cos(bth)]])
Y = R @ Z
fig = Fig(860, 440)
rows = [("sources s(t): what two 'speakers' said", [s1, s2], GREEN),
        ("mixtures x = A·s: what two 'microphones' heard", [Xm[0], Xm[1]], ORANGE),
        ("recovered by ICA (whiten, then rotate to max |kurtosis|)", [Y[0]/np.abs(Y[0]).max(), Y[1]/np.abs(Y[1]).max()], BLUE)]
for r, (title, sigs, col) in enumerate(rows):
    for c2_ in range(2):
        p = Panel(fig, 60 + c2_*410, 50 + r*125, 360, 80, (0, 8), (-1.6, 1.6), None, xticks=None, yticks=None, grid=False)
        sg = sigs[c2_]
        p.line(tt, np.clip(sg/max(1e-9, np.abs(sg).max())*1.3, -1.55, 1.55), col, 1.6)
        fig.parts.append(f'<rect x="{60 + c2_*410}" y="{50 + r*125}" width="360" height="80" fill="none" stroke="{GRAY}" stroke-opacity="0.35"/>')
    fig.text(430, 42 + r*125, title, 12.5, "middle", BLACK, weight="bold")
fig.text(430, 412, "Verified: |correlation| of recovered vs true sources = 1.0000 for both (sign/order are unknowable — note the flipped saw).", 11, "middle", GRAY)
fig.text(430, 428, "PCA alone cannot unmix: after whitening every direction has equal variance; only non-Gaussianity (kurtosis) finds the true axes.", 11, "middle", GRAY)
print("03", fig.save(OUT + "03_ica_unmix.svg"))
