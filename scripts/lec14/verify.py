"""Verify every number quoted on the Lecture 14 page (differential geometry).

Run: python3 scripts/lec14/verify.py
"""
import numpy as np

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

# ---------------------------------------------------------------
# 1. The helix alpha(t) = (a cos t, a sin t, b t), a=2, b=1
#    speed c = sqrt(a^2+b^2); K = a/(a^2+b^2); tau = b/(a^2+b^2)
# ---------------------------------------------------------------
a, b = 2.0, 1.0
c = np.hypot(a, b)
al = lambda t: np.array([a*np.cos(t), a*np.sin(t), b*t])
d1 = lambda t: np.array([-a*np.sin(t), a*np.cos(t), b])
check("helix speed constant sqrt(a^2+b^2)", np.linalg.norm(d1(0.7)), c, 1e-12)
check("speed at another t (constant)", np.linalg.norm(d1(2.3)), c, 1e-12)
# arclength reparam beta(s) = alpha(s/c) has unit speed
h = 1e-6
beta = lambda s: al(s/c)
sp = np.linalg.norm((beta(1.0+h) - beta(1.0-h))/(2*h))
check("arc-length reparam has unit speed", sp, 1.0, 1e-6)

# analytic Frenet frame of the helix (arclength s, c = sqrt(a^2+b^2)):
#   T = (-a sin(s/c), a cos(s/c), b)/c
#   N = (-cos(s/c), -sin(s/c), 0)
#   B = ( b sin(s/c), -b cos(s/c), a)/c
Tf = lambda s: np.array([-a*np.sin(s/c), a*np.cos(s/c), b])/c
Nf = lambda s: np.array([-np.cos(s/c), -np.sin(s/c), 0.0])
Bf = lambda s: np.array([b*np.sin(s/c), -b*np.cos(s/c), a])/c
K_true = a/(a*a + b*b)
tau_true = b/(a*a + b*b)
s0 = 2.0
T, N, B = Tf(s0), Nf(s0), Bf(s0)
# confirm the analytic frame against raw numerics (h = 1e-4)
h2 = 1e-4
Tnum = (beta(s0+h2) - beta(s0-h2))/(2*h2)
Tpnum = (beta(s0+h2) - 2*beta(s0) + beta(s0-h2))/h2**2
check("analytic T matches numeric beta'", np.linalg.norm(Tnum - T), 0.0, 1e-7)
check("numeric |beta''| = K = a/(a^2+b^2) = 0.4", np.linalg.norm(Tpnum), K_true, 1e-6)
check("numeric beta'' direction = N", np.linalg.norm(Tpnum/np.linalg.norm(Tpnum) - N), 0.0, 1e-6)
check("B = T x N", np.linalg.norm(np.cross(T, N) - B), 0.0, 1e-12)
# torsion via analytic B'(s) = -tau N
Bp = (Bf(s0+h2) - Bf(s0-h2))/(2*h2)
tau = -Bp @ N
check("helix torsion tau = b/(a^2+b^2) = 0.2", tau, tau_true, 1e-8)
check("frame orthonormal: T.N", T@N, 0.0, 1e-14)
check("frame orthonormal: |B|", np.linalg.norm(B), 1.0, 1e-14)

# circle K = 1/a ; line K = 0
ca = 3.0
circ = lambda s: np.array([ca*np.cos(s/ca), ca*np.sin(s/ca), 0])
cp2 = (circ(1+1e-4) - 2*circ(1) + circ(1-1e-4))/1e-8
check("circle curvature = 1/a", np.linalg.norm(cp2), 1/ca, 1e-6)
line = lambda s: np.array([0.6, 0.8, 0])*s + np.array([1, 2, 3])
lp2 = (line(1+1e-4) - 2*line(1) + line(1-1e-4))/1e-8
check("line curvature = 0", np.linalg.norm(lp2), 0.0, 1e-6)

# ---------------------------------------------------------------
# 2. Frenet formulas: N' = -K T + tau B   (slide 23 prints +K T and the
#    matrix (1,2) entry K: verify the minus sign is required)
# ---------------------------------------------------------------
Np = (Nf(s0+h2) - Nf(s0-h2))/(2*h2)
good_resid = np.linalg.norm(Np - (-K_true*T + tau_true*B))
bad_resid  = np.linalg.norm(Np - (+K_true*T + tau_true*B))
print(f"   N' residual with -K T: {good_resid:.2e}; with slide's +K T: {bad_resid:.3f} (= 2K)")
check("N' = -K T + tau B holds", good_resid, 0.0, 1e-7)
check("slide's N' = +K T + tau B fails (residual = 2K)", bad_resid, 2*K_true, 1e-7)
Tp_ = (Tf(s0+h2) - Tf(s0-h2))/(2*h2)
check("T' = K N", np.linalg.norm(Tp_ - K_true*N), 0.0, 1e-7)
check("B' = -tau N", np.linalg.norm(Bp - (-tau_true*N)), 0.0, 1e-7)

# ---------------------------------------------------------------
# 3. Cornu spiral: integrate the 2D Frenet system with K(s) = s;
#    recover curvature numerically = s (clothoid, used in road design)
# ---------------------------------------------------------------
ds = 1e-3
S = np.arange(0, 3, ds)
th = 0.5*S*S                    # theta' = K = s
x = np.cumsum(np.cos(th))*ds
y = np.cumsum(np.sin(th))*ds
for s0 in (0.5, 1.5, 2.5):
    i = int(s0/ds)
    # numeric curvature from three points
    p0, p1, p2 = np.array([x[i-40], y[i-40]]), np.array([x[i], y[i]]), np.array([x[i+40], y[i+40]])
    aa = np.linalg.norm(p1-p0); bb = np.linalg.norm(p2-p1); cc = np.linalg.norm(p2-p0)
    d1_, d2_ = p1-p0, p2-p0
    area = abs(d1_[0]*d2_[1] - d1_[1]*d2_[0])/2
    Knum = 4*area/(aa*bb*cc)
    check(f"clothoid curvature at s={s0} equals s", Knum, s0, 0.01)

# ---------------------------------------------------------------
# 4. Covariant derivatives on the circle fields (slide 29):
#    W = (x,y)/r (radial), V = (-y,x)/r (tangential)
#    grad_V W = V/r ; grad_W V = 0
# ---------------------------------------------------------------
def Wf(p): r = np.linalg.norm(p); return p/r
def Vf(p): r = np.linalg.norm(p); return np.array([-p[1], p[0]])/r
def cov(F, dirv, p, eps=1e-6):
    return (F(p + eps*dirv) - F(p - eps*dirv))/(2*eps)
p = np.array([1.2, 0.9])
r = np.linalg.norm(p)
gVW = cov(Wf, Vf(p), p)
gWV = cov(Vf, Wf(p), p)
check("grad_V W = V/r (component match)", np.linalg.norm(gVW - Vf(p)/r), 0.0, 1e-6)
check("grad_W V = 0", np.linalg.norm(gWV), 0.0, 1e-6)
# constant fields: both derivatives vanish
gc = cov(lambda q: np.array([1.0, 0.0]), np.array([0.0, 1.0]), p)
check("constant fields: grad = 0", np.linalg.norm(gc), 0.0, 1e-9)

# ---------------------------------------------------------------
# 5. Parallel parking: car fields V_i = (cos phi_i cos th, cos phi_i sin th,
#    sin phi_i / l).  Lie bracket [V1,V2] = sin(phi1-phi2)/l (-sin th, cos th, 0)
# ---------------------------------------------------------------
l = 2.5
phi1, phi2 = 0.5, -0.3
def Vcar(phi):
    return lambda q: np.array([np.cos(phi)*np.cos(q[2]), np.cos(phi)*np.sin(q[2]), np.sin(phi)/l])
V1, V2 = Vcar(phi1), Vcar(phi2)
q0 = np.array([0.3, -0.2, 0.8])
def covq(F, dirv, q, eps=1e-6):
    return (F(q + eps*dirv) - F(q - eps*dirv))/(2*eps)
bracket = covq(V2, V1(q0), q0) - covq(V1, V2(q0), q0)   # grad_{V1}V2 - grad_{V2}V1
pred = np.sin(phi1 - phi2)/l*np.array([-np.sin(q0[2]), np.cos(q0[2]), 0])
check("[V1,V2] = sin(phi1-phi2)/l (-sin th, cos th, 0)", np.linalg.norm(bracket - pred), 0.0, 1e-5)
check("bracket vanishes when phi1 = phi2", np.linalg.norm(covq(Vcar(0.5), Vcar(0.5)(q0), q0) - covq(Vcar(0.5), Vcar(0.5)(q0), q0)), 0.0, 0)
# the bracket points SIDEWAYS: perpendicular to the heading, no rotation
heading = np.array([np.cos(q0[2]), np.sin(q0[2]), 0])
check("bracket is perpendicular to the heading", bracket@heading, 0.0, 1e-6)
check("bracket has no theta component", bracket[2], 0.0, 1e-6)

# the 4-phase wiggle: flow eps along V1, V2, -V1, -V2 -> net ~ eps^2 [V1,V2]
def flow(V, q, t, n=200):
    dt = t/n
    for _ in range(n):
        q = q + dt*V(q)
    return q
eps = 0.05
q = q0.copy()
for Vf_, sgn in ((V1, 1), (V2, 1), (V1, -1), (V2, -1)):
    q = flow((lambda Vf2, s: (lambda z: s*Vf2(z)))(Vf_, sgn), q, eps)
net = q - q0
ratio = np.linalg.norm(net)/ (eps*eps*np.linalg.norm(pred))
print(f"   4-phase wiggle, eps={eps}: net displacement {np.linalg.norm(net):.3e} "
      f"~ eps^2|[V1,V2]| = {eps*eps*np.linalg.norm(pred):.3e} (ratio {ratio:.3f})")
check("wiggle displacement ~ eps^2 bracket (ratio ~ 1)", ratio, 1.0, 0.1)
check("net motion is sideways (dot with pred > 0.99 cos)", net@pred/(np.linalg.norm(net)*np.linalg.norm(pred)), 1.0, 0.01)

# ---------------------------------------------------------------
# 6. Stereographic projection phi(P) = (x/(1-z), y/(1-z)): roundtrip
# ---------------------------------------------------------------
rng = np.random.default_rng(14)
for _ in range(5):
    v = rng.standard_normal(3); v /= np.linalg.norm(v)
    if v[2] > 0.99: continue
    u = np.array([v[0]/(1-v[2]), v[1]/(1-v[2])])
    d = 1 + u@u
    back = np.array([2*u[0]/d, 2*u[1]/d, (u@u - 1)/d])
    check("stereographic roundtrip", np.linalg.norm(back - v), 0.0, 1e-12)
# the chart blows up at the north pole
zs = np.array([0.9, 0.99, 0.999])
norms = np.sqrt(1 - zs**2)/(1 - zs)
print(f"   |phi(P)| as z -> 1: {norms.round(1)} -> infinity (north pole needs another chart)")
check("chart diverges at the pole", float(norms[-1] > 40), 1.0, 0)

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
