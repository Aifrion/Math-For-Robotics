"""Lecture 7 (integration & ODEs) verification."""
import numpy as np

print("=== quadrature orders on I = int_0^pi sin = 2 ===")
f = np.sin; a, b = 0, np.pi; I = 2.0
def trap(N):
    x = np.linspace(a, b, N + 1); y = f(x); h = (b - a) / N
    return h * (y[0]/2 + y[1:-1].sum() + y[-1]/2)
def simp(N):
    x = np.linspace(a, b, N + 1); y = f(x); h = (b - a) / N
    return h/3 * (y[0] + 4*y[1:-1:2].sum() + 2*y[2:-2:2].sum() + y[-1])
def midp(N):
    h = (b - a) / N; x = a + (np.arange(N) + 0.5) * h
    return h * f(x).sum()
for N in [4, 8, 16, 32, 64]:
    print(f"  N={N:3d}  trap err {abs(trap(N)-I):.2e}  simp err {abs(simp(N)-I):.2e}  mid err {abs(midp(N)-I):.2e}")
print("  trap ratio:", abs(trap(16)-I)/abs(trap(32)-I), " simp ratio:", abs(simp(16)-I)/abs(simp(32)-I))

print("\n=== Romberg from trapezoid ===")
R = [[trap(2**k)] for k in range(1, 6)]
for k in range(1, 5):
    for j in range(1, k + 1):
        R[k].append(R[k][j-1] + (R[k][j-1] - R[k-1][j-1]) / (4**j - 1))
for k, row in enumerate(R):
    print("  " + "  ".join(f"{abs(v-I):.1e}" for v in row))

print("\n=== improper: int_0^1 x^(-1/2) dx = 2 via tanh rule ===")
g = lambda x: 1/np.sqrt(x)
aa, bb = 0.0, 1.0
t = np.linspace(-6, 6, 201)
x = 0.5*(bb+aa) + 0.5*(bb-aa)*np.tanh(t)
dxdt = 0.5*(bb-aa)/np.cosh(t)**2
val = np.trapezoid(g(x)*dxdt, t)
print("  tanh-rule trapezoid, 201 pts on t in [-6,6]:", val, " (naive trapezoid would hit 1/sqrt(0) = inf)")

print("\n=== ODE orders on y' = y, y(0)=1, to x=1 (e) ===")
def euler(h):
    y = 1.0
    for _ in range(round(1/h)): y += h*y
    return y
def rk2(h):
    y = 1.0
    for _ in range(round(1/h)):
        k1 = h*y; k2 = h*(y + k1/2); y += k2
    return y
def rk4(h):
    y = 1.0
    for _ in range(round(1/h)):
        k1 = h*y; k2 = h*(y+k1/2); k3 = h*(y+k2/2); k4 = h*(y+k3); y += k1/6+k2/3+k3/3+k4/6
    return y
E = np.e
for h in [0.1, 0.05, 0.025]:
    print(f"  h={h:<6} euler {abs(euler(h)-E):.2e}  rk2 {abs(rk2(h)-E):.2e}  rk4 {abs(rk4(h)-E):.2e}")
print("  halving ratios:", abs(euler(.05)-E)/abs(euler(.025)-E), abs(rk2(.05)-E)/abs(rk2(.025)-E), abs(rk4(.05)-E)/abs(rk4(.025)-E))

print("\n=== modified midpoint + Richardson (Bulirsch-Stoer core) ===")
def modmid(H, n, y0=1.0):
    h = H/n
    z0 = y0; z1 = z0 + h*z0
    for m in range(1, n):
        z0, z1 = z1, z0 + 2*h*z1
    return 0.5*(z1 + z0 + h*z1)
H = 1.0
yn, yn2 = modmid(H, 8), modmid(H, 4)
rich = (4*yn - yn2)/3
print(f"  modmid n=4: err {abs(yn2-E):.2e}, n=8: err {abs(yn-E):.2e}, Richardson (4y8-y4)/3: err {abs(rich-E):.2e}")

print("\n=== stiff system from the slides ===")
# u' = 998u + 1998v ; v' = -999u - 1999v; u0=1, v0=0
# solution u = 2e^-x - e^-1000x ; v = -e^-x + e^-1000x  -> verify
x = 0.37
u = 2*np.exp(-x) - np.exp(-1000*x); v = -np.exp(-x) + np.exp(-1000*x)
du = -2*np.exp(-x) + 1000*np.exp(-1000*x)
print("  residual u':", du - (998*u + 1998*v))
A = np.array([[998, 1998], [-999, -1999]])
print("  eigenvalues:", np.linalg.eigvals(A), " -> time scales 1 and 1/1000, ratio 1000 (stiffness)")
def euler_stiff(h, steps):
    w = np.array([1.0, 0.0])
    for _ in range(steps): w = w + h*(A @ w)
    return w
for h in [0.0019, 0.0021]:
    w = euler_stiff(h, round(0.5/h))
    exact = np.array([2*np.exp(-0.5), -np.exp(-0.5)])
    print(f"  explicit Euler h={h}: u(0.5) = {w[0]:.4g} (exact {exact[0]:.4f})  {'OK' if abs(w[0])<10 else 'BLOWS UP'}")
print("  stability limit h = 2/1000 = 0.002 even though e^-1000x is dead after x=0.01")
