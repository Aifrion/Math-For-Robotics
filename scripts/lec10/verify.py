"""Verify every number quoted on the Lecture 10 page (spectral analysis).

Run: python3 scripts/lec10/verify.py
"""
import numpy as np

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

rng = np.random.default_rng(10)

# ---------------------------------------------------------------
# 1. Fourier series of a square wave (amplitude 1, period 1):
#    b_q = 4/(pi q) for odd q, 0 for even. Partial-sum max error.
# ---------------------------------------------------------------
T = 1.0
t = np.linspace(0, T, 4096, endpoint=False)
sq = np.sign(np.sin(2*np.pi*t)); sq[sq == 0] = 1
for q in [1, 2, 3, 4, 5]:
    bq = 2/T*np.trapezoid(sq*np.sin(2*np.pi*q*t), t)
    want = 4/(np.pi*q) if q % 2 else 0.0
    check(f"square-wave b_{q}", bq, want, 1e-3)
    aq = 2/T*np.trapezoid(sq*np.cos(2*np.pi*q*t), t)
    check(f"square-wave a_{q} = 0", aq, 0.0, 1e-3)
def partial(n):
    s = np.zeros_like(t)
    for q in range(1, n+1, 2):
        s += 4/(np.pi*q)*np.sin(2*np.pi*q*t)
    return s
for n, rms_want in [(1, None), (5, None), (49, None)]:
    r = np.sqrt(np.mean((partial(n) - sq)**2))
    print(f"   square wave, terms up to q={n}: rms error {r:.4f}")

# ---------------------------------------------------------------
# 2. Discrete Fourier coefficients (slide's A_q, B_q formulas)
#    on x_n = 2 + 3 cos(2 pi 2 n/N) + 1.5 sin(2 pi 3 n/N)
# ---------------------------------------------------------------
N = 64
n = np.arange(1, N+1)          # slide indexes n = 1..N
x = 2 + 3*np.cos(2*np.pi*2*n/N) + 1.5*np.sin(2*np.pi*3*n/N)
A0 = np.mean(x)
A2 = 2/N*np.sum(x*np.cos(2*np.pi*2*n/N))
B3 = 2/N*np.sum(x*np.sin(2*np.pi*3*n/N))
A3 = 2/N*np.sum(x*np.cos(2*np.pi*3*n/N))
check("A_0 = mean = 2", A0, 2.0, 1e-10)
check("A_2 = 3", A2, 3.0, 1e-10)
check("B_3 = 1.5", B3, 1.5, 1e-10)
check("A_3 = 0", A3, 0.0, 1e-10)

# ---------------------------------------------------------------
# 3. Nyquist & aliasing: f_c = 1/(2h). Sample 7 Hz at 10 Hz ->
#    indistinguishable from 3 Hz (and the DFT peak lands at 3 Hz).
# ---------------------------------------------------------------
fs, f = 10.0, 7.0
check("Nyquist of 10 Hz sampling", fs/2, 5.0, 0)
ts = np.arange(0, 10, 1/fs)     # 100 samples -> 0.1 Hz bins, 3 Hz on-grid
xs = np.sin(2*np.pi*f*ts)
alias = np.sin(2*np.pi*(fs - f)*ts)   # 3 Hz, sign-flipped phase
check("7 Hz and -3 Hz agree at every sample", np.max(np.abs(xs + alias)), 0.0, 1e-9)
F = np.fft.rfft(xs)
fgrid = np.fft.rfftfreq(len(xs), 1/fs)
fpeak = fgrid[np.argmax(np.abs(F))]
check("DFT peak appears at 3 Hz", fpeak, 3.0, 1e-9)

# ---------------------------------------------------------------
# 4. ACF of a rectangular pulse of height A, width theta is the
#    triangle A^2 theta (1 - |tau|/theta) [per-unit-time version:
#    using finite-energy convention R(tau) = int f(t)f(t+tau)dt]
# ---------------------------------------------------------------
A_, th = 1.5, 2.0
dt = 0.001
tt = np.arange(-1, 4, dt)
pulse = np.where((tt >= 0) & (tt < th), A_, 0.0)
R = np.correlate(pulse, pulse, "full")*dt
lags = (np.arange(len(R)) - (len(pulse)-1))*dt
for tau in [0.0, 0.5, 1.0, 1.9]:
    i = np.argmin(np.abs(lags - tau))
    want = A_*A_*th*(1 - abs(tau)/th)
    check(f"pulse ACF at tau={tau}", R[i], want, 5e-3)

# ---------------------------------------------------------------
# 5. moments: Psi^2 = R(0), sigma^2 = Psi^2 - mu^2 on a test signal
# ---------------------------------------------------------------
sig = 1.0 + 0.8*np.sin(2*np.pi*3*t) + 0.3*rng.standard_normal(len(t))
mu = np.mean(sig); Psi2 = np.mean(sig**2); var = np.var(sig)
check("sigma^2 = Psi^2 - mu^2", var, Psi2 - mu**2, 1e-10)
# sine alone: Psi^2 = amplitude^2/2
s0 = 0.8*np.sin(2*np.pi*3*t)
check("sine mean square = A^2/2", np.mean(s0**2), 0.32, 1e-4)

# ---------------------------------------------------------------
# 6. Cross-correlation finds a hidden delay (slide: "lag = 4.02")
# ---------------------------------------------------------------
fs2 = 50.0
tb = np.arange(0, 20, 1/fs2)
template = np.exp(-0.5*((tb - 3.0)/0.5)**2)           # bump at t=3
DELAY = 4.0
noisyA = template + 0.40*rng.standard_normal(len(tb)) + 1.0
noisyB = np.exp(-0.5*((tb - 3.0 - DELAY)/0.5)**2) + 0.40*rng.standard_normal(len(tb))
a = noisyA - noisyA.mean(); b = noisyB - noisyB.mean()
cc = np.correlate(b, a, "full")
lag = (np.argmax(cc) - (len(a)-1))/fs2
print(f"   recovered delay {lag:.2f} s (true 4.00, SNR ~ 2.5)")
check("cross-correlation recovers the delay", lag, DELAY, 0.15)

# ---------------------------------------------------------------
# 7. Leakage and windowing: 4 cycles vs 4.5 cycles in the record
# ---------------------------------------------------------------
M = 256
k = np.arange(M)
def spec(cycles, window=None):
    s = np.sin(2*np.pi*cycles*k/M)
    if window is not None: s = s*window
    S = np.abs(np.fft.rfft(s))/ (M/2)
    return S
S_int  = spec(4.0)
S_frac = spec(4.5)
hann = 0.5 - 0.5*np.cos(2*np.pi*k/M)      # cos^2(pi t / 2) form
S_hann = spec(4.5, hann)
# integer cycles: single clean line
check("integer cycles: all energy in one bin", S_int[4], 1.0, 1e-9)
check("integer cycles: neighbor bin empty", S_int[5], 0.0, 1e-9)
# non-integer: the biggest bin holds far less, energy leaks everywhere
print(f"   4.5 cycles no window: peak bin {S_frac.max():.3f} (vs 1.0), "
      f"10 bins away still {S_frac[14]:.4f}")
check("leakage: peak bin drops under 0.7", float(S_frac.max() < 0.7), 1.0, 0)
check("leakage: energy 10 bins away > 1%", float(S_frac[14] > 0.01), 1.0, 0)
# Hann: far-away leakage crushed (note amplitude scale changes by the
# window's coherent gain 0.5)
print(f"   4.5 cycles + Hann: 10 bins away {S_hann[14]:.2e} "
      f"({S_frac[14]/S_hann[14]:.0f}x below unwindowed)")
check("Hann cuts far leakage ~100x", float(S_hann[14] < 0.02*S_frac[14]), 1.0, 0)

# ---------------------------------------------------------------
# 8. Wiener filter beats naive inverse deconvolution with noise
# ---------------------------------------------------------------
u = np.zeros(M); u[60] = 1; u[100] = 0.7; u[150] = -0.5     # true sparse signal
r = np.exp(-0.5*((np.arange(-16,17))/4.0)**2); r /= r.sum() # blur
s = np.convolve(u, r, "same")
c = s + 0.02*rng.standard_normal(M)
Rf = np.fft.rfft(np.roll(np.pad(r, (0, M-len(r))), -(len(r)//2)))
Cf = np.fft.rfft(c)
naive = np.fft.irfft(Cf/np.where(np.abs(Rf) < 1e-12, 1e-12, Rf), M)
Sf2 = np.abs(np.fft.rfft(s))**2
Nf2 = (0.02**2)*M
phi = Sf2/(Sf2 + Nf2)
wiener = np.fft.irfft(Cf*phi/np.where(np.abs(Rf) < 1e-3, np.inf, Rf), M)
mse_naive = np.mean((naive - u)**2)
mse_wien = np.mean((wiener - u)**2)
print(f"   deconvolution MSE: naive inverse {mse_naive:.2f}, Wiener {mse_wien:.4f}")
check("Wiener >> naive inverse", float(mse_wien < 0.01*mse_naive), 1.0, 0)

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
