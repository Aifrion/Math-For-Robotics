"""Lecture 2 (linear systems) verification."""
import numpy as np

print("=== slides' Gauss elimination example ===")
A = np.array([[0, 4, -1], [1, 1, 1], [2, -2, 1]], float)
b = np.array([5, 6, 1], float)
x = np.linalg.solve(A, b)
print("x =", x, " (slides' sequence ends at x1,x2,x3 = 1,2,3)")
# reproduce the elimination steps
M = np.hstack([A, b[:, None]])
M[[0, 1]] = M[[1, 0]]                    # swap rows (pivot 0)
M[2] -= 2 * M[0]                          # row3 -= 2 row1
print("after swap + eliminate col1:\n", M)
M[2] += M[1]                              # row3 += row2  -> [0,0,-2 | -6]
print("after eliminate col2:\n", M)
print("back-substitute: x3 =", M[2,3]/M[2,2], "x2 =", (M[1,3]-M[1,2]*3)/M[1,1], "x1 =", M[0,3]-M[0,1]*2-M[0,2]*3)

print("\n=== LDU of PA ===")
import scipy.linalg as sla
P, L, U = sla.lu(A)
D = np.diag(np.diag(U))
U1 = np.linalg.inv(D) @ U
print("P@A == L@D@U1?", np.allclose(P.T @ A, L @ D @ U1))
print("D diag (pivots):", np.diag(D))

print("\n=== arm system from CLAUDE.md: J theta_dot = v at shoulder 90deg, elbow -90deg ===")
q = [np.pi/2, -np.pi/2]
s1, c1 = np.sin(q[0]), np.cos(q[0])
s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
J = np.array([[-s1 - s12, -s12], [c1 + c12, c12]])
print("J =\n", J.round(3), " (columns: shoulder (-1,1), elbow (0,1))")
v = np.array([0.5, 0])
print("solve J x = (0.5, 0):", np.linalg.solve(J, v), " (expect (-0.5, 0.5))")

print("\n=== eigen-coordinates demo numbers: A=[[2,1],[1,2]] ===")
A2 = np.array([[2, 1], [1, 2]], float)
lam, V = np.linalg.eigh(A2)
print("eigenvalues:", lam, " eigvecs (cols):\n", V)
print("check: v1=(1,1)/sqrt2 lambda 3; v2=(1,-1)/sqrt2 lambda 1; trace", A2.trace(), "det", np.linalg.det(A2))
bb = np.array([1.0, 0.5])
c = V.T @ bb
xx = V @ (c / lam)
print("b =", bb, "-> x =", xx, " check A x =", A2 @ xx)

print("\n=== Gram-Schmidt / QR / LSQ ===")
A3 = np.array([[1, 1], [1, 2], [1, 3]], float)
b3 = np.array([1, 2, 2], float)
Q, R = np.linalg.qr(A3)
x_qr = np.linalg.solve(R, Q.T @ b3)
x_ne = np.linalg.solve(A3.T @ A3, A3.T @ b3)
print("QR lsq:", x_qr, " normal eqs:", x_ne, " (line fit through (1,1),(2,2),(3,2))")

print("\n=== near-singular example and SVD ===")
A4 = np.array([[1.01, 1, 1], [1, 1.01, 1], [1, 1, 1.01]])
lam4 = np.linalg.eigvalsh(A4)
print("eigenvalues:", lam4.round(4), " (slides: 3.01, 0.01, 0.01)")
U4, s4, Vt4 = np.linalg.svd(A4)
print("singular values:", s4.round(4), " condition number:", (s4[0]/s4[-1]).round(1))
# sensitivity: small change in b -> big change in x
b4 = np.array([3.01, 3.01, 3.01])
x4 = np.linalg.solve(A4, b4)
b4p = b4 + np.array([0.01, 0, -0.01])
x4p = np.linalg.solve(A4, b4p)
print("x for b:", x4.round(3), "  x for b+0.01-perturbation:", x4p.round(3))
print("|db|/|b| =", (np.linalg.norm(b4p-b4)/np.linalg.norm(b4)).round(5),
      " |dx|/|x| =", (np.linalg.norm(x4p-x4)/np.linalg.norm(x4)).round(3),
      " amplification ~ condition number", (s4[0]/s4[-1]).round(0))
# SVD pseudo-inverse with zeroed sigma on a truly singular matrix
A5 = np.array([[1, 1], [1, 1], [0, 1]], float)
U5, s5, Vt5 = np.linalg.svd(A5, full_matrices=False)
b5 = np.array([2, 2, 0.5], float)
xbar = Vt5.T @ ((U5.T @ b5) / s5)
print("SVD lsq on 3x2:", xbar.round(4), " == lstsq:", np.linalg.lstsq(A5, b5, rcond=None)[0].round(4))
