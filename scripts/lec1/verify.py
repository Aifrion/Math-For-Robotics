"""Lecture 1 (transformations) verification."""
import numpy as np

def Rz(t): c,s=np.cos(t),np.sin(t); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Ry(t): c,s=np.cos(t),np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rx(t): c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])

print("=== order matters ===")
A = Rz(np.pi/2) @ Rx(np.pi/2)
B = Rx(np.pi/2) @ Rz(np.pi/2)
print("Rz(90)Rx(90) e_y =", (A @ [0,1,0]).round(3))
print("Rx(90)Rz(90) e_y =", (B @ [0,1,0]).round(3))
print("max |A-B| =", np.abs(A-B).max().round(3))

print("\n=== ZYX Euler compose + extraction round trip ===")
al, be, ga = 0.4, -0.7, 1.1   # alpha(z), beta(y), gamma(x)
R = Rz(al) @ Ry(be) @ Rx(ga)
# slide's extraction
be2 = np.arctan2(-R[2,0], np.hypot(R[0,0], R[1,0]))
al2 = np.arctan2(R[1,0]/np.cos(be2), R[0,0]/np.cos(be2))
ga2 = np.arctan2(R[2,1]/np.cos(be2), R[2,2]/np.cos(be2))
print("in :", (al, be, ga))
print("out:", (round(al2,4), round(be2,4), round(ga2,4)))

print("\n=== gimbal lock at beta = 90deg ===")
R1 = Rz(0.3) @ Ry(np.pi/2) @ Rx(0.5)
R2 = Rz(0.8) @ Ry(np.pi/2) @ Rx(1.0)   # alpha+0.5, gamma+0.5
print("R(a=0.3,g=0.5) == R(a=0.8,g=1.0)?  max diff =", np.abs(R1-R2).max())
print("-> at beta=90 only (alpha - gamma) matters: one DOF truly lost")

print("\n=== quaternion <-> matrix ===")
def qmul(a, b):
    a0, a1, a2, a3 = a; b0, b1, b2, b3 = b
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 + a2*b0 + a3*b1 - a1*b3,
        a0*b3 + a3*b0 + a1*b2 - a2*b1])
def q_axis(theta, a):
    a = np.asarray(a, float); a = a/np.linalg.norm(a)
    return np.concatenate([[np.cos(theta/2)], np.sin(theta/2)*a])
def q2R(e):
    e0, e1, e2, e3 = e
    return np.array([
        [1-2*(e2*e2+e3*e3), 2*(e1*e2-e0*e3), 2*(e1*e3+e0*e2)],
        [2*(e1*e2+e0*e3), 1-2*(e1*e1+e3*e3), 2*(e2*e3-e0*e1)],
        [2*(e1*e3-e0*e2), 2*(e2*e3+e0*e1), 1-2*(e1*e1+e2*e2)]])
q = q_axis(np.pi/2, [0,0,1])
print("quat for Rz(90):", q.round(4), "\nq2R == Rz(90)?", np.allclose(q2R(q), Rz(np.pi/2)))
# composition: quaternion product == matrix product (same order)
qa, qb = q_axis(0.7, [0,0,1]), q_axis(1.2, [1,0,0])
print("q(a)q(b) -> R equals Ra Rb?", np.allclose(q2R(qmul(qa, qb)), q2R(qa) @ q2R(qb)))
# R -> q (slide formulas)
R = Rz(0.4) @ Ry(-0.7) @ Rx(1.1)
e0 = 0.5*np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
e1 = (R[2,1]-R[1,2])/(4*e0); e2 = (R[0,2]-R[2,0])/(4*e0); e3 = (R[1,0]-R[0,1])/(4*e0)
print("R -> q -> R round trip ok?", np.allclose(q2R([e0,e1,e2,e3]), R))
print("unit norm?", round(e0**2+e1**2+e2**2+e3**2, 10))

print("\n=== homogeneous transform ===")
T = np.eye(4); T[:3,:3] = Rz(np.pi/2); T[:3,3] = [1, 2, 0]
p = np.array([1, 0, 0, 1])
print("T @ (1,0,0,1) =", (T @ p).round(3), " (rotate then translate: (1,3,0))")
# inverse: [R t]^-1 = [R^T, -R^T t]
Ti = np.eye(4); Ti[:3,:3] = T[:3,:3].T; Ti[:3,3] = -T[:3,:3].T @ T[:3,3]
print("T^-1 T == I?", np.allclose(Ti @ T, np.eye(4)))
