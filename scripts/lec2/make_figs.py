"""Lecture 2 figures (specs from CLAUDE.md section 6)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec2/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"

def arrow(fig, x1, y1, x2, y2, col, w=2):
    d = np.hypot(x2-x1, y2-y1)
    if d < 4: return
    ux, uy = (x2-x1)/d, (y2-y1)/d; h = 9
    fig.parts.append(f'<path d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}M{x2-h*ux+h*0.5*uy:.1f} {y2-h*uy-h*0.5*ux:.1f}L{x2:.1f} {y2:.1f}L{x2-h*ux-h*0.5*uy:.1f} {y2-h*uy+h*0.5*ux:.1f}" stroke="{col}" stroke-width="{w}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')

def draw_arm(fig, bx, by, s, qs, col=BLACK, w=4):
    """2-link arm, link length s, joint angles qs (absolute world convention: q1, q1+q2)."""
    e = (bx + s*np.cos(qs[0]), by - s*np.sin(qs[0]))
    h = (e[0] + s*np.cos(qs[0]+qs[1]), e[1] - s*np.sin(qs[0]+qs[1]))
    fig.parts.append(f'<rect x="{bx-12}" y="{by}" width="24" height="7" rx="2" fill="none" stroke="{GRAY}"/>')
    fig.parts.append(f'<path d="M{bx} {by}L{e[0]:.1f} {e[1]:.1f}L{h[0]:.1f} {h[1]:.1f}" stroke="{col}" stroke-width="{w}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    fig.parts.append(f'<circle cx="{e[0]:.1f}" cy="{e[1]:.1f}" r="4" fill="none" stroke="{col}" stroke-width="1.5"/>')
    return e, h

# ---------- 01 Jacobian anatomy ----------
fig = Fig(860, 360)
fig.text(430, 26, "Anatomy of the arm system J·θ̇ = v  (shoulder 90°, elbow −90°)", 13.5, "middle", BLACK, weight="bold")
bx, by, s = 170, 230, 110
e, h = draw_arm(fig, bx, by, s, [np.pi/2, -np.pi/2])
# hand at (110, up 110): e=(170,120), h=(280,120)
arrow(fig, h[0], h[1], h[0] - 70, h[1] - 70, BLUE, 2.5)
fig.text(h[0]-78, h[1]-78, "column 1 = (−1, 1)", 11.5, "end", BLUE, weight="bold")
fig.text(h[0]-78, h[1]-62, "what the shoulder does", 10.5, "end", GRAY)
arrow(fig, h[0], h[1], h[0], h[1] - 85, RED, 2.5)
fig.text(h[0]+8, h[1]-80, "column 2 = (0, 1)", 11.5, "start", RED, weight="bold")
fig.text(h[0]+8, h[1]-64, "what the elbow does", 10.5, "start", GRAY)
arrow(fig, h[0], h[1], h[0] + 75, h[1], AMBER, 2.5)
fig.text(h[0]+54, h[1]+20, "goal v = (0.5, 0)", 11.5, "middle", AMBER, weight="bold")
fig.text(bx+55, by-40, "shoulder", 10.5, "middle", GRAY)
fig.text(225, 108, "elbow", 10.5, "middle", GRAY)
fig.text(h[0]+10, h[1]+40, "hand", 10.5, "middle", GRAY)
# matrix equation
X0 = 470
fig.text(X0+160, 90, "joints → columns   ·   hand rules → rows", 12, "middle", GRAY)
rows = [("hand x rule →", "−1", "0", "θ̇₁", "0.5"), ("hand y rule →", "1", "1", "θ̇₂", "0")]
for r, (lab, a1, a2, xv, bv) in enumerate(rows):
    y = 130 + r*34
    fig.text(X0+40, y, lab, 11.5, "end", GRAY)
    fig.text(X0+80, y, a1, 15, "middle", BLUE, weight="bold")
    fig.text(X0+125, y, a2, 15, "middle", RED, weight="bold")
    fig.text(X0+185, y, xv, 15, "middle", BLACK)
    fig.text(X0+248, y, "=", 15, "middle", BLACK)
    fig.text(X0+290, y, bv, 15, "middle", AMBER, weight="bold")
for xx in (X0+58, X0+148):
    fig.parts.append(f'<path d="M{xx} 108V172" stroke="{BLACK}" stroke-width="1.3" fill="none"/>')
for xx in (X0+168, X0+205, X0+272, X0+310):
    fig.parts.append(f'<path d="M{xx} 108V172" stroke="{BLACK}" stroke-width="1.3" fill="none"/>')
fig.text(X0+80, 200, "shoulder", 10.5, "middle", BLUE)
fig.text(X0+128, 200, "elbow", 10.5, "middle", RED)
fig.text(X0+160, 250, "solution x = (−0.5, 0.5):  swing the shoulder back, bend the elbow forward,", 11.5, "middle", BLACK)
fig.text(X0+160, 268, "their vertical parts cancel and the hand slides right at 0.5. Verified.", 11.5, "middle", BLACK)
fig.text(430, 330, "People expect rows = joints. It's the other way: each COLUMN is a joint's effect on the hand; each ROW is one rule about a hand direction.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_jacobian_anatomy.svg"))

# ---------- 02 m vs n arms ----------
fig = Fig(860, 330)
titles = [("m < n: 3 joints, 2 rules — choices left over", GREEN),
          ("m = n: 2 joints, 2 rules — one answer", BLUE),
          ("m > n: 1 joint, 2 rules — can't satisfy both", RED)]
for i, (title, col) in enumerate(titles):
    ox = 40 + i*280
    fig.text(ox+120, 60, title.split("—")[0], 12, "middle", BLACK, weight="bold")
    fig.text(ox+120, 78, "— " + title.split("—")[1].strip(), 11.5, "middle", col)
tx, ty = 40+120+60, 160
# panel 1: 3-joint arm reaching target in 2 poses
def arm3(fig, bx, by, s, qs, col, w=3.5, dash=None):
    p = np.array([bx, by], float); a = 0
    pts = [p.copy()]
    for q in qs:
        a += q
        p = p + [s*np.cos(a), -s*np.sin(a)]
        pts.append(p.copy())
    dstr = "M" + "L".join(f"{q[0]:.1f} {q[1]:.1f}" for q in pts)
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    fig.parts.append(f'<path d="{dstr}" stroke="{col}" stroke-width="{w}" fill="none" stroke-linecap="round" stroke-linejoin="round"{extra}/>')
    for q in pts[1:-1]:
        fig.parts.append(f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="3.5" fill="none" stroke="{col}" stroke-width="1.3"/>')
    return pts[-1]
bx1, by1 = 100, 265
fig.parts.append(f'<rect x="{bx1-12}" y="{by1}" width="24" height="7" rx="2" fill="none" stroke="{GRAY}"/>')
t1 = arm3(fig, bx1, by1, 62, [1.25, -0.9, -0.8], BLACK)
arm3(fig, bx1, by1, 62, [0.5, 0.6292, -1.6837], GRAY, 3, "5 4")
fig.parts.append(f'<circle cx="{t1[0]:.1f}" cy="{t1[1]:.1f}" r="7" fill="{AMBER}"/>')
fig.text(t1[0]+12, t1[1]-8, "same hand, two poses", 10.5, "start", GRAY)
# panel 2: 2-joint single pose
bx2, by2 = 380, 265
fig.parts.append(f'<rect x="{bx2-12}" y="{by2}" width="24" height="7" rx="2" fill="none" stroke="{GRAY}"/>')
t2 = arm3(fig, bx2, by2, 86, [1.15, -1.1], BLACK)
fig.parts.append(f'<circle cx="{t2[0]:.1f}" cy="{t2[1]:.1f}" r="7" fill="{AMBER}"/>')
fig.text(t2[0]+12, t2[1]-6, "exactly one pose", 10.5, "start", GRAY)
# panel 3: 1-joint arm missing 2-D target
bx3, by3 = 650, 265
fig.parts.append(f'<rect x="{bx3-12}" y="{by3}" width="24" height="7" rx="2" fill="none" stroke="{GRAY}"/>')
t3 = arm3(fig, bx3, by3, 120, [0.9], BLACK)
tgt = (bx3+150, by3-135)
fig.parts.append(f'<circle cx="{tgt[0]}" cy="{tgt[1]}" r="7" fill="{AMBER}"/>')
fig.parts.append(f'<path d="M{t3[0]:.1f} {t3[1]:.1f}L{tgt[0]} {tgt[1]}" stroke="{RED}" stroke-width="1.6" stroke-dasharray="4 3"/>')
fig.text((t3[0]+tgt[0])/2+10, (t3[1]+tgt[1])/2, "least-squares gap", 10.5, "start", RED)
# arc showing reachable circle
fig.parts.append(f'<path d="M{bx3+120} {by3} A120 120 0 0 0 {bx3-20} {by3-118.3}" stroke="{GRAY}" stroke-width="0.8" stroke-dasharray="3 4" fill="none"/>')
fig.text(430, 318, "m counts rules, n counts joints. m < n: a null space of free motion. m = n: unique solve. m > n: no exact solution — fit by least squares (the robot-calibration regime).", 11, "middle", GRAY)
print("02", fig.save(OUT + "02_m_vs_n.svg"))
