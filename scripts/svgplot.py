"""Tiny SVG plotter that emits compact markup (integer coords, shared styles)."""
import numpy as np

BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK = "#2563eb", "#dc2626", "#16a34a", "#6b7280", "#ea580c", "#7c3aed", "#111827"


class Fig:
    def __init__(self, w, h, title=None):
        self.w, self.h = w, h
        self.parts = []
        if title:
            self.text(w / 2, 20, title, size=14, anchor="middle", weight="bold")

    def text(self, x, y, s, size=12, anchor="start", color=BLACK, weight=None):
        wt = f' font-weight="{weight}"' if weight else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" text-anchor="{anchor}" fill="{color}"{wt}>{s}</text>')

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'font-family="Helvetica,Arial,sans-serif"><rect width="100%" height="100%" fill="#fff"/>' + "".join(self.parts) + "</svg>")

    def save(self, path):
        s = self.svg()
        open(path, "w").write(s)
        return len(s)


class Panel:
    def __init__(self, fig, x, y, w, h, xr, yr, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None, grid=True, logy=False):
        self.f, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.xr, self.yr, self.logy = xr, yr, logy
        P = fig.parts
        if title:
            fig.text(x + w / 2, y - 8, title, size=12.5, anchor="middle", weight="bold")
        if grid and yticks is not None:
            for t in yticks:
                yy = self.Y(t)
                P.append(f'<path d="M{x} {yy:.0f}H{x + w}" stroke="#e5e7eb"/>')
        P.append(f'<path d="M{x} {y}V{y + h}H{x + w}" stroke="{BLACK}" fill="none"/>')
        if xticks is not None:
            for t in xticks:
                xx = self.X(t)
                P.append(f'<path d="M{xx:.0f} {y + h}v4" stroke="{BLACK}"/>')
                fig.text(xx, y + h + 17, fmt(t), size=11, anchor="middle")
        if yticks is not None:
            for t in yticks:
                yy = self.Y(t)
                P.append(f'<path d="M{x} {yy:.0f}h-4" stroke="{BLACK}"/>')
                lab = f"1e{int(t)}" if logy else fmt(t)
                fig.text(x - 7, yy + 4, lab, size=11, anchor="end")
        if xlabel:
            fig.text(x + w / 2, y + h + 34, xlabel, size=12, anchor="middle")
        if ylabel:
            fig.parts.append(f'<text x="{x - 38}" y="{y + h / 2:.0f}" font-size="12" text-anchor="middle" transform="rotate(-90 {x - 38} {y + h / 2:.0f})">{ylabel}</text>')
        self.clip = f"c{len(P)}"
        P.append(f'<clipPath id="{self.clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}"/></clipPath>')

    def X(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def line(self, xs, ys, color=BLUE, width=2.2, dash=None, clip=True):
        pts = []
        last = None
        for a, b in zip(xs, ys):
            p = (round(self.X(a)), round(max(-2000, min(2000, self.Y(b)))))
            if p != last:
                pts.append(p)
                last = p
        pts = rdp(pts, 0.5)
        d = "M" + " ".join(f"{p[0]} {p[1]}" for p in pts[:1]) + "L" + " ".join(f"{p[0]} {p[1]}" for p in pts[1:])
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        cp = f' clip-path="url(#{self.clip})"' if clip else ""
        self.f.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none"{ds}{cp}/>')

    def band(self, xs, lo, hi, color=GRAY, opacity=0.2):
        top = [(round(self.X(a)), round(self.Y(b))) for a, b in zip(xs, hi)]
        bot = [(round(self.X(a)), round(self.Y(b))) for a, b in zip(xs, lo)][::-1]
        d = "M" + "L".join(f"{p[0]} {p[1]}" for p in top + bot) + "Z"
        self.f.parts.append(f'<path d="{d}" fill="{color}" fill-opacity="{opacity}" clip-path="url(#{self.clip})"/>')

    def dots(self, xs, ys, color=RED, r=4.5, hollow=False):
        fill = "#fff" if hollow else color
        for a, b in zip(xs, ys):
            self.f.parts.append(f'<circle cx="{self.X(a):.0f}" cy="{self.Y(b):.0f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="1.6"/>')

    def vseg(self, x0, y0, y1, color=GRAY, width=1.2, dash="3 3"):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.f.parts.append(f'<path d="M{self.X(x0):.0f} {self.Y(y0):.0f}V{self.Y(y1):.0f}" stroke="{color}" stroke-width="{width}"{ds}/>')

    def hline(self, v, color=GRAY, width=1, dash="4 3"):
        self.f.parts.append(f'<path d="M{self.x} {self.Y(v):.0f}H{self.x + self.w}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def bar(self, x0, v, width, color, base=None):
        b = self.yr[0] if base is None else base
        x = self.X(x0 - width / 2)
        w = self.X(x0 + width / 2) - x
        y1, y0 = self.Y(v), self.Y(b)
        self.f.parts.append(f'<rect x="{x:.0f}" y="{min(y0, y1):.0f}" width="{max(1, w):.0f}" height="{abs(y0 - y1):.0f}" fill="{color}" clip-path="url(#{self.clip})"/>')

    def label(self, xv, yv, s, color=BLACK, size=11, anchor="start", dx=0, dy=0, weight=None):
        self.f.text(self.X(xv) + dx, self.Y(yv) + dy, s, size=size, anchor=anchor, color=color, weight=weight)

    def legend(self, items, x=None, y=None, size=11):
        x = self.x + 10 if x is None else x
        y = self.y + 14 if y is None else y
        for i, (lab, col, dash) in enumerate(items):
            yy = y + i * 17
            ds = f' stroke-dasharray="{dash}"' if dash else ""
            self.f.parts.append(f'<path d="M{x} {yy - 4}h20" stroke="{col}" stroke-width="2.5"{ds}/>')
            self.f.text(x + 26, yy, lab, size=size)


def fmt(t):
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t))).replace("-", "−")
    return f"{t:g}".replace("-", "−")


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5 or 1
    dmax, idx = 0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * (pts[i][0] - x0) - dx * (pts[i][1] - y0)) / n
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(pts[: idx + 1], eps)[:-1] + rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]
