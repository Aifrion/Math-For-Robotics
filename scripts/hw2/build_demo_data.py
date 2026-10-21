"""Inject the waypoint data into the trajectory demo template.

Reads docs/demos/hw2/trajectory_reconstruction.html, replaces the line
containing /*DATA*/ with compact centimetre-integer arrays.
"""
import numpy as np, re

W = np.loadtxt("hw2/waypoints.csv", delimiter=",")
xi = np.round(W[:, 0] * 100).astype(int)
yi = np.round(W[:, 1] * 100).astype(int)
data = ("const WX=[" + ",".join(map(str, xi)) + "],WY=[" +
        ",".join(map(str, yi)) + "];/*DATA*/")
p = "docs/demos/hw2/trajectory_reconstruction.html"
s = open(p).read()
s = re.sub(r"const WX=.*?/\*DATA\*/|/\*DATA\*/", data, s, count=1, flags=re.S)
open(p, "w").write(s)
print("injected", len(xi), "points,", len(s), "bytes total")
