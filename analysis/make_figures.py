"""Render README figures from the two re-runs.
usage: python make_figures.py <dir with ordered.pt and shuffled.pt> ../assets
"""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, A = sys.argv[1], sys.argv[2]
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8a85", "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
o, s = torch.load(f"{S}/ordered.pt"), torch.load(f"{S}/shuffled.pt")

# --- validation curves: ordered vs shuffled split
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
for d, c, name in [(o, BLUE, "Ordered split (as in notebook)"), (s, ORANGE, "Shuffled split")]:
    L = d["res"]["log"][3:]
    x = [r["i"] for r in L]; y = [r["val"] for r in L]
    ax.plot(x, y, color=c, lw=2, label=name)
    ax.annotate(f"{d['res']['val']:.3f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=10)
ax.set_xlabel("Training step"); ax.set_ylabel("Validation cross-entropy (nats/char)")
ax.set_title("Same model and recipe; only the train/val split differs", color=INK, loc="left")
ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_xlim(right=108000)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{A}/split_comparison.png"); plt.close(fig)

# --- embeddings PCA (ordered run = the notebook's recipe)
C = o["C"]; stoi = o["stoi"]; itos = {i: ch for ch, i in stoi.items()}
Cc = C - C.mean(0)
U, Sv, Vh = torch.linalg.svd(Cc, full_matrices=False)
V = Vh.T
Z = Cc @ V[:, :2]
if Z[stoi['a'], 0] > 0: Z[:, 0] *= -1
if Z[stoi['a'], 1] < 0: Z[:, 1] *= -1
import itertools
for a, b in itertools.combinations(range(27), 2):
    if (Z[a] - Z[b]).norm() < 0.6: print('close:', itos[a], itos[b], Z[a].tolist(), Z[b].tolist())
ev = (Sv[:2] ** 2 / (Cc ** 2).sum()).tolist()
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
groups = {"vowel": (set("aeiou"), BLUE), "consonant": (set("bcdfghjklmnpqrstvwxz"), GRAY),
          "y": (set("y"), BLUE), ".": (set("."), ORANGE)}
seen = set()
OFF = {"d": (-11, 7), "q": (-11, -5), "v": (11, 3), "t": (11, -3), "m": (-11, -2), "k": (11, 0), "h": (0, -12), "l": (-10, 0), ".": (10, 0), "i": (-10, 0), "e": (10, 0)}
for i in range(27):
    ch = itos[i]
    lab, col = ("vowel (a e i o u) + y", BLUE) if ch in "aeiouy" else (("start/end ." , ORANGE) if ch == "." else ("consonant", GRAY))
    ax.scatter(Z[i, 0], Z[i, 1], s=110, color=col, edgecolor="white", lw=1.5, zorder=5 if ch == "." else 3,
               label=None if lab in seen else lab)
    seen.add(lab)
    off = OFF.get(ch, (0, 11))
    ax.annotate(ch, (Z[i, 0], Z[i, 1]), xytext=off, textcoords="offset points", ha="center", va="center",
                color=INK, fontsize=11 if ch != "." else 16, fontweight="bold", zorder=6)
ax.set_xlabel(f"PC 1 ({ev[0]*100:.0f}% of variance)"); ax.set_ylabel(f"PC 2 ({ev[1]*100:.0f}% of variance)")
ax.set_title("Learned character embeddings C (10-d → 2-d PCA)", color=INK, loc="left")
ax.grid(color="#eeede8", lw=0.8); ax.legend(frameon=False, loc="best", markerscale=0.6)
fig.tight_layout(); fig.savefig(f"{A}/embeddings.png"); plt.close(fig)

# --- numbers for the README
for name, d in [("ordered", o), ("shuffled", s)]:
    r = d["res"]
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("log", "first")})
print("first", o["res"]["first"])
# nearest neighbours in embedding space (cosine)
Cn = C / C.norm(dim=1, keepdim=True)
sim = Cn @ Cn.T
for ch in "aeiouy.qxz":
    i = stoi[ch]; nn = sim[i].argsort(descending=True)[1:4].tolist()
    print(ch, "->", [itos[j] for j in nn])
