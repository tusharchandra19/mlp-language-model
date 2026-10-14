"""Reproduce the main notebook's training config (seeded) and report clean metrics.
usage: python rerun.py ../data/names.txt ordered.pt 0   # split as in the notebook
       python rerun.py ../data/names.txt shuffled.pt 1  # shuffled split
(~25-40 min each on CPU)
"""
import sys, random, json, torch, torch.nn.functional as F
torch.set_num_threads(1)
path, out, shuffle = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
torch.manual_seed(42)
words = open(path).read().splitlines()
if shuffle:
    random.seed(42); random.shuffle(words)
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}; stoi['.'] = 0

def build(ws, bs=5):
    X, Y = [], []
    for w in ws:
        ctx = [0] * bs
        for ch in w + '.':
            ix = stoi[ch]; X.append(ctx); Y.append(ix); ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build(words[:n1]); Xva, Yva = build(words[n1:n2]); Xte, Yte = build(words[n2:])

C = torch.randn((27, 10), requires_grad=True)
W1 = torch.randn((50, 500), requires_grad=True); b1 = torch.randn(500, requires_grad=True)
W2 = torch.randn((500, 27), requires_grad=True); b2 = torch.randn(27, requires_grad=True)
P = [C, W1, b1, W2, b2]; V = [torch.zeros_like(p) for p in P]
lr, beta = 0.32, 0.85

@torch.no_grad()
def ce(X, Y):
    h = (C[X].view(-1, 50) @ W1 + b1).tanh()
    return F.cross_entropy(h @ W2 + b2, Y).item()

log = []
for i in range(100000):
    ix = torch.randint(0, Xtr.shape[0], (1024,))
    h = (C[Xtr[ix]].view(-1, 50) @ W1 + b1).tanh()
    cel = F.cross_entropy(h @ W2 + b2, Ytr[ix])
    r1 = 0.001 * (W1 ** 2).sum(); r2 = 0.001 ** (W2 ** 2).sum()
    loss = cel + r1 + r2
    for p in P: p.grad = None
    loss.backward()
    with torch.no_grad():
        for p, v in zip(P, V):
            v.mul_(beta).add_(p.grad); p -= lr * v
    if i % 1000 == 0:
        log.append(dict(i=i, ce=cel.item(), r1=r1.item(), r2=r2.item(), val=ce(Xva, Yva), lr=lr, beta=beta))
    if i % 10000 == 0:
        lr = max(lr / 2, 0.0002); beta = min(beta + 0.01, 0.95)
    if i % 20000 == 0: print(i, log[-1], flush=True)

with torch.no_grad():
    preact = (C[Xtr[:5000]].view(-1, 50) @ W1 + b1)
res = dict(shuffle=shuffle, train=ce(Xtr, Ytr), val=ce(Xva, Yva), test=ce(Xte, Yte),
           final_r1=log[-1]['r1'], final_r2=log[-1]['r2'], W2_sq=(W2 ** 2).sum().item(),
           sat_frac=(preact.tanh().abs() > 0.99).float().mean().item(), first=log[0], log=log)
torch.save(dict(C=C.detach(), W1=W1.detach(), b1=b1.detach(), W2=W2.detach(), b2=b2.detach(), stoi=stoi, res=res), out)
print(json.dumps({k: v for k, v in res.items() if k != 'log'}))
