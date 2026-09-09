import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .resnet20 import resnet20
from . import chunking as C
import warnings
warnings.filterwarnings("ignore")


def moments(x):
    '''first four moments, kurtosis is the fisher excess so a gaussian gives 0'''
    x = x.double().flatten()
    mu = x.mean()
    sd = x.std(unbiased=False)
    z = (x - mu) / sd
    return {
        "mean": float(mu),
        "var": float(sd ** 2),
        "skew": float((z ** 3).mean()),
        "kurt": float((z ** 4).mean() - 3.0),
    }


def tail_mass(x):
    '''fraction of weights beyond 3 and 5 sigma'''
    '''a gaussian gives 0.270% and 0.0000573%'''
    x = x.double().flatten()
    z = (x - x.mean()).abs() / x.std(unbiased=False)
    return {
        "p3sig": float((z > 3).double().mean() * 100),
        "p5sig": float((z > 5).double().mean() * 100),
        "maxz": float(z.max()),
    }


def per_filter_kurt(W):
    '''kurtosis measured inside each output filter, then averaged'''
    '''this removes the variance heterogeneity confound, see the note below'''
    '''a layer of gaussian filters with unequal norms has positive POOLED kurtosis'''
    '''but zero per filter kurtosis, only the second is evidence of real tails'''
    if W.dim() == 4:
        F = W.reshape(W.shape[0], -1).double()
    elif W.dim() == 2:
        F = W.double()
    else:
        return None
    mu = F.mean(1, keepdim=True)
    sd = F.std(1, unbiased=False, keepdim=True).clamp_min(1e-12)
    z = (F - mu) / sd
    k = (z ** 4).mean(1) - 3.0
    '''spread of filter norms, this is what inflates the pooled number'''
    norms = F.norm(dim=1)
    return {
        "kurt_per_filter": float(k.mean()),
        "kurt_per_filter_max": float(k.max()),
        "filter_norm_ratio": float(norms.max() / norms.min().clamp_min(1e-12)),
    }


def pca_spectrum(W, k=10):
    '''explained variance ratio of the leading components, the fig 2 analogue'''
    if W.dim() == 4:
        M = W.reshape(W.shape[0], -1).double()
    elif W.dim() == 2:
        M = W.double()
    else:
        return None
    M = M - M.mean(0, keepdim=True)
    '''singular values squared are proportional to the eigenvalues'''
    s = torch.linalg.svdvals(M)
    ev = (s ** 2) / (s ** 2).sum()
    return [float(v) for v in ev[:k]]


def analyse(state_dict, top_k=10):
    rows = []
    for key, W in state_dict.items():
        if not C.is_kept(key, W):
            continue
        W = W.detach().float().cpu()
        r = {"key": key, "shape": tuple(W.shape), "n": W.numel()}
        r.update(moments(W))
        r.update(tail_mass(W))
        pf = per_filter_kurt(W)
        if pf:
            r.update(pf)
        r["pca"] = pca_spectrum(W, top_k)
        r["stage"] = C.stage_of(key)
        rows.append(r)
    return rows


def report(rows):
    print(f"\n{'key':<24} {'var':>9} {'skew':>8} {'kurt':>8} {'kurt/filt':>10} "
          f"{'norm_r':>7} {'>3sig%':>7} {'>5sig%':>8} {'maxz':>6}")
    print("-" * 100)
    for r in rows:
        print(f"{r['key']:<24} {r['var']:>9.2e} {r['skew']:>8.3f} {r['kurt']:>8.3f} "
              f"{r.get('kurt_per_filter', float('nan')):>10.3f} "
              f"{r.get('filter_norm_ratio', float('nan')):>7.2f} "
              f"{r['p3sig']:>7.3f} {r['p5sig']:>8.4f} {r['maxz']:>6.2f}")

    print("\n" + "-" * 100)
    print(f"{'gaussian reference':<24} {'':>9} {0.0:>8.3f} {0.0:>8.3f} {0.0:>10.3f} "
          f"{'':>7} {0.270:>7.3f} {0.0001:>8.4f}")

    for st in sorted({r["stage"] for r in rows}):
        sub = [r for r in rows if r["stage"] == st]
        k = np.mean([r["kurt"] for r in sub])
        kf = np.mean([r.get("kurt_per_filter", np.nan) for r in sub])
        print(f"  stage {st}: n={len(sub):2d}  mean kurt {k:+.3f}  "
              f"mean kurt/filter {kf:+.3f}")

    '''the headline comparison, pooled minus per filter is the confound'''
    kp = np.mean([r["kurt"] for r in rows])
    kf = np.mean([r.get("kurt_per_filter", np.nan) for r in rows])
    print(f"\n  pooled excess kurtosis     {kp:+.3f}")
    print(f"  per filter excess kurtosis {kf:+.3f}")
    print(f"  inflation from filter norm spread {kp - kf:+.3f}")
    print(f"\n  paper reports gemma self attn up to ~15.2, mlp ~1.1-3.3")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = resnet20(num_classes=100)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        '''checkpoints are saved as a dict with metadata around the weights'''
        sd = ck["state_dict"] if "state_dict" in ck else ck
        model.load_state_dict(sd, strict=True)
    else:
        print("no checkpoint given, running on a random init (means nothing)")

    rows = analyse(model.state_dict())
    report(rows)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n  wrote {args.out}")
