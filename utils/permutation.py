
import numpy as np
import torch
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings("ignore")


STAGE_PLANES = (16, 32, 64)
BLOCKS_PER_STAGE = 3


@dataclass
class ResNetPerm:
    """res[s] is the stage-s residual stream permutation, inner[s][b] the block's
    conv1-to-conv2 channel permutation"""
    res: List[np.ndarray]
    inner: List[List[np.ndarray]]

    def sizes(self):
        return [len(r) for r in self.res]


def _residual_from_prev(prev, planes_prev, planes, outer):
    """build a stage's stream permutation from the previous stage's and a free
    permutation of the outer (zero-padded) positions"""
    pad = planes // 4
    assert pad * 2 + planes_prev == planes, \
        f'option-A layout broken: 2*{pad} + {planes_prev} != {planes}'
    R = np.empty(planes, dtype=np.int64)
    '''the middle block carries the shortcut, so it is pinned to the previous stage'''
    R[pad:pad + planes_prev] = pad + prev
    outer_pos = np.concatenate([np.arange(pad), np.arange(pad + planes_prev, planes)])
    R[outer_pos] = outer_pos[outer]
    return R


def identity_perm():
    return ResNetPerm(
        res=[np.arange(p, dtype=np.int64) for p in STAGE_PLANES],
        inner=[[np.arange(p, dtype=np.int64) for _ in range(BLOCKS_PER_STAGE)]
               for p in STAGE_PLANES],
    )


def build_perm(r1, outer2, outer3, inner):
    """assemble a valid permutation from the free variables"""
    res1 = np.asarray(r1, dtype=np.int64)
    res2 = _residual_from_prev(res1, STAGE_PLANES[0], STAGE_PLANES[1], np.asarray(outer2))
    res3 = _residual_from_prev(res2, STAGE_PLANES[1], STAGE_PLANES[2], np.asarray(outer3))
    return ResNetPerm(res=[res1, res2, res3],
                      inner=[[np.asarray(q, dtype=np.int64) for q in row] for row in inner])


def random_perm(seed=0):
    rng = np.random.default_rng(seed)
    r1 = rng.permutation(STAGE_PLANES[0])
    '''the outer sets have 2*pad entries: 16 for stage 2, 32 for stage 3'''
    outer2 = rng.permutation(2 * (STAGE_PLANES[1] // 4))
    outer3 = rng.permutation(2 * (STAGE_PLANES[2] // 4))
    inner = [[rng.permutation(p) for _ in range(BLOCKS_PER_STAGE)] for p in STAGE_PLANES]
    return build_perm(r1, outer2, outer3, inner)


def outer_positions(stage_idx):
    """the zero-padded positions of a stage's stream, the only part free to move"""
    planes, planes_prev = STAGE_PLANES[stage_idx], STAGE_PLANES[stage_idx - 1]
    pad = planes // 4
    return np.concatenate([np.arange(pad), np.arange(pad + planes_prev, planes)])


def permutation_plan(perm):
    """(state_dict key, axis, permutation) triples describing the whole symmetry.
    axis 0 is the output-filter axis, axis 1 the input-channel axis, and 'vec' a
    per-channel buffer such as a batchnorm parameter"""
    R = perm.res
    plan = []

    '''stem: its output IS the stage-1 residual stream'''
    plan.append(('conv1.weight', 0, R[0]))
    for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
        plan.append((f'bn1.{suffix}', 'vec', R[0]))

    for s in range(3):
        stage = s + 1
        for b in range(BLOCKS_PER_STAGE):
            Q = perm.inner[s][b]
            pre = f'layer{stage}.{b}.'
            '''conv1 reads the incoming stream and writes the block's inner channels'''
            in_stream = R[s - 1] if (b == 0 and s > 0) else R[s]
            plan.append((pre + 'conv1.weight', 0, Q))
            plan.append((pre + 'conv1.weight', 1, in_stream))
            for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                plan.append((pre + f'bn1.{suffix}', 'vec', Q))
            '''conv2 reads the inner channels and writes back into the stream'''
            plan.append((pre + 'conv2.weight', 0, R[s]))
            plan.append((pre + 'conv2.weight', 1, Q))
            for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                plan.append((pre + f'bn2.{suffix}', 'vec', R[s]))

    '''global average pooling keeps the channel axis, so fc reads the last stream'''
    plan.append(('fc.weight', 1, R[2]))
    return plan


def apply_perm(sd, perm):
    """return a permuted copy of the state dict. functionally identical to the input"""
    out = {k: v.clone() for k, v in sd.items()}
    for key, axis, p in permutation_plan(perm):
        if key not in out:
            continue
        t = torch.as_tensor(p, dtype=torch.long)
        if axis == 0:
            out[key] = out[key][t]
        elif axis == 1:
            out[key] = out[key][:, t]
        else:
            out[key] = out[key][t]
    return out


def invert(perm):
    inv = lambda p: np.argsort(p)
    r1 = inv(perm.res[0])
    '''inverting the derived streams directly keeps the option-A constraint intact,
    because the inverse of a constrained permutation satisfies the same constraint'''
    return ResNetPerm(res=[inv(r) for r in perm.res],
                      inner=[[inv(q) for q in row] for row in perm.inner])


def compose(p_outer, p_inner):
    """apply p_inner first, then p_outer"""
    c = lambda a, b: a[b]
    return ResNetPerm(res=[c(a, b) for a, b in zip(p_outer.res, p_inner.res)],
                      inner=[[c(a, b) for a, b in zip(ra, rb)]
                             for ra, rb in zip(p_outer.inner, p_inner.inner)])


def check_valid(perm):
    """a permutation is only a symmetry of THIS network if the option-A constraint holds.
    matching code that solves the streams independently silently violates it"""
    for s, r in enumerate(perm.res):
        assert sorted(r.tolist()) == list(range(len(r))), f'res[{s}] is not a permutation'
    for s, row in enumerate(perm.inner):
        for b, q in enumerate(row):
            assert sorted(q.tolist()) == list(range(len(q))), f'inner[{s}][{b}] is not a permutation'
    for s in (1, 2):
        planes, planes_prev = STAGE_PLANES[s], STAGE_PLANES[s - 1]
        pad = planes // 4
        mid = perm.res[s][pad:pad + planes_prev]
        assert np.array_equal(mid, pad + perm.res[s - 1]), (
            f'stage {s + 1} stream violates the option-A shortcut constraint: the middle '
            f'block must equal the previous stream shifted by {pad}')
        outer_pos = np.concatenate([np.arange(pad), np.arange(pad + planes_prev, planes)])
        assert set(perm.res[s][outer_pos].tolist()) == set(outer_pos.tolist()), \
            f'stage {s + 1} stream moves a zero-padded position into the shortcut block'
    return True


@torch.no_grad()
def verify_symmetry(model_fn, sd, loader, device, n_batches=4, seeds=(0, 1), tol=1e-3):
    """apply random permutations and confirm the network computes the same function.

    this is the gate for every alignment method built on top. a permutation that is
    almost right produces a model that loads without complaint and predicts noise,
    and no downstream number would reveal which of the two it was"""
    base = model_fn().to(device)
    base.load_state_dict(sd, strict=True)
    base.eval()

    worst = 0.0
    for seed in seeds:
        perm = random_perm(seed)
        check_valid(perm)
        other = model_fn().to(device)
        other.load_state_dict(apply_perm(sd, perm), strict=True)
        other.eval()

        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            a, b = base(images), other(images)
            worst = max(worst, float((a - b).abs().max()))

    assert worst < tol, (
        f'permuted network is not functionally identical: max |logit diff| {worst:.3e}. '
        f'the permutation plan or the option-A constraint is wrong')
    return worst
