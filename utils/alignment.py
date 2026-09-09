
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from . import chunking as C
from . import permutation as P
from .constants import CHUNK_SIZE, TOKENS_PER_SEQ
from .permutation import STAGE_PLANES, BLOCKS_PER_STAGE, ResNetPerm, build_perm, outer_positions
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- the paper's OT

def _sym_sqrt(S, eps=1e-10):
    '''symmetric psd square root via eigendecomposition, eigenvalues clamped at zero'''
    vals, vecs = torch.linalg.eigh(S)
    vals = vals.clamp_min(eps)
    return (vecs * vals.sqrt()) @ vecs.T


def _sym_inv_sqrt(S, eps=1e-10):
    vals, vecs = torch.linalg.eigh(S)
    vals = vals.clamp_min(eps)
    return (vecs * vals.rsqrt()) @ vecs.T


def gaussian_stats(Z, shrinkage=1e-3):
    """Z is [n, d]: n latent vectors of dimension d for one layer.

    shrinkage is not optional at this scale. a layer1 conv is 16 chunks in a 144
    dimensional latent, so the empirical covariance has rank at most 15 out of 144 and
    Sigma^-1/2 is unbounded. we regularise with lambda * mean(eigenvalue) * I, which is
    also worth stating as a caveat: the paper's Gaussian OT is estimated from far fewer
    samples than dimensions for every layer below layer3 in a network this small"""
    Z = torch.as_tensor(Z, dtype=torch.float64)
    mu = Z.mean(0)
    Zc = Z - mu
    S = (Zc.T @ Zc) / max(len(Z) - 1, 1)
    if shrinkage:
        S = S + shrinkage * (torch.trace(S) / S.shape[0]) * torch.eye(S.shape[0], dtype=S.dtype)
    return mu, S


def rank_report(Z, tol=1e-8):
    """how badly under-determined the Gaussian fit is for this layer"""
    Z = torch.as_tensor(Z, dtype=torch.float64)
    n, d = Z.shape
    s = torch.linalg.svdvals(Z - Z.mean(0))
    rank = int((s > tol * s.max()).sum())
    return {'n_samples': n, 'd': d, 'rank': rank, 'rank_deficient': rank < d}


def bures_map(Zs, Zt, eps=1e-10, shrinkage=1e-3):
    """the closed-form Monge map between two Gaussians, equation 2 of the paper"""
    mu_s, Ss = gaussian_stats(Zs, shrinkage)
    mu_t, St = gaussian_stats(Zt, shrinkage)
    Ss_h = _sym_sqrt(Ss, eps)
    Ss_ih = _sym_inv_sqrt(Ss, eps)
    middle = _sym_sqrt(Ss_h @ St @ Ss_h, eps)
    A = Ss_ih @ middle @ Ss_ih
    return A, mu_s, mu_t


def apply_bures(Z, A, mu_s, mu_t):
    Z = torch.as_tensor(Z, dtype=torch.float64)
    return (Z - mu_s) @ A.T + mu_t


def bures_distance(Zs, Zt, eps=1e-10, shrinkage=1e-3):
    '''2-Wasserstein distance between the fitted Gaussians, mean part plus Bures part'''
    mu_s, Ss = gaussian_stats(Zs, shrinkage)
    mu_t, St = gaussian_stats(Zt, shrinkage)
    Ss_h = _sym_sqrt(Ss, eps)
    cross = _sym_sqrt(Ss_h @ St @ Ss_h, eps)
    bures = torch.trace(Ss) + torch.trace(St) - 2 * torch.trace(cross)
    return float(((mu_s - mu_t) ** 2).sum() + bures.clamp_min(0.0))


def bures_report(Zs, Zt, eps=1e-10, shrinkage=1e-3):
    """how much work the paper's OT actually does between these two latent clouds.

    dev_from_identity is ||A - I||_F / sqrt(d), so 0 means the map is a no-op and the
    alignment cannot be what makes or breaks a merge between these two models.
    mean_shift_rel puts the translation part on the same relative footing"""
    A, mu_s, mu_t = bures_map(Zs, Zt, eps, shrinkage)
    d = A.shape[0]
    I = torch.eye(d, dtype=A.dtype)
    scale = float(torch.linalg.norm(mu_s)) + float(torch.linalg.norm(mu_t))
    return {
        'd': d,
        'dev_from_identity': float(torch.linalg.norm(A - I) / np.sqrt(d)),
        'mean_shift': float(torch.linalg.norm(mu_s - mu_t)),
        'mean_shift_rel': float(torch.linalg.norm(mu_s - mu_t)) / max(0.5 * scale, 1e-12),
        'w2': bures_distance(Zs, Zt, eps, shrinkage),
        'trace_ratio': float(torch.trace(A) / d),
        **rank_report(Zs),
        'A': A, 'mu_s': mu_s, 'mu_t': mu_t,
    }


# ---------------------------------------------------------------- unit-level OT

def sinkhorn(C, reg=0.05, iters=500, tol=1e-9):
    """entropic OT plan between two uniform marginals over units, minimising cost C"""
    C = torch.as_tensor(C, dtype=torch.float64)
    n, m = C.shape
    K = torch.exp(-C / (reg * (C.abs().max() + 1e-12)))
    u = torch.full((n,), 1.0 / n, dtype=torch.float64)
    v = torch.full((m,), 1.0 / m, dtype=torch.float64)
    a = torch.full((n,), 1.0 / n, dtype=torch.float64)
    b = torch.full((m,), 1.0 / m, dtype=torch.float64)
    for _ in range(iters):
        u_prev = u
        u = a / (K @ v).clamp_min(1e-300)
        v = b / (K.T @ u).clamp_min(1e-300)
        if float((u - u_prev).abs().max()) < tol:
            break
    return torch.diag(u) @ K @ torch.diag(v)


def solve_assignment(C, solver='hungarian', reg=0.05):
    """C is a COST matrix, returns p with new index j taking old index p[j]"""
    C = np.asarray(torch.as_tensor(C, dtype=torch.float64).cpu())
    if solver == 'hungarian':
        rows, cols = linear_sum_assignment(C)
        p = np.empty(C.shape[0], dtype=np.int64)
        p[rows] = cols
        return p
    if solver == 'sinkhorn':
        plan = sinkhorn(C, reg=reg).numpy()
        '''read the entropic plan out as a hard permutation, a soft plan would not be a
        symmetry of the network and the merged weights would not be well defined'''
        rows, cols = linear_sum_assignment(-plan)
        p = np.empty(C.shape[0], dtype=np.int64)
        p[rows] = cols
        return p
    raise ValueError(f'unknown solver: {solver!r}')


# ---------------------------------------------------------------- unit descriptors

def conv_keys():
    """(key, stage index, block index, role) for every conv the permutations touch"""
    out = [('conv1.weight', 0, None, 'stem')]
    for s in range(3):
        for b in range(BLOCKS_PER_STAGE):
            out.append((f'layer{s+1}.{b}.conv1.weight', s, b, 'inner'))
            out.append((f'layer{s+1}.{b}.conv2.weight', s, b, 'stream'))
    return out


def weight_filter_rows(sd):
    """each output filter flattened, the raw-weight descriptor of a unit"""
    return {k: sd[k].detach().float().reshape(sd[k].shape[0], -1)
            for k, _, _, _ in conv_keys() if k in sd}


def latent_filter_codes(chunk_codes, meta, mode='concat'):
    """per-filter latent codes, the learned descriptor of a unit.

    a filter is exactly 1, 2 or 4 consecutive chunks for every conv in this network -
    that is what verify_alignment guarantees - so a filter's code is the concatenation
    of its chunks' codes. fc.weight is skipped: its filters are not chunk aligned and
    the class axis is not a free permutation anyway"""
    codes = {}
    C = torch.as_tensor(chunk_codes, dtype=torch.float32)
    for lm in meta.layers:
        if lm.layer_type == 'linear':
            continue
        cpf = lm.chunks_per_filter
        if abs(cpf - round(cpf)) > 1e-9:
            continue
        cpf = int(round(cpf))
        rows = C[lm.chunk_start:lm.chunk_start + lm.c_out * cpf]
        z = rows.reshape(lm.c_out, cpf, -1)
        codes[lm.key] = z.reshape(lm.c_out, -1) if mode == 'concat' else z.mean(1)
    return codes


@torch.no_grad()
def encode_state_dict(tvae, sd, device, batch_size=64):
    """chunk a state dict and run it through the encoder, returning the per-chunk means
    in row order plus the manifest. this is the bridge that lets the matching driver use
    learned descriptors instead of raw weights"""
    chunks, _, _, meta = C.extract_model(sd, 'align_tmp')
    T = meta.tokens_per_seq
    x = torch.from_numpy(chunks).float().view(-1, T, meta.chunk_size)
    n_seq = x.shape[0]

    depth = torch.zeros(n_seq, dtype=torch.long)
    stage = torch.zeros(n_seq, dtype=torch.long)
    seq = torch.zeros(n_seq, dtype=torch.long)
    for lm in meta.layers:
        for k in range(lm.n_seq):
            i = lm.seq_start + k
            depth[i], stage[i], seq[i] = lm.depth_index, lm.stage, k

    tvae.eval()
    out = []
    for i in range(0, n_seq, batch_size):
        sl = slice(i, i + batch_size)
        mu, _ = tvae.encode(x[sl].to(device), depth[sl].to(device),
                            stage[sl].to(device), seq[sl].to(device))
        out.append(mu.cpu())
    mu = torch.concat(out, 0).reshape(-1, tvae.latent_dim)
    assert mu.shape[0] == meta.n_chunks_total
    return mu.numpy(), meta


def layer_latents(chunk_codes, meta, include_pad=False):
    """per-layer latent clouds, the samples the paper's Gaussian OT is fitted to"""
    out = {}
    C_ = np.asarray(chunk_codes)
    for lm in meta.layers:
        end = lm.chunk_end if not include_pad else lm.seq_start * meta.tokens_per_seq + lm.n_seq * meta.tokens_per_seq
        out[lm.key] = C_[lm.chunk_start:end]
    return out


def latent_descriptor_fn(tvae, device, mode='concat', batch_size=64):
    """descriptor_fn for match_iterative, backed by the encoder rather than the weights"""
    def fn(sd):
        mu, meta = encode_state_dict(tvae, sd, device, batch_size)
        return latent_filter_codes(mu, meta, mode)
    return fn


def _cost_from_descriptors(Da, Db, metric='cosine'):
    """cost matrix between two sets of unit descriptors, lower is a better match"""
    Da = torch.as_tensor(Da, dtype=torch.float64)
    Db = torch.as_tensor(Db, dtype=torch.float64)
    if metric == 'cosine':
        a = Da / Da.norm(dim=1, keepdim=True).clamp_min(1e-12)
        b = Db / Db.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return 1.0 - a @ b.T
    if metric == 'l2':
        return torch.cdist(Da, Db) ** 2
    if metric == 'inner':
        return -(Da @ Db.T)
    raise ValueError(f'unknown metric: {metric!r}')


# ---------------------------------------------------------------- matching driver

def _accumulate(target, block, rows, cols):
    target[np.ix_(rows, cols)] += block


def match_from_descriptors(desc_a, desc_b, metric='cosine', solver='hungarian', reg=0.05):
    """one-sided matching: every free permutation is solved from the descriptors of the
    output filters it governs.

    the aggregation is what respects the option-A constraint. the stage-1 stream also
    fixes the middle of stage 2 and, through it, the middle of stage 3, so its cost has
    to collect the terms from all three streams before the assignment is solved"""
    stream_cost = [np.zeros((p, p)) for p in STAGE_PLANES]
    inner_cost = [[np.zeros((p, p)) for _ in range(BLOCKS_PER_STAGE)] for p in STAGE_PLANES]

    for key, s, b, role in conv_keys():
        if key not in desc_a or key not in desc_b:
            continue
        C = _cost_from_descriptors(desc_a[key], desc_b[key], metric).numpy()
        if role == 'inner':
            inner_cost[s][b] += C
        else:
            '''the stem writes the stage-1 stream, every conv2 writes its own stream'''
            stream_cost[0 if role == 'stem' else s] += C

    '''stage-1 stream: position i of stream 0, position 8+i of stream 1, 24+i of stream 2'''
    pad2, pad3 = STAGE_PLANES[1] // 4, STAGE_PLANES[2] // 4
    n1 = STAGE_PLANES[0]
    C1 = stream_cost[0].copy()
    C1 += stream_cost[1][pad2:pad2 + n1, :][:, pad2:pad2 + n1]
    off = pad3 + pad2
    C1 += stream_cost[2][off:off + n1, :][:, off:off + n1]
    r1 = solve_assignment(C1, solver, reg)

    '''stage-2 outer positions also drive the corresponding stage-3 middle positions'''
    o2 = outer_positions(1)
    C2 = stream_cost[1][np.ix_(o2, o2)] + stream_cost[2][np.ix_(pad3 + o2, pad3 + o2)]
    outer2 = solve_assignment(C2, solver, reg)

    o3 = outer_positions(2)
    C3 = stream_cost[2][np.ix_(o3, o3)]
    outer3 = solve_assignment(C3, solver, reg)

    inner = [[solve_assignment(inner_cost[s][b], solver, reg)
              for b in range(BLOCKS_PER_STAGE)] for s in range(3)]

    perm = build_perm(r1, outer2, outer3, inner)
    P.check_valid(perm)
    return perm


def match_iterative(sd_a, sd_b, descriptor_fn, iters=8, metric='cosine',
                    solver='hungarian', reg=0.05, verbose=False):
    """one-sided matching, iterated.

    a filter's descriptor is not invariant to the permutation of the layer BELOW it: the
    filter is a [c_in, k, k] block laid out in input-channel order, so permuting the
    previous stream reorders the descriptor's own coordinates. A latent code inherits
    this exactly, because the chunk it is encoded from is that same block. So descriptors
    have to be recomputed against the current alignment and the match re-solved, the same
    coordinate descent weight matching needs. A single shot recovers well under half of a
    planted permutation; iterating recovers it"""
    perm = P.identity_perm()
    desc_a = descriptor_fn(sd_a)
    prev = None
    for it in range(iters):
        desc_b = descriptor_fn(P.apply_perm(sd_b, perm))
        step = match_from_descriptors(desc_a, desc_b, metric, solver, reg)
        perm = P.compose(perm, step)
        P.check_valid(perm)
        obj = objective(sd_a, sd_b, perm)
        if verbose:
            print(f'  descriptor matching iter {it + 1}: objective {obj:.4f}')
        if prev is not None and abs(obj - prev) < 1e-9:
            break
        prev = obj
    return perm


def weight_match(sd_a, sd_b, iters=8, solver='hungarian', reg=0.05, verbose=False):
    """Git Re-Basin style weight matching: coordinate descent maximising the sum of
    <A, permuted B> over every tensor, so a unit is matched on the filter it writes AND
    the column it is read through. the one-sided version above sees only the first half,
    which is why the notebook also reports a rows-only weight control"""
    perm = P.identity_perm()
    prev = None
    for it in range(iters):
        stream_cost = [np.zeros((p, p)) for p in STAGE_PLANES]
        inner_cost = [[np.zeros((p, p)) for _ in range(BLOCKS_PER_STAGE)] for p in STAGE_PLANES]

        for key, axis, target in _tensor_slots():
            if key not in sd_a or key not in sd_b:
                continue
            C = _pair_cost(sd_a[key].detach().float(), sd_b[key].detach().float(),
                           key, axis, perm)
            if C is None:
                continue
            '''solve_assignment minimises, the objective is a similarity, so subtract'''
            if target[0] == 'inner':
                inner_cost[target[1]][target[2]] -= C
            else:
                stream_cost[target[1]] -= C

        pad2, pad3 = STAGE_PLANES[1] // 4, STAGE_PLANES[2] // 4
        n1 = STAGE_PLANES[0]
        C1 = stream_cost[0].copy()
        C1 += stream_cost[1][pad2:pad2 + n1, :][:, pad2:pad2 + n1]
        off = pad3 + pad2
        C1 += stream_cost[2][off:off + n1, :][:, off:off + n1]
        r1 = solve_assignment(C1, solver, reg)

        o2 = outer_positions(1)
        C2 = stream_cost[1][np.ix_(o2, o2)] + stream_cost[2][np.ix_(pad3 + o2, pad3 + o2)]
        outer2 = solve_assignment(C2, solver, reg)
        o3 = outer_positions(2)
        outer3 = solve_assignment(stream_cost[2][np.ix_(o3, o3)], solver, reg)
        inner = [[solve_assignment(inner_cost[s][b], solver, reg)
                  for b in range(BLOCKS_PER_STAGE)] for s in range(3)]

        perm = build_perm(r1, outer2, outer3, inner)
        P.check_valid(perm)

        obj = objective(sd_a, sd_b, perm)
        if verbose:
            print(f'  weight matching iter {it + 1}: objective {obj:.4f}')
        if prev is not None and abs(obj - prev) < 1e-9:
            break
        prev = obj
    return perm


def _tensor_slots():
    """(key, axis, target) for every place a permutation acts. target is ('stream', i)
    or ('inner', stage, block) and is RESOLVED here, because the input axis of the first
    conv of a stage is fed by the PREVIOUS stage's stream, not its own"""
    slots = [('conv1.weight', 0, ('stream', 0)),
             ('bn1.weight', 'vec', ('stream', 0)),
             ('bn1.bias', 'vec', ('stream', 0))]
    for s in range(3):
        for b in range(BLOCKS_PER_STAGE):
            pre = f'layer{s+1}.{b}.'
            in_stream = s - 1 if (b == 0 and s > 0) else s
            slots.append((pre + 'conv1.weight', 0, ('inner', s, b)))
            slots.append((pre + 'conv1.weight', 1, ('stream', in_stream)))
            slots.append((pre + 'conv2.weight', 0, ('stream', s)))
            slots.append((pre + 'conv2.weight', 1, ('inner', s, b)))
            slots.append((pre + 'bn1.weight', 'vec', ('inner', s, b)))
            slots.append((pre + 'bn1.bias', 'vec', ('inner', s, b)))
            slots.append((pre + 'bn2.weight', 'vec', ('stream', s)))
            slots.append((pre + 'bn2.bias', 'vec', ('stream', s)))
    slots.append(('fc.weight', 1, ('stream', 2)))
    return slots


def _pair_cost(A, B, key, axis, perm):
    """similarity block for one tensor slot, holding the other permutations fixed"""
    if axis == 'vec':
        return torch.outer(A.double(), B.double()).numpy()
    if key == 'fc.weight':
        return (A.double().T @ B.double()).numpy()

    s, b = _slot_stage_block(key)
    if axis == 0:
        q = _in_perm_for(key, perm)
        Bp = B[:, torch.as_tensor(q, dtype=torch.long)] if q is not None else B
        return (A.reshape(A.shape[0], -1).double() @
                Bp.reshape(Bp.shape[0], -1).double().T).numpy()
    p = _out_perm_for(key, perm)
    Bp = B[torch.as_tensor(p, dtype=torch.long)] if p is not None else B
    Af = A.permute(1, 0, 2, 3).reshape(A.shape[1], -1).double()
    Bf = Bp.permute(1, 0, 2, 3).reshape(Bp.shape[1], -1).double()
    return (Af @ Bf.T).numpy()


def _slot_stage_block(key):
    if not key.startswith('layer'):
        return 0, None
    parts = key.split('.')
    return int(parts[0][-1]) - 1, int(parts[1])


def _in_perm_for(key, perm):
    if key == 'conv1.weight':
        return None
    s, b = _slot_stage_block(key)
    if key.endswith('conv1.weight'):
        return perm.res[s - 1] if (b == 0 and s > 0) else perm.res[s]
    return perm.inner[s][b]


def _out_perm_for(key, perm):
    if key == 'conv1.weight':
        return perm.res[0]
    s, b = _slot_stage_block(key)
    return perm.inner[s][b] if key.endswith('conv1.weight') else perm.res[s]


def objective(sd_a, sd_b, perm):
    """sum of <A, permuted B> over every float tensor, the quantity weight matching
    maximises. also a cheap sanity number: it must not decrease across iterations"""
    sd_p = P.apply_perm(sd_b, perm)
    tot = 0.0
    for k, v in sd_a.items():
        if not v.dtype.is_floating_point or k not in sd_p:
            continue
        tot += float((v.double() * sd_p[k].double()).sum())
    return tot


def perm_agreement(p, q):
    """fraction of units two permutations place identically, per variable"""
    out = {}
    for s, (a, b) in enumerate(zip(p.res, q.res)):
        out[f'res{s+1}'] = float((a == b).mean())
    for s in range(3):
        for bi in range(BLOCKS_PER_STAGE):
            out[f'inner{s+1}.{bi}'] = float((p.inner[s][bi] == q.inner[s][bi]).mean())
    return out
