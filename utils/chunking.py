import json
import torch
import hashlib
import warnings
import numpy as np
from .constants import *
from pathlib import Path
from dataclasses import dataclass, asdict, field
warnings.filterwarnings("ignore")


def compand(x, kind="log1p"):
    '''applied after z-scoring, where log1p is actually nonlinear'''
    if kind == "none":
        return x
    if kind == "log1p":
        return torch.sign(x) * torch.log1p(torch.abs(x))
    raise ValueError(f"unknown companding: {kind!r}")


def decompand(y, kind="log1p"):
    '''inverse of compand, expm1 amplifies error so keep it in float32 at least'''
    if kind == "none":
        return y
    if kind == "log1p":
        return torch.sign(y) * torch.expm1(torch.abs(y))
    raise ValueError(f"unknown companding: {kind!r}")


@dataclass
class LayerMeta:
    '''fields needed to rebuild the tensor from the chunk array'''
    key: str
    shape: tuple
    L: int
    L_padded: int
    n_chunks: int
    chunk_start: int
    chunk_end: int
    dtype: str

    '''fields needed to invert the normalize and compand steps'''
    norm_mu: float
    norm_sigma: float
    companding_type: str
    n_pad_values: int
    n_pad_tokens: int

    '''fields fed to the encoder/decoder as conditioning embeddings'''
    layer_type: str
    depth_index: int
    stage: int
    c_in: int
    c_out: int
    k: int
    chunks_per_filter: float

    '''per layer moments and errors, filled in later by the stats pass'''
    stats: dict = field(default_factory=dict)


@dataclass
class ModelMeta:
    '''identity of this checkpoint inside the zoo'''
    model_id: str
    architecture: str
    checkpoint_path: str
    split_id: str
    seed: int
    epoch: int
    '''which shared initialization this model descends from'''
    init_group: str

    '''chunking settings this model was extracted with'''
    chunk_size: int
    tokens_per_seq: int
    companding_type: str
    n_chunks_total: int
    n_sequences: int

    '''coverage bookkeeping'''
    total_params: int
    kept_params: int
    excluded_keys: list

    layers: list

    def to_json(self, path):
        d = asdict(self)
        '''layers may already be plain dicts if this came back from disk'''
        d["layers"] = [asdict(l) if not isinstance(l, dict) else l for l in self.layers]
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path):
        d = json.loads(Path(path).read_text())
        '''rebuild the layer entries as dataclasses again'''
        d["layers"] = [LayerMeta(**l) for l in d["layers"]]
        return ModelMeta(**d)


def is_kept(key, tensor):
    '''only weight tensors go through the vae'''
    if not key.endswith(".weight"):
        return False
    '''4d is conv, 2d is linear, anything else is a norm layer or a bias'''
    if tensor.dim() not in (2, 4):
        return False
    if key in EXCLUDE_EXACT:
        return False
    if any(s in key for s in EXCLUDE_SUBSTRINGS):
        return False
    '''linear layers are opt in, keep them out unless the caller overrides this'''
    if tensor.dim() == 2 and not INCLUDE_LINEAR:
        return False
    return True


def stage_of(key):
    '''which resnet stage this layer belongs to, 0 for the stem and the head'''
    for s in (1, 2, 3):
        if key.startswith(f"layer{s}."):
            return s
    return 0


def extract_model(state_dict, model_id, checkpoint_path="",
                  architecture="resnet20_cifar100", split_id="unknown",
                  seed=-1, epoch=-1, init_group="unknown",
                  chunk_size=CHUNK_SIZE, tokens_per_seq=TOKENS_PER_SEQ,
                  companding_type="log1p"):

    '''split the state dict into layers we encode and layers we copy'''
    kept, excluded = [], []
    for k, v in state_dict.items():
        if is_kept(k, v):
            kept.append(k)
        else:
            excluded.append(k)

    '''count floats only, integer buffers like num_batches_tracked are ignored'''
    total_params = sum(v.numel() for v in state_dict.values() if v.dtype.is_floating_point)
    kept_params = sum(state_dict[k].numel() for k in kept)

    all_chunks, all_masks, all_seq = [], [], []
    layers = []
    #running index into the global chunk array
    cursor = 0
    #running index into the global sequence array
    seq_cursor = 0

    for depth, key in enumerate(kept):
        W = state_dict[key].detach().to(torch.float32).cpu()
        shape = tuple(W.shape)

        '''conv is [c_out, c_in, k, k], linear is [c_out, c_in]'''
        if len(shape) == 4:
            c_out, c_in, k_h, k_w = shape
            layer_type = f"conv{k_h}x{k_w}"
        elif len(shape) == 2:
            c_out, c_in = shape
            k_h = k_w = 1
            layer_type = "linear"
        else:
            raise ValueError(f"{key}: unsupported shape {shape}")

        '''flatten row major, this puts each output filter contiguously'''
        w = W.reshape(-1)
        L = w.numel()

        '''normalize per layer in float64 so mu and sigma survive the json round trip'''
        w64 = w.to(torch.float64)
        mu = float(w64.mean())
        sigma = float(w64.std(unbiased=False))
        '''guard against a constant layer'''
        if sigma < 1e-12:
            sigma = 1.0
        wn = ((w64 - mu) / sigma).to(torch.float32)

        '''compand after normalizing, not before'''
        wc = compand(wn, companding_type)

        '''pad the value sequence up to a whole number of chunks'''
        n_chunks = -(-L // chunk_size)
        L_padded = n_chunks * chunk_size
        n_pad_values = L_padded - L

        padded = torch.zeros(L_padded, dtype=torch.float32)
        padded[:L] = wc
        '''mask marks which positions are real, padding must not enter the loss'''
        vmask = torch.zeros(L_padded, dtype=torch.bool)
        vmask[:L] = True

        chunks = padded.view(n_chunks, chunk_size)
        cmask = vmask.view(n_chunks, chunk_size)

        '''pad the chunk count up to a whole number of sequences'''
        '''sequences never cross a layer boundary'''
        n_seq = -(-n_chunks // tokens_per_seq)
        n_pad_tokens = n_seq * tokens_per_seq - n_chunks
        if n_pad_tokens:
            chunks = torch.cat([chunks, torch.zeros(n_pad_tokens, chunk_size)], 0)
            cmask = torch.cat([cmask, torch.zeros(n_pad_tokens, chunk_size, dtype=torch.bool)], 0)

        n_rows = chunks.shape[0]
        seq_ids = seq_cursor + torch.arange(n_rows) // tokens_per_seq

        '''how many chunks make up one output filter, tells the decoder the sub filter phase'''
        filter_size = c_in * k_h * k_w

        layers.append(LayerMeta(
            key=key,
            shape=shape,
            L=L,
            L_padded=L_padded,
            n_chunks=n_chunks,
            chunk_start=cursor,
            #chunk_end stops before the sequence pad rows
            chunk_end=cursor + n_chunks,
            dtype="float32",
            norm_mu=mu,
            norm_sigma=sigma,
            companding_type=companding_type,
            n_pad_values=n_pad_values,
            n_pad_tokens=n_pad_tokens,
            layer_type=layer_type,
            depth_index=depth,
            stage=stage_of(key),
            c_in=c_in,
            c_out=c_out,
            k=k_h,
            chunks_per_filter=filter_size / chunk_size,
            stats={},
        ))

        all_chunks.append(chunks)
        all_masks.append(cmask)
        all_seq.append(seq_ids)
        '''advance past the pad rows too so the next layer starts clean'''
        cursor += n_rows
        seq_cursor += n_seq

    chunks = torch.cat(all_chunks, 0).numpy().astype(np.float32)
    mask = torch.cat(all_masks, 0).numpy()
    seq_index = torch.cat(all_seq, 0).numpy().astype(np.int64)

    meta = ModelMeta(
        model_id=model_id,
        architecture=architecture,
        checkpoint_path=str(checkpoint_path),
        split_id=split_id,
        seed=seed,
        epoch=epoch,
        init_group=init_group,
        chunk_size=chunk_size,
        tokens_per_seq=tokens_per_seq,
        companding_type=companding_type,
        n_chunks_total=int(chunks.shape[0]),
        n_sequences=int(seq_cursor),
        total_params=int(total_params),
        kept_params=int(kept_params),
        excluded_keys=excluded,
        layers=layers,
    )
    return chunks, mask, seq_index, meta


def reconstruct_layer(chunks, lm):
    '''slice this layer's rows out of the global chunk array'''
    rows = torch.from_numpy(chunks[lm.chunk_start:lm.chunk_end]).to(torch.float32)
    '''unchunk then drop the value padding'''
    flat = rows.reshape(-1)[: lm.L]
    '''invert compand, then invert the z-score'''
    wn = decompand(flat, lm.companding_type).to(torch.float64)
    w = wn * lm.norm_sigma + lm.norm_mu
    return w.reshape(lm.shape).to(torch.float32)


def reconstruct_state_dict(chunks, meta, target_state_dict):
    '''start from a full copy of the target so buffers and biases are all present'''
    out = {k: v.clone() for k, v in target_state_dict.items()}
    '''overwrite only the keys that went through the vae'''
    for lm in meta.layers:
        out[lm.key] = reconstruct_layer(chunks, lm)
    return out


def verify_roundtrip(chunks, meta, state_dict, tol=1e-6, verbose=True):
    '''rebuild every layer and compare it against the original tensor'''
    worst, worst_key = 0.0, None
    for lm in meta.layers:
        orig = state_dict[lm.key].detach().to(torch.float32).cpu()
        rec = reconstruct_layer(chunks, lm)
        err = float((orig - rec).abs().max())
        if err > worst:
            worst, worst_key = err, lm.key
        assert rec.shape == orig.shape, f"{lm.key}: shape {rec.shape} != {orig.shape}"
        assert err < tol, f"{lm.key}: max abs err {err:.3e} > {tol:.1e}"
    if verbose:
        print(f"  roundtrip ok, worst |err| = {worst:.3e} ({worst_key})")
    return worst


def verify_coverage(meta, state_dict):
    '''kept plus excluded must account for every float in the state dict'''
    '''this is what catches a silently dropped layer'''
    excl = sum(state_dict[k].numel() for k in meta.excluded_keys
               if state_dict[k].dtype.is_floating_point)
    assert meta.kept_params + excl == meta.total_params, (
        f"coverage: kept {meta.kept_params} + excluded {excl} != {meta.total_params}")
    return excl


def verify_alignment(meta):
    '''every kept conv should need no padding at either level'''
    for lm in meta.layers:
        '''linear layers are expected to pad, skip them'''
        if lm.layer_type == "linear":
            continue
        assert lm.n_pad_values == 0, f"{lm.key}: {lm.n_pad_values} pad values"
        assert lm.n_pad_tokens == 0, f"{lm.key}: {lm.n_pad_tokens} pad tokens"
        '''a chunk must be a whole number of filters, or a whole fraction of one'''
        whole = abs(lm.chunks_per_filter - round(lm.chunks_per_filter)) < 1e-9
        frac = abs(1 / lm.chunks_per_filter - round(1 / lm.chunks_per_filter)) < 1e-9
        assert whole or frac, \
            f"{lm.key}: chunks_per_filter={lm.chunks_per_filter} not filter-aligned"


def save(outdir, chunks, mask, seq_index, meta):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    '''chunks stay contiguous across the zoo so training reads are sequential'''
    np.save(outdir / "chunks.npy", chunks)
    np.save(outdir / "mask.npy", mask)
    np.save(outdir / "seq_index.npy", seq_index)
    meta.to_json(outdir / "model_meta.json")
    '''fingerprint the array so a stale manifest cannot be paired with new chunks'''
    h = hashlib.sha256(chunks.tobytes()).hexdigest()[:16]
    (outdir / "chunks.sha256").write_text(h)
    return outdir


def load(outdir):
    outdir = Path(outdir)
    return (
        np.load(outdir / "chunks.npy"),
        np.load(outdir / "mask.npy"),
        np.load(outdir / "seq_index.npy"),
        ModelMeta.from_json(outdir / "model_meta.json"),
    )
