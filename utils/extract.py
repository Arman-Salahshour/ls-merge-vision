import argparse
from pathlib import Path
import torch
from .resnet20 import resnet20
from . import chunking as C
import warnings
warnings.filterwarnings("ignore")


def run(state_dict, args):
    '''run the full chain, flatten normalize compand pad chunk group'''
    chunks, mask, seq_index, meta = C.extract_model(
        state_dict,
        model_id=args.model_id,
        checkpoint_path=args.ckpt or "",
        split_id=args.split_id,
        seed=args.seed,
        epoch=args.epoch,
        init_group=args.init_group,
        companding_type=args.companding,
    )

    '''print the layer table, this is the hand derived table to check against'''
    print(f"\n{'key':<26} {'shape':<20} {'L':>7} {'chunks':>7} {'seq':>4} "
          f"{'c/filt':>7} {'mu':>10} {'sigma':>8}")
    print("-" * 96)
    for lm in meta.layers:
        print(f"{lm.key:<26} {str(lm.shape):<20} {lm.L:>7} {lm.n_chunks:>7} "
              f"{lm.n_chunks // meta.tokens_per_seq:>4} {lm.chunks_per_filter:>7.2f} "
              f"{lm.norm_mu:>10.3e} {lm.norm_sigma:>8.4f}")

    '''three gates, none of them may be skipped before the vae sees this data'''
    excl = C.verify_coverage(meta, state_dict)
    C.verify_alignment(meta)
    C.verify_roundtrip(chunks, meta, state_dict, tol=args.tol)

    print(f"\n  kept {len(meta.layers)} layers / {meta.kept_params:,} params")
    print(f"  excluded {len(meta.excluded_keys)} keys / {excl:,} params")
    print(f"  total {meta.total_params:,}  "
          f"({100 * meta.kept_params / meta.total_params:.1f}% covered)")
    print(f"  chunks {chunks.shape}  sequences {meta.n_sequences}  "
          f"mask true {mask.mean() * 100:.2f}%")

    if args.out:
        C.save(args.out, chunks, mask, seq_index, meta)
        '''reload from disk and verify again, this catches a broken json round trip'''
        c2, m2, s2, meta2 = C.load(args.out)
        C.verify_roundtrip(c2, meta2, state_dict, tol=args.tol, verbose=False)
        print(f"  saved to {args.out}/ and reloaded ok")

    return chunks, mask, seq_index, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    '''with no checkpoint the script self tests on a randomly initialized model'''
    p.add_argument("ckpt", nargs="?", default=None)
    p.add_argument("out", nargs="?", default=None)
    '''zoo bookkeeping, init_group is the one you cannot recover later'''
    p.add_argument("--model-id", default="selftest")
    p.add_argument("--split-id", default="unknown")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--epoch", type=int, default=-1)
    p.add_argument("--init-group", default="unknown")
    p.add_argument("--companding", default="log1p", choices=["log1p", "none"])
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    model = resnet20(num_classes=100)
    if args.ckpt:
        '''strict load so a mismatched architecture fails loudly here'''
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    run(model.state_dict(), args)
