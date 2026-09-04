import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from constants import device
from dataset import ResZoo
import warnings
warnings.filterwarnings("ignore")

EXPERT_COLORS = {0: '#4C72B0', 1: '#DD8452', 2: '#55A868', 3: '#C44E52', 4: '#8172B3'}
BACKBONE_LABEL = -1


@torch.no_grad()
def collect_latents(model, dataset, batch_size=64):
    '''eval mode so reparameterize returns mu, embeddings must be deterministic'''
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    lat, depths, stages, model_idx = [], [], [], []
    for batch in loader:
        '''encode only, we want mu not a decoded reconstruction'''
        mu, _ = model.encode(
            batch['chunks'].to(device),
            batch['depth'].to(device),
            batch['stage'].to(device))
        lat.append(mu.cpu().numpy())
        depths.append(batch['depth'].numpy())
        stages.append(batch['stage'].numpy())
        model_idx.append(batch['model_idx'].numpy())

    lat = np.concatenate(lat)
    return (lat, np.concatenate(depths), np.concatenate(stages),
            np.concatenate(model_idx))


def reduce_2d(L, seed=0, perplexity=30):
    '''flatten the token axis so one sequence becomes one point'''
    X = L.reshape(len(L), -1)
    '''pca first, t-sne on raw high dimensional data is slow and picks up noise'''
    n_pc = min(50, X.shape[1], len(X) - 1)
    X = PCA(n_components=n_pc, random_state=seed).fit_transform(X)
    '''perplexity must stay below a third of the sample count'''
    perp = min(perplexity, max(5, (len(X) - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init='pca',
                random_state=seed).fit_transform(X)


def build_latents(model, zoo_dir, model_ids, reference_dir=None, reference_ids=None):
    '''backbone lives under a different root and its split_id is the string 'full',
    so it is built separately and merged at the array level'''
    ds = ResZoo(root_dir=zoo_dir, model_ids=model_ids)
    lat, depths, stages, midx = collect_latents(model, ds)
    splits = np.array([int(ds.meta_list[m].split_id) for m in midx])

    if reference_dir is not None and reference_ids:
        ds_ref = ResZoo(root_dir=reference_dir, model_ids=reference_ids)
        l2, d2, s2, m2 = collect_latents(model, ds_ref)
        lat = np.concatenate([lat, l2])
        depths = np.concatenate([depths, d2])
        stages = np.concatenate([stages, s2])
        '''label the backbone -1 so it never collides with a real expert id'''
        splits = np.concatenate([splits, np.full(len(m2), BACKBONE_LABEL)])

    return lat, depths, stages, splits


def plot_depth_grid(lat, depths, splits, depth_list, n_cols=3, tag='',
                    seed=0, perplexity=20, savepath=None):
    n_rows = int(np.ceil(len(depth_list) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    for k, d in enumerate(depth_list):
        ax = axes[k // n_cols][k % n_cols]
        sel = np.where(depths == d)[0]
        if len(sel) < 6:
            ax.set_title(f'depth {d}: too few points')
            ax.axis('off')
            continue

        emb = reduce_2d(lat[sel], seed, perplexity)
        lbl = splits[sel]
        for sp in sorted(set(lbl) - {BACKBONE_LABEL}):
            m = lbl == sp
            ax.scatter(emb[m, 0], emb[m, 1], c=EXPERT_COLORS[sp],
                       s=12, alpha=0.7, label=f'expert {sp}')

        '''backbone drawn last so it stays visible on top of the clusters'''
        mb = lbl == BACKBONE_LABEL
        if mb.any():
            ax.scatter(emb[mb, 0], emb[mb, 1], c='black', marker='*', s=90,
                       zorder=5, label='backbone', edgecolors='white', linewidths=0.4)

        ax.set_title(f'depth {d}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(len(depth_list), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis('off')

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=9, frameon=False)
    fig.suptitle(f'Latents by depth{tag}', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    if savepath:
        fig.savefig(savepath, dpi=150)
    plt.show()
    return fig


def expert_separation(lat, depths, splits, depth_list, exclude_backbone=True, verbose=True):
    '''within vs between expert distance, t-sne separation on few points is unreliable
    so this is the number to trust rather than the picture'''
    rows = []
    for d in depth_list:
        sel = np.where(depths == d)[0]
        lbl = splits[sel]
        if exclude_backbone:
            keep = lbl != BACKBONE_LABEL
            sel, lbl = sel[keep], lbl[keep]
        if len(sel) < 2:
            continue
        X = lat[sel].reshape(len(sel), -1)
        '''vectorized pairwise distances, the double loop is far too slow at this size'''
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        iu = np.triu_indices(len(X), k=1)
        same = lbl[:, None] == lbl[None, :]
        w = D[iu][same[iu]].mean()
        b = D[iu][~same[iu]].mean()
        rows.append((d, w, b, b / w))
        if verbose:
            print(f'depth {d:>2}: within {w:8.3f}  between {b:8.3f}  ratio {b/w:.3f}')
    return rows


def backbone_distance(lat, depths, splits, depth_list, verbose=True):
    '''the backbone is the common ancestor so it should sit near the expert centroid'''
    rows = []
    for d in depth_list:
        sel = np.where(depths == d)[0]
        if len(sel) == 0:
            continue
        X = lat[sel].reshape(len(sel), -1)
        lbl = splits[sel]
        bb = X[lbl == BACKBONE_LABEL]
        if len(bb) == 0:
            continue
        experts = sorted(set(lbl) - {BACKBONE_LABEL})
        dists = [np.linalg.norm(bb[:, None, :] - X[lbl == sp][None, :, :], axis=-1).mean()
                 for sp in experts]
        rows.append((d, experts, dists))
        if verbose:
            print(f'depth {d:>2}: backbone to experts ' +
                  '  '.join(f'e{sp}={v:7.3f}' for sp, v in zip(experts, dists)))
    return rows
