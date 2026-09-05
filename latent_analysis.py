import torch
import warnings
import numpy as np
from dataset import ResZoo
from constants import device
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from torch.utils.data import DataLoader

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





@torch.no_grad()
def collect_latents_sampled(model, dataset, n_samples=8, batch_size=64, seed=0):
    '''draw from the posterior instead of taking mu, so the plot shows the
    spread the encoder assigns rather than a single point per sequence'''
    model.eval()
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    lat, depths, splits_idx = [], [], []
    for batch in loader:
        mu, logvar = model.encode(
            batch['chunks'].to(device),
            batch['depth'].to(device),
            batch['stage'].to(device))
        std = torch.exp(0.5 * logvar)
        for _ in range(n_samples):
            z = mu + std * torch.randn_like(std)
            lat.append(z.cpu().numpy())
            depths.append(batch['depth'].numpy())
            splits_idx.append(batch['model_idx'].numpy())

    return (np.concatenate(lat), np.concatenate(depths), np.concatenate(splits_idx))


def draw_ellipse(ax, pts, color, nsig=2.0):
    '''2 sigma ellipse from the empirical covariance of the projected samples'''
    c = pts.mean(0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse(c, w, h, angle=ang, facecolor=color,
                         alpha=0.15, edgecolor=color, lw=1.5))


def plot_overlap_grid(lat, depths, splits, depth_list, n_cols=3, tag='',
                      seed=0, perplexity=20, savepath=None):
    '''same grid as plot_depth_grid but each expert gets a 2 sigma ellipse,
    so overlap is visible rather than inferred from point positions'''
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
            ax.scatter(emb[m, 0], emb[m, 1], c=EXPERT_COLORS[sp], s=4, alpha=0.25)
            draw_ellipse(ax, emb[m], EXPERT_COLORS[sp])
            ax.scatter(*emb[m].mean(0), c=EXPERT_COLORS[sp], s=70, marker='o',
                       edgecolors='k', lw=1, zorder=6, label=f'expert {sp}')

        ax.set_title(f'depth {d}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for k in range(len(depth_list), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis('off')

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=9, frameon=False)
    fig.suptitle(f'Posterior overlap by depth{tag}', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    if savepath:
        fig.savefig(savepath, dpi=150)
    plt.show()
    return fig


@torch.no_grad()
def posterior_overlap(model, dataset, depth_list, batch_size=64, verbose=True):
    '''between expert distance measured in units of the encoder's own uncertainty.
    below 1 means the distributions overlap within their own spread'''
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    MU, LV, D, M = [], [], [], []
    for batch in loader:
        mu, logvar = model.encode(batch['chunks'].to(device),
                                  batch['depth'].to(device),
                                  batch['stage'].to(device))
        MU.append(mu.cpu().numpy()); LV.append(logvar.cpu().numpy())
        D.append(batch['depth'].numpy()); M.append(batch['model_idx'].numpy())
    MU = np.concatenate(MU); LV = np.concatenate(LV)
    D = np.concatenate(D); M = np.concatenate(M)
    spl = np.array([int(dataset.meta_list[m].split_id) for m in M])

    rows = []
    for d in depth_list:
        sel = np.where(D == d)[0]
        if len(sel) < 2:
            continue
        experts = sorted(set(spl[sel]))
        means = {sp: MU[sel][spl[sel] == sp].reshape((spl[sel] == sp).sum(), -1).mean(0)
                 for sp in experts}
        '''total posterior spread over the flattened latent'''
        sigma = np.sqrt((np.exp(LV[sel]) ** 1).sum(axis=(1, 2))).mean()
        ds = [np.linalg.norm(means[a] - means[b])
              for i, a in enumerate(experts) for b in experts[i + 1:]]
        ratio = np.mean(ds) / sigma
        rows.append((d, np.mean(ds), sigma, ratio))
        if verbose:
            print(f'depth {d:>2}: between-expert {np.mean(ds):8.3f}  '
                  f'posterior sigma {sigma:8.3f}  ratio {ratio:.3f}')
    return rows