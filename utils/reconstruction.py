import torch
from . import chunking as C
from .resnet20 import resnet20
from .constants import device
import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def reconstruct_model(tvae, loader, dataset, target_state_dict):
    """It takes the tvae and the dataset related to one expert and try to encode and reconstruct that again."""
    assert len(dataset.meta_list) == 1, 'reconstruct_model expects a single model dataset'
    meta = dataset.meta_list[0]

    chunks_hat = []
    tvae.eval()
    for batch in loader:
        chunks = batch['chunks'].to(device, non_blocking=True)
        depth = batch['depth'].to(device, non_blocking=True)
        stage = batch['stage'].to(device, non_blocking=True)

        xhat, _, _ = tvae(chunks, depth, stage)

        '''append the RECONSTRUCTION, not the input'''
        chunks_hat.append(xhat.cpu())

    '''flatten [n_seq, 16, 144] back to [1904, 144] in the original row order'''
    chunks = torch.concat(chunks_hat).reshape(-1, meta.chunk_size).numpy()
    assert chunks.shape[0] == meta.n_chunks_total, f'row count {chunks.shape[0]} != manifest {meta.n_chunks_total}'

    out = C.reconstruct_state_dict(chunks, meta, target_state_dict)

    return out


@torch.no_grad()
def evaluate_grouped(model, loader, expert_id):
    '''accuracy inside the expert's own 20 classes vs everything else'''
    model.eval()
    lo, hi = expert_id * 20, expert_id * 20 + 20
    own_c = own_n = oth_c = oth_n = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        pred = model(images).argmax(1)
        mask = (labels >= lo) & (labels < hi)
        own_c += (pred[mask] == labels[mask]).sum().item();  own_n += mask.sum().item()
        oth_c += (pred[~mask] == labels[~mask]).sum().item(); oth_n += (~mask).sum().item()
    return (100.0 * own_c / max(own_n, 1),
            100.0 * oth_c / max(oth_n, 1),
            100.0 * (own_c + oth_c) / (own_n + oth_n))


def reset_bn(model):
    '''running stats are properties of the weights and data together,
    they are invalid after any weight modification'''
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches=100):
    '''train mode lets the running stats update, no optimizer step so weights are untouched'''
    reset_bn(model)
    model.train()
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(images.to(device, non_blocking=True))
    model.eval()
    return model


def evaluate_reconstruction(sd_hat, sd_orig, expert_id, recal_loader, test_loader):
    '''compare the reconstructed weights against the original on the same splits'''
    m_hat = resnet20(num_classes=100).to(device)
    m_hat.load_state_dict(sd_hat, strict=True)
    recalibrate_bn(m_hat, recal_loader)

    m_ref = resnet20(num_classes=100).to(device)
    m_ref.load_state_dict(sd_orig, strict=True)
    m_ref.eval()

    own_r, oth_r, all_r = evaluate_grouped(m_ref, test_loader, expert_id)
    own_h, oth_h, all_h = evaluate_grouped(m_hat, test_loader, expert_id)

    print(f'{"":<14}{"own":>8}{"other":>8}{"all":>8}')
    print(f'{"original":<14}{own_r:>8.2f}{oth_r:>8.2f}{all_r:>8.2f}')
    print(f'{"reconstructed":<14}{own_h:>8.2f}{oth_h:>8.2f}{all_h:>8.2f}')
    print(f'{"delta":<14}{own_h-own_r:>8.2f}{oth_h-oth_r:>8.2f}{all_h-all_r:>8.2f}')

    '''per layer relative frobenius error, read as a distribution not a mean'''
    print()
    errs = []
    for k in sd_hat:
        if k not in sd_orig or sd_orig[k].dtype != torch.float32 or sd_orig[k].dim() < 2:
            continue
        a, b = sd_orig[k].float(), sd_hat[k].float().cpu()
        e = (a - b).norm() / a.norm()
        errs.append((k, e.item()))
    for k, e in sorted(errs, key=lambda x: -x[1])[:5]:
        print(f'  worst rel err  {k:<26} {e:.4f}')

    '''the specific tail test, this weight sits ~11 sigma out and is stable across the zoo'''
    v = sd_hat['layer1.2.conv2.weight'].flatten()[1249].item()
    ref = sd_orig['layer1.2.conv2.weight'].flatten()[1249].item()
    print(f'\n  index 1249 of layer1.2.conv2: original {ref:.4f}  reconstructed {v:.4f}')

    return all_h - all_r



def interp_to_state_dict(out, dataset, target_state_dict):
    '''same reassembly as reconstruct_model, just starting from a decoded tensor'''
    meta = dataset.meta_list[0]
    chunks = out.reshape(-1, meta.chunk_size).numpy()
    assert chunks.shape[0] == meta.n_chunks_total
    return C.reconstruct_state_dict(chunks, meta, target_state_dict)


def eval_state_dict(sd, expert_id, recal_loader, test_loader):
    '''load, recalibrate batchnorm, evaluate'''
    m = resnet20(num_classes=100).to(device)
    m.load_state_dict(sd, strict=True)
    recalibrate_bn(m, recal_loader)
    return evaluate_grouped(m, test_loader, expert_id)