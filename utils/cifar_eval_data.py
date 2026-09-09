from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

MEAN = (0.5071, 0.4865, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
)

'''no augmentation for evaluation'''
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
)


def load_cifar_eval(root='./res_data/'):
    '''returns the cifar datasets, the full 100 class test loader, and the class index groups'''
    train_dataset_cifar = CIFAR100(root, train=True, download=False, transform=train_transforms)
    test_dataset_cifar  = CIFAR100(root, train=False, download=False, transform=test_transforms)

    '''evaluate_grouped does the own/other split internally, so it needs the FULL 100 class set'''
    full_test_loader = DataLoader(test_dataset_cifar, batch_size=1000, shuffle=False,
                                  num_workers=4, pin_memory=True)

    '''class index groups, expert k owns classes 20k .. 20k+19'''
    zoo_train_feed = {i: [] for i in range(5)}
    zoo_test_feed  = {i: [] for i in range(5)}

    for idx, label in enumerate(train_dataset_cifar.targets):
        zoo_train_feed[label // 20].append(idx)

    for idx, label in enumerate(test_dataset_cifar.targets):
        zoo_test_feed[label // 20].append(idx)

    return (train_dataset_cifar, test_dataset_cifar, full_test_loader,
            zoo_train_feed, zoo_test_feed)
