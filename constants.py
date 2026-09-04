import torch
import random, numpy as np

SEED = 1246
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''size of one chunk, chosen so every kept conv filter divides evenly'''
CHUNK_SIZE = 144

'''number of chunks grouped into one transformer sequence'''
TOKENS_PER_SEQ = 16

'''keys excluded from the vae, they are copied from the target model instead'''
'''stem conv is excluded because its filter size 27 does not divide 144'''
EXCLUDE_EXACT = {"conv1.weight"}

'''option-b projection shortcuts, only present in some resnet variants'''
EXCLUDE_SUBSTRINGS = ("downsample",)

'''include the classifier head, needed when experts specialize by class'''
INCLUDE_LINEAR = True


IMG_PATH = "/fp/homes01/u01/ec-armans/DLAI/imgs"