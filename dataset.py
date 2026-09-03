import os
import torch
import warnings
import chunking as C
from constants import *
import random, numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")

class ResZoo(Dataset):
    def __init__(self, root_dir='./zoo_chunks', model_ids=None):
        self.root_dir = root_dir
        '''filter to an explicit split, never load the whole directory blindly'''
        if model_ids is None:
            self.splits = sorted(os.listdir(root_dir))
        else:
            self.splits = sorted(model_ids)

        self.ids, self.chunks_list, self.mask_list, self.seq_index_list, self.meta_list = [], [], [], [], []
        '''flat index of (model_idx, row_start) so one item is one sequence'''
        self.index = []

        for split in self.splits:
            path = os.path.join(root_dir, split)
            chunks, mask, seq_index, meta = self.load_split_chunk(path)
            m = len(self.ids)
            self.ids.append(split)
            self.chunks_list.append(chunks)
            self.mask_list.append(mask)
            self.seq_index_list.append(seq_index)
            self.meta_list.append(meta)

            '''one entry per sequence, tokens_per_seq rows each'''
            T = meta.tokens_per_seq
            for start in range(0, chunks.shape[0], T):
                self.index.append((m, start))

        self.tokens_per_seq = self.meta_list[0].tokens_per_seq
        self.chunk_size = self.meta_list[0].chunk_size

        '''map each row back to its layer, needed for conditioning embeddings'''
        self.row_layer = []
        for meta in self.meta_list:
            lut = np.zeros(meta.n_chunks_total, dtype=np.int64)
            for li, lm in enumerate(meta.layers):
                lut[lm.chunk_start:lm.chunk_end] = li
            self.row_layer.append(lut)

    def load_split_chunk(self, path):
        chunks, mask, seq_index, meta = C.load(path)
        return chunks, mask, seq_index, meta

    def __len__(self):
        '''one item is one sequence, not one model'''
        return len(self.index)

    def __getitem__(self, idx):
        m, start = self.index[idx]
        end = start + self.tokens_per_seq
        lm_ids = self.row_layer[m][start:end]
        meta = self.meta_list[m]
        '''depth and stage of the layer this sequence came from'''
        depth = meta.layers[int(lm_ids[0])].depth_index
        stage = meta.layers[int(lm_ids[0])].stage
        return {
            'chunks': torch.from_numpy(self.chunks_list[m][start:end]).float(),
            'mask': torch.from_numpy(self.mask_list[m][start:end]).bool(),
            'depth': torch.tensor(depth, dtype=torch.long),
            'stage': torch.tensor(stage, dtype=torch.long),
            'model_idx': torch.tensor(m, dtype=torch.long),
            'row_start': torch.tensor(start, dtype=torch.long),
        }