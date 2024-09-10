import math
import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re

class SmileDataset(Dataset):

    def __init__(self, data, content, block_size, aug_prob = 0, prop = None, scaffold = None, scaffold_maxlen = None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles, prop, scaffold = self.data[idx], self.prop[idx], self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()
        
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        
        original_smiles = smiles
        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)
        smiles_tokens = regex.findall(smiles)
        if self.max_len - len(smiles_tokens) < 0:
            smiles = original_smiles
            
        if self.max_len - len(regex.findall(smiles)) >= 0:
            smiles += str('<')*(self.max_len - len(regex.findall(smiles)))
        else:
            smiles = smiles[:self.max_len]

        smiles=regex.findall(smiles)
        
        if self.max_len >= len(regex.findall(scaffold)):
            scaffold += str('<')*(self.scaf_max_len - len(regex.findall(scaffold))) 
        else:
            scaffold = scaffold[:self.scaf_max_len]

        scaffold=regex.findall(scaffold)

        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype = torch.float)
        return x, y, prop, sca_tensor
