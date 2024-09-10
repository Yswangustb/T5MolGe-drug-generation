import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import MambaLMHeadModel,MambaConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
from sklearn.model_selection import train_test_split

from utils import *
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--model_path', type=str,
                        help="path of model", default='', required=False)
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')

    parser.add_argument('--data_name', type=str, default='tinib_random',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--n_blocks', type=int, default=0, help='Number of blocks to freeze')
    parser.add_argument('--freeze_conv', action='store_true', help='Flag to freeze all conv')
    
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--aug_prob', type=float, default = 0, help="aug_prob", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=30,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=256,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    
    args = parser.parse_args()
    
    set_seed(42)
    
    wandb.init(project="transfer_gpt", name=args.run_name)
    
    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
    
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    data = pd.read_csv('datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.001).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    
    # 先将数据分为训练集和其它（验证集+测试集）
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42,shuffle = True)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    
    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props
    
    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']
    
    max_len = 100
    scaffold_max_len=100
    
    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]
    
    scaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in vscaffold]
    
    train_dataset = SmileDataset(smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen= scaffold_max_len)
    valid_dataset = SmileDataset(vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen= scaffold_max_len)
    
    mconf = MambaConfig(n_layer = args.n_layer,num_props = args.num_props,scaffold = args.scaffold,dropout = 0.175)
    model = MambaLMHeadModel(mconf,device = 'cuda',dtype = torch.float32)
    
    path = args.model_path
    model.load_state_dict(torch.load(path))
    freeze_layers(model,args.n_blocks,freeze_conv = args.freeze_conv)
#     l = check_frozen_parameters(model)
    
#     with open("output.txt", "a") as f:  
#         # 写入 run_name 以及 best_loss, valid_ratio, 和 unique_ratio
#         f.write(f"{args.run_name}: {l}\n")
        
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=0, ckpt_path=f'./cond_gpt/weights/{args.run_name}/{args.run_name}.pt', block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)
    
    best_loss,valid_ratio,uni_ratio,similar_ratio = trainer.train(wandb,generate = True)
    
    with open("output.txt", "a") as f:  
        # 写入 run_name 以及 best_loss, valid_ratio, 和 unique_ratio
        f.write(f"{args.run_name}: {best_loss},{valid_ratio},{uni_ratio},{similar_ratio}\n")