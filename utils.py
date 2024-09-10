import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from moses.utils import get_mol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import json   
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def sample_search(model, prompt, block_size, topk, scaffold=None, prop=None, vocab='stoi.json', rev_vocab='itos.json'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断是否有GPU可以使用
    model.eval()  # 将模型设置为评估模式
    model.to(device)  # 将模型移动到设备上

    with open(vocab, 'r') as f:  # 加载词汇表
        vocab = json.load(f)
    with open(rev_vocab, 'r') as f:  # 加载反向词汇表
        rev_vocab = json.load(f)
    if prompt.dim() == 1:  # 如果prompt是一维的，增加一个维度
        prompt = prompt.unsqueeze(0)
    if scaffold.dim() == 1:  # 如果scaffold是一维的，增加一个维度
        scaffold = scaffold.unsqueeze(0)

    input_ids = prompt.to(device)  # 将输入移动到设备上

    with torch.no_grad():
        for _ in range(block_size - 1):  # 对于每一个block
            outputs = model(input_ids, scaffold=scaffold, prop=prop).logits  # 获取模型输出
            next_token_logits = outputs[-1, :] if outputs.dim() == 2 else outputs[:, -1, :]  # 获取下一个token的logits
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)  # 计算下一个token的概率
            
            # 应用top-k采样
            topk_probs, topk_ids = torch.topk(next_token_probs, topk)  # 获取最可能的topk个token
            topk_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)  # 归一化概率
            sampled_ids = torch.multinomial(topk_probs, 1)  # 从topk概率中采样
            
            # 在原始词汇表中找到对应的ids
            sampled_ids = topk_ids.gather(-1, sampled_ids)
            if any(str(id.item()) not in rev_vocab for id in sampled_ids.view(-1)):
                continue  # 如果有任何一个id不在rev_vocab中，跳过这个循环
            
            input_ids = torch.cat([input_ids, sampled_ids], dim=-1)  # 添加新的token到输入序列
    
    all_smiles = [''.join([rev_vocab[str(tok.item())] for tok in input_ids[i]]) for i in range(prompt.shape[0])]  # 将最终的输入序列转换为SMILES字符串
            
    return all_smiles  # 返回所有的SMILES字符串


@torch.no_grad()
def beam_search(model, prompt, block_size, beam_width, scaffold=None, prop=None, vocab='vocab.json', rev_vocab='rev_vocab.json'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断是否有GPU可以使用
    model.eval()  # 将模型设置为评估模式
    model.to(device)  # 将模型移动到设备上

    with open(vocab, 'r') as f:  # 加载词汇表
        vocab = json.load(f)
    with open(rev_vocab, 'r') as f:  # 加载反向词汇表
        rev_vocab = json.load(f)
    if prompt.dim() == 1:  # 如果prompt是一维的，增加一个维度
        prompt = prompt.unsqueeze(0)
    if scaffold.dim() == 1:  # 如果scaffold是一维的，增加一个维度
        scaffold = scaffold.unsqueeze(0)

    input_ids = prompt.to(device)  # 将输入移动到设备上
    beam_candidates = [[(input_ids[i,:], 0)] for i in range(prompt.shape[0])]  # 初始化候选序列和得分

    with torch.no_grad():
        for _ in range(block_size - 1):  # 对于每一个block
            new_candidates = [[] for _ in range(prompt.shape[0])]  # 初始化新的候选序列

            for k in range(len(beam_candidates[0])):  # 对于每一个候选序列
                k_column = [item[k] for item in beam_candidates]
                input_ids = torch.stack([item[0] for item in k_column]).to(device)  # 获取输入序列
                score = torch.tensor([item[1] for item in k_column]).to(device)  # 获取得分

                outputs,_,_ = model(input_ids, scaffold=scaffold, prop=prop)  # 获取模型输出
                next_token_logits = outputs[-1, :] if outputs.dim() == 2 else outputs[:, -1, :]  # 获取下一个token的logits
                next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)  # 计算下一个token的概率
                
                topk_probs, topk_ids = next_token_probs.topk(beam_width)  # 获取最可能的topk个token
                
                for i in range(beam_width):  # 对于每一个topk token
                    new_input_ids = torch.cat([input_ids, topk_ids[:, i:i+1]], dim=-1)  # 添加新的token到输入序列
                    new_score = score + topk_probs[:, i].log()  # 更新得分
                    for j in range(prompt.shape[0]):
                        new_candidates[j].append((new_input_ids[j], new_score[j]))  # 添加新的候选序列和得分

            for i in range(prompt.shape[0]):
                new_candidates[i] = sorted(new_candidates[i], key=lambda x: x[1], reverse=True)[:beam_width]  # 保留得分最高的topk个候选序列

            beam_candidates = new_candidates  # 更新候选序列

    all_smiles = [''.join([rev_vocab[str(tok.item())] for tok in beam_candidates[i][0][0]]) for i in range(prompt.shape[0])]  # 将最终的候选序列转换为SMILES字符串
            
    return all_smiles  # 返回所有的SMILES字符串

import re
from tqdm import tqdm
from rdkit import RDLogger
import pandas as pd
def g_test(model,data,block_size,scaffold_maxlen,stoi,itos, batch_size = 512, num = 10):
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = stoi
    rev_vocab = itos
    train_data, s_data = train_test_split(data, test_size=0.1, random_state=42)
    test_data, val_data = train_test_split(s_data, test_size=0.5, random_state=42)
    g_data = test_data.sample(n=num, random_state=42).reset_index(drop=True).drop_duplicates(subset='scaffold_smiles', keep='first').reset_index(drop=True)
    prop_item = ['logp']
    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    prompt_smiles = g_data['smiles'].to_list()
    prompt_list = [i[:1] for i in prompt_smiles]
    scaffold = g_data['scaffold_smiles'].to_list()
    scaffold_list = [[vocab.get(char,vocab.get('C')) for char in regex.findall(s)] + [vocab.get('<')] * (scaffold_maxlen - len(regex.findall(s))) for s in scaffold]
    prop_values = [item[0] for item in g_data[prop_item].values]
    
    smiles_list = []
    smiles_dict = {}
    for i in tqdm(range(len(prompt_list))):
        prompt = torch.full((batch_size, 1), vocab.get(prompt_list[i],vocab.get('C'))).to(device)
        scaffold_input_ids = torch.from_numpy(np.repeat([scaffold_list[i]], batch_size, axis=0)).to(device)
        prop = torch.full((batch_size, 1), prop_values[i]).to(device)
        beam_width = len(whole_string)
        smiles_search = sample_search(model, prompt, block_size, beam_width, scaffold=scaffold_input_ids, prop=prop)
        
        if model.config.scaffold:
            scaffold_value = scaffold[i]
            column_name = f"{scaffold_value}"
            if column_name not in smiles_dict:
                smiles_dict[column_name] = smiles_search
            else:
                smiles_dict[column_name].extend(smiles_search)
        else:
            smiles_list.extend(smiles_search)
    if model.config.scaffold:
        cleaned_smiles_dict = {}
        for scaffold, molecules in smiles_dict.items():
            cleaned_molecules = [molecule.replace('<', '') for molecule in molecules]
            cleaned_smiles_dict[scaffold] = cleaned_molecules
        smiles_dict = cleaned_smiles_dict
        valid_smiles_dict = {}
        for scaffold, molecules in smiles_dict.items():
            valid_molecules = [molecule for molecule in molecules if is_valid(molecule)]
            valid_smiles_dict[scaffold] = valid_molecules
        similarity_values = []
        for scaffold, molecules in valid_smiles_dict.items():
            similar_molecules = [molecule for molecule in molecules if tanimoto_similarity(get_scaffold(molecule), scaffold) > 0.8]
            similarity_values.append(len(similar_molecules) / (len(molecules)))
        smiles_list = [item for sublist in smiles_dict.values() for item in sublist]
        valid_ratio,val_smiles = calculate_validity(smiles_list)
        uni_ratio = calculate_uniqueness(val_smiles)
        num_molecules = [len(molecules) for _, molecules in valid_smiles_dict.items()]
        total_similarity = sum(s * n for s, n in zip(similarity_values, num_molecules)) / (sum(num_molecules))
        return valid_ratio,uni_ratio,total_similarity
    else:
        clean_smiles_list = []
        clean_smiles_list = [molecule.replace('<', '') for molecule in smiles_list]
        smiles_list = clean_smiles_list
        valid_ratio,val_smiles = calculate_validity(smiles_list)
        nov_ratio = calculate_novelty(smiles_list,train_data['smiles'].tolist())
        uni_ratio = calculate_uniqueness(val_smiles)
        print('---------------------------')
        print(valid_ratio,uni_ratio,nov_ratio)
        print('---------------------------')
        return valid_ratio,uni_ratio
    
    
def check_frozen_parameters(model):
    print("Frozen Parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter: {name}")

def freeze_layers(model,n_blocks,freeze_conv=False):
    # 冻结type embeddings
    for param in model.backbone.type_emb.parameters():
        param.requires_grad = False
    # 冻结token embeddings
    for param in model.backbone.tok_emb.parameters():
        param.requires_grad = False
        
    # 冻结卷积层
    if freeze_conv:
        for idx, block in enumerate(model.backbone.layers):
            if idx < n_blocks:
                for name, param in block.named_parameters():
                    if 'conv1d' in name: param.requires_grad = False
    else:
        # 冻结特定的块
        for idx, block in enumerate(model.backbone.layers):
            if idx < n_blocks:
                for param in block.parameters():
                    param.requires_grad = False
                    
def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprint(mol1,3)
    fp2 = AllChem.GetMorganFingerprint(mol2,3)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def mol_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_novelty(generated_smiles, training_smiles):
    training_set = set(training_smiles)  # 将训练集转换为集合
    generated_set = set(generated_smiles)  # 将生成的SMILES字符串转换为集合

    # 过滤掉无效的SMILES字符串
    generated_set = {smiles for smiles in generated_set if Chem.MolFromSmiles(smiles) is not None}

    num_novel = len(generated_set - training_set)  # 计算新颖的SMILES字符串数
    novelty = num_novel / len(generated_set)  # 计算新颖性

    return novelty

def calculate_validity(generated_smiles):
    val_smiles = []
    def is_valid_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            val_smiles.append(smiles)
        return mol is not None

    valid_smiles_count = sum(is_valid_smiles(smiles) for smiles in generated_smiles)
    valid_ratio = valid_smiles_count / len(generated_smiles)
    return valid_ratio,val_smiles

def calculate_uniqueness(generated_smiles):
    unique_molecules = set(generated_smiles)
    uniqueness = len(unique_molecules) / (len(generated_smiles))
    return uniqueness

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

    #Experimental Class for Smiles Enumeration, Iterator and SmilesIterator adapted from Keras 1.2.2

class Iterator(object):
    """Abstract base class for data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)




class SmilesIterator(Iterator):
    """Iterator yielding data from a SMILES array.
    # Arguments
        x: Numpy array of SMILES input data.
        y: Numpy array of targets data.
        smiles_data_generator: Instance of `SmilesEnumerator`
            to use for random SMILES generation.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        dtype: dtype to use for returned batch. Set to keras.backend.floatx if using Keras
    """

    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + [ self.smiles_data_generator.pad, self.smiles_data_generator._charlen]), dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j+1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer
    
    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot

      
    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """       
        smiles = []
        for v in vect:
            #mask v 
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)