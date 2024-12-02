
from .utils import START_TOKEN, STOP_TOKEN, UNK_TOKEN, load_meddra_id2name
from .preprocessing import ner_get_preprocessed_data, disambig_get_preprocessed_data, BTAG, ITAG
from typing import Mapping
import random

class BiLSTM_CRF_Dataset:
    
    def __init__(self, dataset:str="train"):
        self.start_token : str = START_TOKEN
        self.stop_token : str = STOP_TOKEN
        self.unk_token : str = UNK_TOKEN
        self.data : list[list[str]] = None
        self.tags : list[list[str]] = None
        self.data, self.tags = ner_get_preprocessed_data(dataset=dataset)
        
        self._get_tagset()
        self._get_vocab()
        
        
    def _get_tagset(self):
        self.tag_to_idx : Mapping[str, int]= {}
        
        self.tag_to_idx[self.start_token] = 0
        self.tag_to_idx[self.stop_token] = 1
        self.tag_to_idx[self.unk_token] = 2
        self.tag_to_idx['O'] = 3
        
        for tagline in self.tags:
            for tag in tagline:
                if tag == 'O':
                    continue
                
                # split tag into BIO and type, then for both B and I, add B-Type and I-Type
                # print(tag.split("-"))
                _, name = tuple(tag.split("--"))
                
                candidates = [BTAG+name, ITAG+name]
                for cand in candidates:
                    if cand not in self.tag_to_idx.keys():
                        self.tag_to_idx[cand] = len(self.tag_to_idx)
                    
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
                    
    
    def _get_vocab(self):
        self.word_to_idx : Mapping[str, int]= {}
        
        self.word_to_idx[UNK_TOKEN] = 0
        
        for sent in self.data:
            for word in sent:
                if word not in self.word_to_idx.keys():
                    self.word_to_idx[word] = len(self.word_to_idx)
                    
    def iterate(self):
        """ a generator function for the data (RANDOMIZED) """
        
        all_data = list(zip(self.data, self.tags))
        random.shuffle(all_data)
        
        for sentence, tagline in all_data:
            yield (sentence, tagline)
                    
        
        
class DisambiguationDataset:
    
    def __init__(self):
        self.all_drugs = disambig_get_preprocessed_data()
        self.keys = list(self.all_drugs.keys())
        random.shuffle(self.keys)
        
    def iterate(self):
        """ a generator function for the data """
        for drug in self.keys:
            yield (drug, self.all_drugs[drug])
        
        
        
        
        