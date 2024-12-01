
from utils import START_TOKEN, STOP_TOKEN, UNK_TOKEN
from preprocessing import ner_get_preprocessed_data
from typing import Mapping

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
        
        for tagline in self.tags:
            for tag in tagline:
                if tag not in self.tag_to_idx.keys():
                    self.tag_to_idx[tag] = len(self.tag_to_idx)
                    
        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
                    
    
    def _get_vocab(self):
        self.word_to_idx : Mapping[str, int]= {}
        
        for sent in self.data:
            for word in sent:
                if word not in self.word_to_idx.keys():
                    self.word_to_idx[word] = len(self.word_to_idx)
                    
    def iterate(self):
        """ a generator function for the data """
        for sentence, tagline in zip(self.data, self.tags):
            yield (sentence, tagline)
                    
        
                    
        
        
if __name__=="__main__":
    td = BiLSTM_CRF_Dataset("train")
    print(td.tag_to_idx)
    print(td.word_to_idx)