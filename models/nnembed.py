
from utils.utils import load_meddra_id2name
from gensim.models import KeyedVectors
import fasttext

from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()

class DisambigModel:
    
    def __init__(self):
        self.meddra = load_meddra_id2name()

        self.model = fasttext.load_model("pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")
        self.kv = self._generate_kv()
        
    def _generate_kv(self):
        ids, embeds = [], []
        for id, term in self.meddra.items():
            embeds.append(self.model.get_sentence_vector(term))
            # print(term, id, embeds[-1].shape)
            ids.append(id)
            
        kv = KeyedVectors(200)
        kv.add_vectors(ids, embeds)
        return kv
    
    def predict_ids(self, terms: list[str]):
        """ given a list of N terms (phrases), return a list of N closest matches to meddra dataset """
        matches = []
        # assert type(self.kv) is KeyedVectors
        for term in terms:
            embed = self.model.get_sentence_vector(term)
            matches.append(
                self.kv.most_similar(positive=[embed], topn=1)[0][0]
                )
        return matches
        
    




class DisambigRuleModel:
    
    def __init__(self):
        self.meddra = load_meddra_id2name()
        self.meddra_term2id = {v: k for k,v in self.meddra.items()}

        self.model = fasttext.load_model("pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")
        self.kv = self._generate_kv()
        
    def _preprocess(self, term: str):
        term = term.replace("'", "").replace("\"","").replace(",", "").replace("`", "").replace(";", "")
        return " ".join(sorted([stemmer.stem(w) for w in word_tokenize(term)]))
        
    def _generate_kv(self):
        ids, embeds = [], []
        
        for id, term in self.meddra.items():
            clean_term = self._preprocess(term)
            # print(clean_term)
            embeds.append(self.model.get_word_vector(clean_term))
            # print(term, id, embeds[-1].shape)
            ids.append(id)
            
        kv = KeyedVectors(200)
        kv.add_vectors(ids, embeds)
        return kv
    
    def predict_ids(self, terms: list[str]):
        """ given a list of N terms (phrases), return a list of N closest matches to meddra dataset """
        matches = []
        # assert type(self.kv) is KeyedVectors
        for term in terms:
            clean_term = self._preprocess(term)
            # print(clean_term)
            embed = self.model.get_word_vector(clean_term)
            matches.append(
                self.kv.most_similar(positive=[embed], topn=1)[0][0]
                )
        return matches
        
    

