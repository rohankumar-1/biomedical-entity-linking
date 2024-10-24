from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import DictVectorizer

from nltk.metrics.distance import jaro_similarity

from preprocessing import get_tac, preprocess_dataset
from evaluation import evaluate_f1

from utils import levenshtein, load_meddra, iob_to_entities
from random import choice

import numpy as np
from tqdm import tqdm


class Baseline:
    
    def __init__(self):
        self.recognition_model = GaussianNB()
        self.MEDDRA = load_meddra()
        self.vectorizer = DictVectorizer()
        self.fitted = False
        
        
    def fit(self, trainset, labels):
        """ trains the Naive Bayes estimator to find all mentions given training set and labels
        
        trainset: list of texts, one per drug
        labels: list of list of entities to extract from each text (mentions), one list per drug
        """
        
        features, labels = preprocess_dataset(trainset, labels)
        print(". . . . Preprocessed dataset")
        
        # fit+transform vectorizer, then fit the Naive Bayes
        X = self.vectorizer.fit_transform(features)
        print(". . . . Fit vectorizer")
        self.recognition_model.fit(np.asarray(X.todense()), labels)
        print(". . . . Fit recognition model")
        
        # update state
        self.fitted = True
        
    def _find_closest_meddra_id(self, y_preds):
        """ given a list of guessed entities, find the closest meddra id using jaro-winkler distance """

        pred_ids = []
        for entity in tqdm(y_preds):
            max_sim, closest = 0, None
            for k, v in self.MEDDRA.items():
                sim = jaro_similarity(entity[0], v)
                if sim > max_sim:
                    max_sim = sim
                    closest = str(k)
                    
            pred_ids.append(closest)
            
        return pred_ids
    
    
    def predict(self, text: list[str]):
        if not self.fitted:
            raise Exception("model has not been fit yet . . . ")
        
        features, _ = preprocess_dataset([text], [[]])
        guesses = self.recognition_model.predict(np.asarray(self.vectorizer.transform(features).todense()))
    
        pred_entities = iob_to_entities(features, guesses)
        
        pred_ids = set(self._find_closest_meddra_id(pred_entities))
        
        return pred_ids
    
    
        
if __name__=="__main__":
    
    # train
    X_train, entities_train, _ = get_tac("train")
    print(". . . Loaded training data")
    
    model = Baseline()
    SIZE = 100
    model.fit(X_train[:SIZE], entities_train[:SIZE])
    print(". . . Fit baseline model")
    
    # test
    X_test, entites_test, reactions_test = get_tac("test")
    print(". . . Loaded testing data")
    for i in range(5):
        pred_ids = model.predict(X_test[i])
        f1_score = evaluate_f1(pred_ids, [react["meddra_pt_id"] for react in reactions_test[1]])
        print(f"Model achieved an f1 on sample {i}:", f1_score)

# compute levenstein distance between each reaction and all entries in knowledge base (MedDRA)