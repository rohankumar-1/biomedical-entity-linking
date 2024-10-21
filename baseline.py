from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import get_tac
from evaluation import evaluate_f1

from utils import levenshtein, load_meddra
from random import choice

# predict all reactions (mentions)

class Baseline:
    
    def __init__(self):
        self.recognition_model = GaussianNB()
        self.MEDDRA = load_meddra()
        
    def fit(self, trainset, labels):
        """ trains the Naive Bayes estimator to find all mentions given training set and labels
        
        trainset: list of texts, one per drug
        labels: list of list of entities to extract from each text (mentions), one list per drug
        """
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(trainset)
        self.recognition_model.fit(X, labels)
    
    def _predict_mentions(self, text):
        return self.recognition_model.predict(text)
    
    def predict(self, texts: list[str]):
        guesses = self.recognition_model.predict(texts)
        return self.recognition_model.predict(texts)
    
        
if __name__=="__main__":
    
    # train
    X_train, MENTIONS_LIST_train, _ = get_tac("train")
    
    model = Baseline()
    model.fit(X_train, MENTIONS_LIST_train)
    
    # test
    X_test, MENTIONS_LIST_test, REACTIONS_LIST_test = get_tac("test")
    guesses = model._predict_mentions(X_test[0])
    
    print(guesses)
    
    # f1_score = evaluate_f1(guesses, labels)
    
    # print("Model achieved an f1 of:", f1_score)

# compute levenstein distance between each reaction and all entries in knowledge base (MedDRA)