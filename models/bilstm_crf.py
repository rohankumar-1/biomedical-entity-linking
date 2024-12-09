# Implementation based on Robert Guthrie's https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

import torch
import torch.nn as nn
import numpy as np

import fasttext
from utils.utils import START_TOKEN, STOP_TOKEN, UNK_TOKEN


def log_sum_exp(smat):
    # status matrix (smat): (tagset_size, tagset_size)
    # @return (1, tagset_size)
    max_score = smat.max(dim=0, keepdim=True).values
    return (smat - max_score).exp().sum(axis=0, keepdim=True).log() + max_score


class BiLSTM_CRF(nn.Module):

    def __init__(self, 
                 tag_to_ix, 
                 word_to_ix, 
                 embedding_dim, 
                 hidden_dim, 
                 word2vec_embeds: bool = False):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.word_to_ix = word_to_ix
        self.tagset_size = len(tag_to_ix)
        self.vocab_size = len(word_to_ix)
        
        self.w2v = word2vec_embeds
        if self.w2v:
            print("Using word2vec-style embedding model...")
            self.biow2v = fasttext.load_model("pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")
            
            print("Loaded embedding model.")
            def embed(seq):
                embed_seq = np.array([], dtype=np.float64)
                for s in seq:
                    embed_seq = np.concatenate([embed_seq, self.biow2v[s]], axis=0)
                    
                return torch.tensor(embed_seq, dtype=torch.float32)
            
            self.word_embeds = embed
        else:
            print("Using embedding layer...")
            self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True)
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TOKEN], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TOKEN]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim), torch.randn(2, 1, self.hidden_dim))
    
    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        alpha = torch.full((1, self.tagset_size), -10000.0)
        alpha[0][self.tag_to_ix[START_TOKEN]] = 0.0
        for feat in feats:
            alpha = log_sum_exp(alpha.T + feat.unsqueeze(0) + self.transitions)
        return log_sum_exp(alpha.T + 0 + self.transitions[:, [self.tag_to_ix[STOP_TOKEN]]]).flatten()[0]
        
    def _prepare_sequence(self, seq):
        """ adjusted for UNK_TOKEN when testing """
        idxs = []
        for w in seq:
            if w in self.word_to_ix.keys():
                idxs.append(self.word_to_ix[w])
            else:
                idxs.append(self.word_to_ix[UNK_TOKEN])
        return torch.tensor(idxs, dtype=torch.long)
    

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        
        if self.w2v is False:
            sentence = self._prepare_sequence(seq=sentence)
            
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TOKEN]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag_to_ix[STOP_TOKEN], tags[-1]]
        return score

    
    def _viterbi_decode(self, feats):
        backtrace = []
        alpha = torch.full((1, self.tagset_size), -10000.0)
        alpha[0][self.tag_to_ix[START_TOKEN]] = 0
        for feat in feats:
            smat = (alpha.T + feat.unsqueeze(0) + self.transitions)  # (tagset_size, tagset_size)
            backtrace.append(smat.argmax(0))  # column_max
            alpha = log_sum_exp(smat)
        # backtrack
        smat = alpha.T + 0 + self.transitions[:, [self.tag_to_ix[STOP_TOKEN]]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # ignore START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return log_sum_exp(smat).item(), best_path[::-1]  # item() return list?



    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    
    

        
    
