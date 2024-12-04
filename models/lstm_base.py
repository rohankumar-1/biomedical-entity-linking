
import torch
import torch.nn as nn

from utils.utils import UNK_TOKEN



class BaseLSTM(nn.Module):

    def __init__(self, tag_to_ix, word_to_ix, embedding_dim, hidden_dim, word2vec_embeds: bool = False):
        super(BaseLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.word_to_ix = word_to_ix
        self.tagset_size = len(tag_to_ix)
        self.vocab_size = len(word_to_ix)
        
        # layers
        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.hidden = self.init_hidden()

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)


    def init_hidden(self):
        return (torch.randn(1, 1, self.hidden_dim), torch.randn(1, 1, self.hidden_dim))
        
    def _prepare_sequence(self, seq):
        """ adjusted for UNK_TOKEN when testing """
        idxs = []
        for w in seq:
            if w in self.word_to_ix.keys():
                idxs.append(self.word_to_ix[w])
            else:
                idxs.append(self.word_to_ix[UNK_TOKEN])
        return torch.tensor(idxs, dtype=torch.long)
    
    def forward(self, sentence):  
        self.hidden = self.init_hidden()    
        sentence = self._prepare_sequence(sentence)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        return self.hidden2tag(lstm_out)
    
    

        
    