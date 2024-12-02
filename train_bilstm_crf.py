    
from models import BiLSTM_CRF, prepare_sequence
import torch.optim as optim
import torch
from tqdm import tqdm

from utils import BiLSTM_CRF_Dataset
from evaluation import evaluate


def test_and_write(model: BiLSTM_CRF, test_data: BiLSTM_CRF_Dataset, outfile: str):
    
    true_tags = []
    pred_tags = []
    
    with torch.no_grad():
        for sentence, tags in tqdm(test_data.iterate(), desc="Testing... ", total=len(test_data.data)):
            sentence_in = prepare_sequence(sentence, train_data.word_to_idx)
            targets = [train_data.tag_to_idx[t] for t in tags]
            
            preds = model(sentence_in)[1]
            
            pred_tags.extend([test_data.idx_to_tag[p] for p in preds])
            true_tags.extend([test_data.idx_to_tag[t] for t in targets])
            
    evaluate(true_tags, pred_tags, verbose=True)
        
        
    
if __name__=="__main__":
    EMBEDDING_DIM = 64
    HIDDEN_DIM=32
    
    # output files
    UNTRAINED_OUTFILE = "generated/untrained.out"
    TRAINED_OUTFILE = "generated/trained.out"
    
    train_data = BiLSTM_CRF_Dataset("train")
    test_data = BiLSTM_CRF_Dataset("test")

    model = BiLSTM_CRF(len(train_data.word_to_idx), train_data.tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    test_and_write(model, test_data, UNTRAINED_OUTFILE)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in tqdm(train_data.iterate(), desc=f"Epoch {epoch}: ", total=len(train_data.data)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, train_data.word_to_idx)
            targets = torch.tensor([train_data.tag_to_idx[t] for t in tags], dtype=torch.long)

            loss = model.neg_log_likelihood(sentence_in, targets)

            loss.backward()
            optimizer.step()

    # Check predictions after training
    test_and_write(model, test_data, TRAINED_OUTFILE)