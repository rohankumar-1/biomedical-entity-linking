
import os
import re
from tqdm import tqdm

import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize


def preprocess_dataset(texts: list[str], entity_lists: list[list[str]]):
    """ turn text into list of dictionaries extracted features """
    
    features = []
    labels = []
    
    for text, entities in zip(texts, entity_lists):
        tokens = word_tokenize(text)
        
        label_list = ["O"] * len(tokens)

        # Mark the entities in the label list
        for entity in entities:
            entity_tokens = word_tokenize(entity)
            pattern = r'\b' + r'\s+'.join(re.escape(token) for token in entity_tokens) + r'\b'
            for match in re.finditer(pattern, text):
                start, end = match.span()
                start_index = len(re.findall(r'\S+', text[:start]))
                end_index = start_index + len(entity_tokens)

                label_list[start_index] = "B-ENTITY"
                for i in range(start_index + 1, end_index):
                    label_list[i] = "I-ENTITY"
        
        labels.extend(label_list)
        for tk in tokens:
            features.append(
                {
                    'word': tk,
                    'is_capitalized': int(tk[0].isupper()),
                    'length': len(tk),
                }   
            )

    return features, labels
    

def get_tac(dataset="train"):
    """ returns X, Y for training set """
    
    files = os.listdir(f"data/tac_2017/{dataset}/")
    X, mentions_list, reactions_list = [], [], []
    
    for fp in files:
        _, text, mentions, reactions = parse_xml_file(f"data/tac_2017/{dataset}/{fp}")
        X.append(text)
        mentions_list.append(mentions)
        reactions_list.append(reactions)
    
    return X, mentions_list, reactions_list



def get_tac_recognition_test():
    raise NotImplementedError


    
def parse_xml_file(fp: str) -> tuple[str, str, list[str], list[dict[str, str]]]:
    """ returns name, text, mentions, and final reactions (labels) for a drug's XML file passed in as fp"""
    root = ET.parse(fp).getroot()
    text = "\n".join([tt.text for tt in root.findall("./Text/Section")])
    mentions = [tt.get("str") for tt in root.findall("./Mentions/Mention")]
    
    reactions = list()
    for tag_type in root.findall("./Reactions/Reaction"):
        norm = tag_type.find("Normalization")
        reactions.append(
            {
                "name": tag_type.get("str"),
                "id": norm.get("id"),
                "meddra_pt": norm.get("meddra_pt"),
                "meddra_pt_id": norm.get("meddra_pt_id")
            }
        )
    
    return root.get("drug"), text, mentions, reactions
