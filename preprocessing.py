
import os

import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize

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


# def process_plain_text(text: list[str]) -> str:
#     data = ""
#     for t in text:
#         data += word_tokenize(t)
#     return data


if __name__=="__main__":
    name, text, mentions, reactions = parse_tac_xml("data/tac_2017/train/train_xml/ADCETRIS.xml")
    
    # print(process_plain_text(text))
    
    