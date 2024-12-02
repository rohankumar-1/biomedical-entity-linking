
import os
import re
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize

BTAG = "B--"
ITAG = "I--"

def ner_get_preprocessed_data(dataset="train"):
    """ 
    Returns parsed, cleaned dataset as list of sentences across all drugs, and also returns IOB taglines for each sentence
    """
    
    files = os.listdir(f"data/tac_2017/{dataset}/")
    sentset = []
    tagset = []
    
    for fp in files:
        _, texts, mentions, _ = parse_xml_file(f"data/tac_2017/{dataset}/{fp}")
        cleaned_texts = clean_text(texts)
        full_text, tags = build_iob_map(cleaned_texts, mentions)
        sentset.extend(full_text)
        tagset.extend(tags)
    
    return sentset, tagset



def parse_xml_file(fp: str) -> tuple[str, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """ returns name, text, mentions, and final reactions (labels) for a drug's XML file passed in as fp"""
    root = ET.parse(fp).getroot()
    texts = [{"text": tt.text,
              "section": tt.get("id")} for tt in root.findall("./Text/Section")]
    mentions = [{"str": tt.get("str"), 
                 "section": tt.get("section"),
                 "type": tt.get("type"),
                 "len": tt.get("len")} for tt in root.findall("./Mentions/Mention")]
    
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
    
    return root.get("drug"), texts, mentions, reactions

def clean_text(texts: list[dict[str, str]]) -> list[dict[str, list[list[str]]]]:
    """ convert each text into a list of lists of sentences -> words"""
    result = []
    
    def bioclean(t):
        return re.sub("[.,?;*!%^&_+():-\\[\\]{}]", "", t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower()).split()
    
    for section in texts:
        cleaned_section = {"section": section["section"], "sentences": []}
        for line in section["text"].split("\n"):
            if len(line)>2:
                line = bioclean(line)
                cleaned_section["sentences"].append(
                    line
                )

        result.append(cleaned_section)
            
    return result


def build_iob_map(texts, mentions):
    """ given texts and mentions, create IOB formatted info """
    full_text, tags = [], []
    # looping through each section
    for i, section in enumerate(texts):

        # get all mentions in the current section
        idx = f"S{i+1}"
        relevant_mentions = {m["str"]: m["type"] for m in mentions if m["section"]==idx}

        for term, type in relevant_mentions.items():
            full_types = []
            for i, word in enumerate(term.split()):
                if i==0:
                    full_types.append(f"{BTAG}{type}")
                else:
                    full_types.append(f"{ITAG}{type}")

            relevant_mentions[term] = " ".join(full_types)
        
        # loop through each mention, and for that mention, tag the word as B-TYPE, I-TYPE, etc
        for line in section["sentences"]:
            full_line = " ".join(line)
            tag_line = full_line
            for term, type in relevant_mentions.items():
                # assert isinstance(term, str)
                # assert isinstance(type, str)
    
                # build regex patterns
                pattern = "{term}[ ,:;]".format(
                    term=re.escape(term)
                    )
                
                repl = "{type} ".format(
                    type=type
                    )
                
                # replace 
                tag_line = re.sub(pattern=pattern, repl=repl, string=tag_line, count=10) # tag_line.replace(term+" ", type+" ")
                # print(tag_line)
            if len(full_line) != 0:
                full_text.append(full_line.split())
                tags.append(tag_line.split())

    for i in range(len(tags)):
        for j in range(len(tags[i])):
            if not tags[i][j].startswith((BTAG, ITAG)):
                tags[i][j] = "O"
            
    # we want to return a full concatenated text, and a full list of tags, one tag for each word in the text
    return full_text, tags
            






###########################################################
##
##         Functions for baseline model below
##
###########################################################

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

                label_list[start_index] = f"{BTAG}ENTITY"
                for i in range(start_index + 1, end_index):
                    label_list[i] = f"{ITAG}ENTITY"
        
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
        _, text, mentions, reactions = parse_xml_file_baseline(f"data/tac_2017/{dataset}/{fp}")
        X.append(text)
        mentions_list.append(mentions)
        reactions_list.append(reactions)
    
    return X, mentions_list, reactions_list



def get_tac_recognition_test():
    raise NotImplementedError


    
def parse_xml_file_baseline(fp: str) -> tuple[str, str, list[str], list[dict[str, str]]]:
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
