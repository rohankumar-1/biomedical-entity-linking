
# Information on Project


![](CS505_Poster__NEW__page-0001.jpg)


## For Instructors

There are two top-level files: train_ner.py and train_nl.py. These are the files that train and evaluate the models for NER and linking, respectively. Models can be found in the models/ subdirectory, and the evaluation script can be found in the evaluation/ subdirectory.

There are two directories that are not included, but are necessary for replication. The first is a subdirectory called pretrained/, which stores the BioWordVec model bin (I exclude this because it is 20GB). The model can be found at https://github.com/ncbi-nlp/BioSentVec. The second is a subdirectory called data/, which houses all of the TAC-ADR-2017 data, as well as the MedDRA excel sheet. 

The structure is the following:

```
data/
|__ MEDDRA.xlsx
|__ tac_2017
    |__ test
        |__ drug1.xml
        |__ drug2.xml
        |__ ...
    |__ train
        |__ drug1.xml
        |__ drug2.xml
        |__ ...
```


