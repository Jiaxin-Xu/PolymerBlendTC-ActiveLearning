import numpy as np 
import pandas as pd
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from rdkit import Chem

def read_blend_data_PE(filename='../polymer_blend_combined_data.csv',npb=213):
    pb_v0 = pd.read_csv(filename)[:npb]
    polymer_embedding_model = word2vec.Word2Vec.load('../POLYINFO_PI1M.pkl')
    # Representation 1 -- Polymer Embedding
    sentences1 = []
    sentences2 = []
    for i in range(len(pb_v0)):
        sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(pb_v0["SMILES1"].iloc[i]), 1))
        sentences1.append(sentence)
        sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(pb_v0["SMILES2"].iloc[i]), 1))
        sentences2.append(sentence)
    
    polymer_embeddings1 = [DfVec(x) for x in sentences2vec(sentences1, polymer_embedding_model, unseen='UNK')]
    polymer_embeddings2 = [DfVec(x) for x in sentences2vec(sentences2, polymer_embedding_model, unseen='UNK')]

    X1_PE = np.array([x.vec.tolist() for x in polymer_embeddings1])
    X2_PE = np.array([x.vec.tolist() for x in polymer_embeddings2])
    X_ws_PE = X1_PE * pb_v0["weight"][:,np.newaxis]/6 + X2_PE * (6-pb_v0["weight"])[:,np.newaxis]/6

    y = np.array(pb_v0["TC_mean"]).astype(np.float)
    label = np.array(pb_v0["outperform"])
    print("Read ",filename,'\n','Good PB ratio: ',sum(label), '/', len(label), round(sum(label)/len(label)*100, 2), '%')

    return X_ws_PE, y, label