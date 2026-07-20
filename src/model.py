import logging
import os
import joblib
import pandas as pd
import numpy as np 
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity  

logger=logging.getLogger("model")
logger.setLevel("DEBUG")

console_logger=logging.StreamHandler()
console_logger.setLevel("DEBUG")

folder_path="logs"
os.makedirs(folder_path,exist_ok=True)
file_logger=logging.FileHandler(os.path.join(folder_path,"model.log"))
file_logger.setLevel("DEBUG")

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_logger.setFormatter(formatter)
file_logger.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(file_logger)


def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug("data loaded successfully")
        return df
    except Exception as e :
        logger.error("error in loading data")
        raise

def perfrom_tokenization(df:pd.DataFrame,istfidf:bool):
    """this takes 2 input dataframe , tfidf==True,bew==False"""
    try:
        if istfidf:
            vectorizor=TfidfVectorizer(max_features=5000)
        else:
            vectorizor=CountVectorizer(max_features=5000)
        vectors=vectorizor.fit_transform(df["overview"])
        logger.debug("tfider vectorization technique is applied")
        return vectorizor,vectors
    except Exception as e:
        logger.error("error in the perform_tokenzation")
        raise

def save_artifact(vectorizor,vectors,file_path)->None:
    """save the vectors and fitted vectorizer to disk"""
    try:
        os.makedirs(file_path,exist_ok=True)
        joblib.dump(vectorizor,os.path.join(file_path,"vectorizor.joblib"))
        joblib.dump(vectors,os.path.join(file_path,"vectors.joblib"))
        logger.debug("vectors and vectorizer saved successfully")
    except Exception as e:
        logger.error("error in the save_artifact")
        raise

def main():
    try:
        logger.info("Feature enginerring started")
        df=load_data("./data/processed/processed.csv")
        vectorizor,vectors=perfrom_tokenization(df,True)
        save_artifact(vectorizor,vectors,"artifacts")
        logger.info("Feature enginerring ended")
    except Exception as e:
        logger.critical("error in model.py")

if __name__=="__main__":
    main()