import logging
import numpy as np 
import pandas as pd
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

logger=logging.getLogger("recommend")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")
log_path="logs"
os.makedirs(log_path,exist_ok=True)
file_handler=logging.FileHandler(os.path.join(log_path,"recommend.log"))
file_handler.setLevel("DEBUG")

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug("data loaded successfully")
        return df
    except Exception as e:
        logger.error("error in loading the data")
        raise e

def load_artifact(folder_path:str,vectorizer_filename,vector_filename:str):
    try:
        vectorizer=joblib.load(os.path.join(folder_path,vectorizer_filename))
        vector=joblib.load(os.path.join(folder_path,vector_filename))
        logger.debug("fetch the artifacts successfully")
        return vectorizer,vector
    except Exception as e:
        logger.error("error in loading the artifacts")
        raise e

def recommend_movie(movie_name:str,vectorizer,vector,df:pd.DataFrame)->list:
    try:
        movie_index= df[df["title"]==movie_name].index[0]
        similarity=cosine_similarity(vector[movie_index],vector)
        arr_dict=list(enumerate(similarity[0])) #with arr_dict we have preserved the index as well as there score .now we can sort this 
        # but  we have to sort in to the 2nd column of similariry score 
        movie_with_highest_similarity=sorted(arr_dict,reverse=True,key=lambda x:x[1])
        top_5=movie_with_highest_similarity[1:6]
        movie_info=[]
        for item in top_5:
            movie_info.append(df.iloc[item[0]])
            print(df.iloc[item[0]]["title"])
        return movie_info
    except Exception as e:
        logger.error("error in the main_function")
        raise

def main():
    try:
        logger.info("recommending started")
        df=load_data("./data/processed/processed.csv")
        vectorizer,vector=load_artifact("./artifacts","vectorizor.joblib","vectors.joblib")
        result=recommend_movie("Spider-Man 3",vectorizer,vector,df)
        logger.info("recommending ended")
    except Exception as e:
        logger.error("error in main function")

if __name__ =="__main__":
    main()