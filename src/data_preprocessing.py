import os
import logging
import pandas as pd
import numpy as np 
import re
import nltk
nltk.download("punkt_tab")
nltk.download("stopwords")
from nltk.corpus import stopwords   #to get all the stopwords
from nltk.tokenize import word_tokenize #to token all the words 
from nltk.stem import PorterStemmer #for perfrom stemming/lemization
import json
import joblib


# make the log dir if not exist
log_folder_name="logs"
os.makedirs(log_folder_name,exist_ok=True)

# define the logger
logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")
# console_ handler
console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")
# file_handler
# create a file for it 

file_handler=logging.FileHandler(os.path.join(log_folder_name,"data_preprocessing.log"))
file_handler.setLevel("DEBUG")
# formatter
log_formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# attaching formatter to console_handler && file_handler
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

# now attach both the handler to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# convert all the data into lower case
def _lower_case(text:str)->str:
    return str.lower(text)

# remove !@#$%^&*() from the text
def _remove_punctuations(text:str)->str:
    return re.sub(r"[^A-Za-z0-9\s]","",text)

# remove stop words
def _remove_stop_words(text:str)->str:
    final_str=[]
    words=stopwords.words("english")
    tokens=word_tokenize(text)
    for token in tokens:
        if token in words:
            continue
        else:
            final_str.append(token)
    return " ".join(final_str)

# perform lemitization
def _perfrom_lemitization(text:str)->str:
    ps=PorterStemmer()
    final_str=[]
    tokens=word_tokenize(text)
    for token in tokens:
        stemed_word=ps.stem(token)
        final_str.append(stemed_word)
    return " ".join(final_str)
    
# keep only keywords or we can say tags
def _keywords_process(text):
    keywords=[]
    arr=json.loads(text)
    for item in arr:
        keywords.append(item["name"].replace(" ",""))
    return keywords

# keep only top 3 actory names
def _cast_process(text):
    final_cast=[]
    arr=json.loads(text)
    count=1
    for item in arr:
        if count <=3:
            name=item["name"]
            name_without_space=name.replace(" ","")
            final_cast.append(name_without_space)
            count+=1
        else:
            break
    return final_cast

# keep only the director name in the list 
def _crew_process(text):
    director_name=[]
    arr=json.loads(text)
    for item in arr:
        if item["job"]=="Director":
            director_name.append(item["name"].replace(" ",""))
            break
    return director_name

# convert list to str
def _convert_list_to_str(given_list):
    return " ".join(given_list)



def load_data(path:str)->pd.DataFrame:
    """load the data from csv"""
    try:
        logger.debug("started of load_Data")
        df=pd.read_csv(path)
        logger.debug("end of load_Data")
        return df
    except Exception as e :
        logger.error("error in loading csv data")
        raise

def data_processing(df:pd.DataFrame)->pd.DataFrame:
    """fxn where columsn are processed"""
    try:
        logger.debug("starting of data_processing fxn")
        df["keywords"]=df["keywords"].apply(_keywords_process)
        df["cast"]=df["cast"].apply(_cast_process)
        df["crew"]=df["crew"].apply(_crew_process)
        logger.debug("column transformation is done")
        df["summary"]=df["keywords"]+df["cast"]+df["crew"]
        logger.debug("all the columsn has been merged into summary columns")
        # convert the summary columns into str  (curretnly it is is list)
        df["summary"]=df["summary"].apply(_convert_list_to_str)
        df["overview"]=df["overview"]+" "+df["summary"]
        df=df[["id","title","overview"]]
        logger.debug("final columns has been choosen")
        logger.debug("ending of data_processing fxn")
        return df
    except Exception as e:
        logger.error("error in the data_processing fxn")
        raise

def text_processing(df:pd.DataFrame)->pd.DataFrame:
    "do all the requried processing on the textual columns"
    try:
        logger.debug("starting of text_processsing fxn")
        df["overview"]=df["overview"].apply(_lower_case)
        df["overview"]=df["overview"].apply(_remove_punctuations)
        df["overview"]=df["overview"].apply(_remove_stop_words)
        df["overview"]=df["overview"].apply(_perfrom_lemitization)
        logger.debug("ending of text_processsing fxn")
        return df
    except Exception as e :
        logger.error("error in the transfrom_text_data")
        raise

def save_data(df:pd.DataFrame,file_path:str,file_name:str)->None:
    try:
        os.makedirs(file_path,exist_ok=True)
        df.to_csv(os.path.join(file_path,file_name))
        logging.debug("csv has been saved successfully")
    except Exception as e:
        logger.error("error in saving data")
        raise

def main():
    try:
        logger.info("START of data_preprocessing pipeline")
        df=load_data("./Data/raw/raw_df.csv")
        df=data_processing(df)
        df=text_processing(df)
        save_data(df,"./data/processed","processed.csv")
        logger.info("END of data_preprocessing pipeline")
    except Exception as e:
        logger.error("error in data_preprocessing pipeline")
        
if __name__=="__main__":
    main()