import logging
import sys
import os
import pandas as pd

# write the code such that if a log folder doeesn't exist it gets created. 
log_folder_name="logs"
os.makedirs(log_folder_name,exist_ok=True)

# create a logger
# logging configuration's

logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")


# create a console handler
console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

# create a file handler
file_handler_log_file_path=os.path.join(log_folder_name,"data_ingestion.log")
file_handler=logging.FileHandler(file_handler_log_file_path)
file_handler.setLevel("DEBUG")

# create a formatter
log_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# set the format of file_handler and streamhandler
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)


# add both the handler to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# load the data
def load_data(movies_url:str,credit_url:str)->dict:
    """ load the data """
    try:    
        logger.debug("inside the load_data fxn")
        
        total_df={}
        logger.debug("inside the loop")
        movies_df=pd.read_csv(movies_url)
        credit_df=pd.read_csv(credit_url)
        logger.debug("df's has been loaded successfully ")
        total_df.update({"movies_df":movies_df,"credit_df":credit_df})
        
        logger.debug("the csv has been added to dict")
        return total_df
    except Exception as e:
        logger.error("error occuered in the loop")
        raise e

def preprocess_data(total_df:dict)->pd.DataFrame:
    """takes a list of df's and merge them """
    try:
        logger.debug("inside the preprocessing_data fxn")
        movies_df=total_df["movies_df"]
        credit_df=total_df["credit_df"]

        movies_df=movies_df[["id","keywords","overview"]]
        merged_df=movies_df.merge(credit_df,left_on="id",right_on="movie_id")
        # check for null & duplicate & if exist drop them 
        merged_df=merged_df.dropna()
        merged_df=merged_df.drop_duplicates()
        logger.debug("completed the prePrcess")
        return merged_df

    except Exception as e:
        logger.error("error in preprocessing_data fxn")
        raise

def save_data(merged_df:pd.DataFrame)->None:
    """save the merged_df to a csv file """
    try:
        logger.debug("started the save_data fxn")
        # make a raw dir if not exist
        # os.makedirs("Data",exist_ok=True)
        os.makedirs("./data/raw",exist_ok=True)
        merged_df.to_csv(os.path.join("./Data/raw","raw_df.csv"),index=False)
        logger.debug("the df has been saved to data/raw/raw_df.csv")
    except Exception as e:
        logger.error("error occured in save_data fxn")
        raise

def main(movie_link:str,credit_link:str):
    try:
        logger.info("START of data_ingestion")
        total_df=load_data(movie_link,credit_link)
        merged_df=preprocess_data(total_df)
        save_data(merged_df)
        logger.info("END of data_ingestion")
    except Exception as e :
        logger.error("failed to compelte the data ingestion pipline")


if __name__ =="__main__":
    main("https://raw.githubusercontent.com/deepanshu5628/Movie_Recommendation_System_ML/refs/heads/main/Data/tmdb_5000_movies.csv","https://raw.githubusercontent.com/deepanshu5628/Movie_Recommendation_System_ML/refs/heads/main/Data/tmdb_5000_credits.csv")