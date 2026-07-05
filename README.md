# Movie_Recommendation_System_ML

so what we will do is . 

1. import the dataset
2. do the preprocessing
3. perform eda
4. create the model using semilatiry search
5. deploy the model

# Preprocessing part
1. check for null data & handle them 
2. check for duplicate values & handle them 

### EDA
1. join the columns to form the tags. with the required info
2. once we get the tag's column. do the text preprocessing. 
    # text preprocessing
        1. lower case
        2. remove punctuation
        3. remove spaces between names if any 
        4. remove stop words
        5. perform stemming/lemitizion
        6. convert the words into vectors (may use bow or TF-IDF techniques)

# create the model .
1. will the vectors do the sementic search by using cosine simpliarity . 

# create the main fxn 
by which we can send the movie name and it should give the most similar 5 movies bases on the 
similar vectors scores