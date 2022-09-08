#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import hvplot.pandas

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Dependencies
import nltk
nltk.download ('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import WordPunctTokenizer


import string


# In[8]:


import warnings
warnings.filterwarnings('ignore')


# In[9]:


file_path = "Data/Philadelphia_businesses.csv"
df_restaurants = pd.read_csv(file_path)


# In[10]:


top_restaurats_Philly = df_restaurants.sort_values(by=['stars_business','review_count'],ascending=False)


# In[11]:


def recommend_restaurants (latitude, longitude):  
    top_restaurants_df = df_restaurants.sort_values(by=["stars_business","review_count"], ascending=False)[:20]
    
    #Elbow method to determine the number of K in Kmeans Clustering
    coords = df_restaurants[["latitude", "longitude"]]

    inertia = []
    K = range(1,25)

    # Calculate the inertia for the range of K values
    for k in K:
        kmeansModel = KMeans(n_clusters=k)
        kmeansModel = kmeansModel.fit(coords)
        inertia.append(kmeansModel.inertia_)
    kmeans = KMeans(n_clusters= 5, init='k-means++')
    kmeans.fit(coords)
    y = kmeans.labels_   
    
    df_restaurants["cluster"]= kmeans.predict(df_restaurants[['latitude', 'longitude']])
    
    top_restaurants_Philly = df_restaurants.sort_values(by=['stars_business','review_count'],ascending=False)
     
    df = top_restaurants_Philly 
    
    # Predict the cluster for longitude and laltiude provided
    cluster = kmeans.predict(np.array([latitude, longitude]).reshape(1, -1))[0]
    #Get the best reataurant in this cluster
    
    return df[df["cluster"]==cluster].iloc[0:50][['name', 'latitude', 'longitude','stars_business','categories','review_count','ID']]


# # Importing Clean Data

# In[12]:


file_path = "Data/Final_philadelphia_reviews.csv"
phillies_df = pd.read_csv(file_path)
yelp_reviews_df = phillies_df[['review_id', 'user_id', 'business_id', 'text', 
                               'stars_business', 'review_count']]


# In[13]:


# Fill with empty string the NaN reviews
yelp_reviews_df.dropna(inplace=True)
yelp_reviews_df[['text']] = yelp_reviews_df[['text']].fillna('')


# In[14]:


yelp_reviews_df.rename (columns={'review_id': 'Review_ID', 'user_id' :'User_Id', 
                        'business_id':'Business_Id', 'text':'Reviews', 
                'stars_business': 'Rating', 'review_count' :'Review_count'}, inplace=True)


# # Begin the reviews cleaning, selecting only stars and text

# In[15]:


#Select only stars and text
reviews_df = yelp_reviews_df[['Business_Id', 'User_Id', 'Rating', 'Reviews']]
reviews_df["Reviews"] = yelp_reviews_df["Reviews"].str.replace(";", " ")


# In[16]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    
   # Check characters to see if they are in punctuation          
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join t('stop_word(he characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return " ".join([word for word in nopunc.split() if word.lower() not in stop])


# In[17]:


from nltk.corpus import stopwords
stop = []

for word in stopwords.words('english'):
    s = [char for char in word if char not in string.punctuation]
    stop.append(''.join(s))


# In[18]:


reviews_df['Reviews'] = reviews_df['Reviews'].apply(text_process)
#Split train test for testing the model later
vld_size=0.15
X_train, X_valid, y_train, y_valid = train_test_split(reviews_df['Reviews'], yelp_reviews_df['Business_Id'], test_size = vld_size) 


# # Create two tables of user, text and bussiness

# In[19]:



def vectorizeReviews (rest_df):
    newreviews_df = reviews_df[reviews_df['Business_Id'].isin(rest_df["Business_Id"])]

    userid_df = newreviews_df[['User_Id','Reviews']]
    business_df = newreviews_df[['Business_Id', 'Reviews']]

    userid_df = userid_df.groupby('User_Id').agg({'Reviews': ' '.join})
    business_df = business_df.groupby('Business_Id').agg({'Reviews': ' '.join})

    # User Tfdf Vectorizer  
    userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)
    userid_vectors = userid_vectorizer.fit_transform(userid_df['Reviews'])
    
    #Business id vectorizer
    businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)
    businessid_vectors = businessid_vectorizer.fit_transform(business_df['Reviews'])
    
    #Matrix Factorization
    userid_rating_matrix = pd.pivot_table(newreviews_df, values='Rating', index=['User_Id'], columns=['Business_Id'])
    P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())
    Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names()) 
    P, Q = matrix_factorization(userid_rating_matrix, P, Q, steps=25, gamma=0.001,lamda=0.02)
    return(P,Q, userid_vectorizer)


# # Gradient Decent Optimization

# In[20]:



def matrix_factorization(R, P, Q, steps=25, gamma=0.001,lamda=0.02):
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        
    return P,Q


# In[21]:


# Store P, Q and vectorizer in pickle file
import pickle
#output = open('yelp_recommendation_model_8.pkl', 'wb')
#pickle.dump(P,output)
#pickle.dump(Q,output)
#pickle.dump(userid_vectorizer,output)
#output.close()


# # Run prediction according User's preference

# In[22]:


def findRecommendations (latitude, longitude, words):

    recommendedlist = []
    
    #Call the first part of ML process.. finding 50 nearest restaurants 
    restaurants_df = recommend_restaurants(latitude,longitude)
    restaurants_df.rename (columns={'ID':'Business_Id'}, inplace=True)
    
    #Process the user preferences
    test_df= pd.DataFrame([words], columns=['Reviews'])
    test_df['Reviews'] = test_df['Reviews'].apply(text_process)
    
    #Call the second part of the ML process...     
    P, Q, userid_vectorizer = vectorizeReviews (restaurants_df) 
    test_vectors = userid_vectorizer.transform(test_df['Reviews'])
    test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

    predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
    foundRestaurants=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:7]

    for i in foundRestaurants.index:
        name = restaurants_df[restaurants_df['Business_Id']==i]['name'].iloc[0]
        categories =restaurants_df[restaurants_df['Business_Id']==i]['categories'].iloc[0]
        latitude = restaurants_df[restaurants_df['Business_Id']==i]['latitude'].iloc[0]
        longitude = restaurants_df[restaurants_df['Business_Id']==i]['longitude'].iloc[0]
        rating = str(restaurants_df[restaurants_df['Business_Id']==i]['stars_business'].iloc[0])

        case = {'Name': name, 'Categories': categories, 'Latitude': latitude, 'Longitude': longitude, 'Rating' : rating}
        recommendedlist.append(case)
        
    topRecommend_df = pd.DataFrame (recommendedlist)
    Data2geojson(topRecommend_df)        
    return(True)   


# In[23]:


import json
import geojson
from geojson import Feature, FeatureCollection, Point


def Data2geojson(df):
    features = []
    insert_features = lambda X: features.append(
                    geojson.Feature(geometry=geojson.Point((X["Longitude"],
                                                    X["Latitude"])),
                    properties=dict(name = X["Name"],
                                    description = X["Categories"],
                                    rating = X['Rating']))
                    )
    df.apply(insert_features, axis=1)
    with open('Templates/top7recommendations.geojson', 'w', encoding='utf8') as fp:
        geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True, ensure_ascii=False,indent=4)  
   


# In[24]:


#findRecommendations(39.951732, -75.142140, "spaguetti music")


# In[ ]:




