# Data Analytics BootCamp Final Project.

# Restaurant Recommedation System
We decided to implement a Restaurant Recommendation System, due Recomendation Systems discover data patterns by learning consumers preferences and search the outcomes that matches their needs and interests. There are ML algorithms that gives resuluts  by filtering useful stuff from a huge pool of information base. Recommendation Systems engines discover data patterns in the data set by learning consumers’ choices and produces the outcomes that co-relates to their needs and interests.

Imagine yo. u are in new city and you need to eat someting. If you have a friende in that city, mabybe yo will call to ypur friend for asking a place to eat in town, according toyou preferences. But, what happened if it is you do not have any friends that lived in that town. You will neeed Your personal an artificial friend: Your personal recommendations system.

This project its intended for a Traveler as an app or html page that gives restaurants or places to visit based in his preferences and  his location. We will define San Francisco, California.

View our presentation here
{google_docs}https://docs.google.com/presentation/d/1ZlSZUL6SJBcRnLjmMwqcynuWotso9JrDRmxAZ9-IRTA/edit#slide=id.p1{/google_docs}

# Descripction of the source of data.
We will us Yelp API bussiness directory. Yelp.com  website and Yelp mobile appp which publish crowd-sources reviews about bussinesses. 

# Questions we hope to answer with the data.
The questions will be: I am in San Franciscoan and I don not want to get disapointed. Where can I go to eat according with my criteria? 

# Details of the project
    We will get the initial information: *user preferences and location* from a HTLM page coding with JavaScript, CSS & Bootstrap. 
    We will take our datasets from Yelp! API  yelp_academic_dataset_bussines  and we will process and clean with Python Pandas.
    We will train our model according to the reviews of the restaurant and its location, using Unsupervised Machine Learning with K-means clustering based on the reviews or opinions of previous users. 
    We will display our recommendations results  in another HTML page.
    We will use Content Based  Filtering  it makes recommendation based on user preferences  en vez de Collaborative based filtering what requieres other users.

# Technologies
    Advanced Data Storage.
    Python API's
    JavaScript, HTML and CSS

# Tool
    Python libraries
    Scikit-learn libraries
    Plotly
    Tableau

# Mockup of the Machine Learning 

    1. Data Preparation
        a)  Data Selecction (Quality of the data: what data is avalable, what data is missing, and what data con be removed) 
        b)  Data Processing (formatting, cleaning and sampling)
        c)  Data Transformation (Transforming our data into a simpler format for future use, such as CSV, spreadsheet or Database) 

    2.  Preprocessing data with Pandas.
        a)  What knowledge do we hope to glean from running an unsupervised learning model on this dataset?
        b)  What data is available? What type? What is missing? What can be removed?
        c)  Is the data in a format that can be passed into an unsupervised learning model?
        d)  Can I quickly hand off this data for others to use?
    
    3.  Clustering data. Clustering is a type of unsupervised learning that groups data points together.
        K-means Algorithm: K-means is an unsupervised learning algorithm used to identify and solve clustering issues. K represents how many clusters are. 



        
The only way to determine what an unsupervised algorithm did with the data is to go through it manually or create visualizations. Since there will be a manual aspect, unsupervised learning is great for when you want to explore the data. May be we'll use the information provided by the unsupervised algorithm to transition to a more targeted, supervised model.


https://towardsdatascience.com/structuring-machine-learning-projects-be473775a1b6

https://towardsdatascience.com/a-basic-machine-learning-project-template-bf0ade0941d3



