# Data Analytics BootCamp Final Project.

Presentation

## Selected Topic

Restaurant Recommendation System

## Reason Why?

Travel is a fun activity, and we always want to enjoy that time to the fullest, including the food experience. We know that there are plenty sources of information that can give us options of restaurants or places we can visit, review the location and even read reviews but that’s hard work and also time consuming. For that reason, we will develop a solution that will give the user recommendations of the best restaurant options based on ratings and comments of other visitors taking into consideration their location in order to assure that these options are not only good but also close to them. Our solution will also consider their specific preferences to make sure every detail of the experience is based on what they like and in consequence get the most of it.

Another point to take in mind is who’s making the recommendations, we don´t all have a friend that lives in every city we visit, and for that reason this tool will be your local buddy, since our source of information will be obtained from data bases created by actual users, assuring no bias on our recommendations. 

The Restaurant Recommendation System will discover data patterns by grouping two kinds of information:

    * Restaurant Data, by taking into account different variables like number. of ratings, number of reviews, type of cuisine and location. 
    
    * User preferences, by using the keywords of the recommendations from other users it will search and cluster restaurants that matches user interests.

These Machine Learning algorithms will cluster information from a huge pool of data, by grouping it based on patterns in the data set with the input of consumer choices and restaurant information; producing outcomes that co-relate to their needs and interests.

In conclusion this project will be a traveler solution platform to help resolve part of the traveler issues by enhancing the experience to the fullest with less effort and time investment.

# Name: Foodiefy – Recommendations from your local digital buddy 

Our plattform schema:

![diagram](/Resources/diagram2.png)

We selected San Francisco, California which is one of the most important touristic places in the world and is also rated as one of the World’s best cities.

* More than 4,300 restaurants
* Tourism: On 2022 it will receive around 21 million visitors

### Presentation:

View our presentation here: <a href='https://docs.google.com/presentation/d/1ZlSZUL6SJBcRnLjmMwqcynuWotso9JrDRmxAZ9-IRTA/edit#slide=id.p1{/google_docs'> Recommendation System </a>

## Description of the source of data

Yelp.com

According to Yelp site, it is platform with trusted local business information, photos and review content, that provides a one-stop local platform for consumers to discover, connect and transact with local businesses of all sizes by making it easy to request a quote, join a waitlist or make a reservation, and make an appointment or purchase.

We will use two Yelp Fusion API, which allows to get local content and user reviews from millions of businesses across 32 countries. 

Databases Used:

* Business Search
* Reviews

EDA:

* Number of ratings:

![rating](/Resources/rating.png)

* Top Eleven Restaurants:

![Topeleven](/Resources/topeleven.png)

* Data Type

![dtype](/Resources/DTYPE.png)

* Null Data

![null](/Resources/NULL.png)

* Clean Duplicates

![dup](/Resources/Screen%20Shot%202022-08-20%20at%2013.36.41.png)

* San Francisco Restaurants Locations:

![SFRest](/Resources/locations.png)

* Clean Data Frame

![SFRest](/Resources/Cleandf.png)

## Questions to answer

1.	Can I obtain a recommendation of the best Restaurants near my location? (Lat & Long)?
2.	Can that recommendation be “real” and not have a bias from maybe the own restaurants?
3.	Can that recommendation be streamlined based on important preferences to me?

# GitHub Repository

*Readme with description:

![Readme](/Resources/readme.png)

* Individual Branches

![Branches](/Resources/branches.png)

# Machine Learning Model

* Takes in data from the provisional database

![ProvDB](/Resources/provdata.png)

* Outputs label for input data

* Elbow Curve:

![Elbow](/Resources/Elbow.png)

* Data Clasification

![Clasif](/Resources/Clasif.png)

*Scatter Plots w & w/o Clusters and 3D:

![Plotcluster](/Resources/scatter1.png)
![Plot3d](/Resources/scatter3d.png)

# Database Integration

* Sample data that mimics the expected final database structure or schema

![DBDF](/Resources/prediction.png)

* Draft machine learning model is connected to the provisional database

![READCSV](/Resources/connection.png)

* Example

![Example](/Resources/mapprueba.png)

# Other details of the project

* Technologies
    Python API's
    

* Tools
    Python libraries
    Scikit-learn libraries
    Plotly
