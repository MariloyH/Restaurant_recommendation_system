# Foddiefy: Restaurant Recommedator System

Team:

Karla Plazas 

Mariloy Hernandez 

Jesus Antonio

Jesus Molina



# Presentation
## Reason Why?

Travel is a fun activity, and we always want to enjoy that time to the fullest, including the food experience. We know that there are plenty sources of information that can give us options of restaurants or places we can visit, review the location and even read reviews but that’s hard work and also time consuming. For that reason, we will develop a solution that will give the user recommendations of the best restaurant options based on ratings and comments of other visitors taking into consideration their location in order to assure that these options are not only good but also close to them. Our solution will also consider their specific preferences to make sure every detail of the experience is based on what they like and in consequence get the most of it.

<img width="660" alt="Captura de Pantalla 2022-09-27 a la(s) 14 20 43" src="https://user-images.githubusercontent.com/102195803/192878036-fcef8f2d-09c7-4dd0-9fbc-6e97f66e1e4b.png">

Another point to take in mind is who’s making the recommendations, we don´t all have a friend that lives in every city we visit, and for that reason this tool will be your local buddy, since our source of information will be obtained from data bases created by actual users, assuring no bias on our recommendations. 

<img width="330" alt="Captura de Pantalla 2022-09-27 a la(s) 14 03 53" src="https://user-images.githubusercontent.com/102195803/192878255-6281c035-4074-4d23-9f53-d27f3d27d292.png">

The Restaurant Recommendation System will discover data patterns by grouping two kinds of information:

    * Restaurant Data, by taking into account different variables like number. of ratings, number of reviews, type of cuisine and location. 
    
    * User preferences, by using the keywords of the recommendations from other users it will search and cluster restaurants that matches user interests.

These Machine Learning algorithms will cluster information from a huge pool of data, by grouping it based on patterns in the data set with the input of consumer choices and restaurant information; producing outcomes that co-relate to their needs and interests.

In conclusion this project will be a traveler solution platform to help resolve part of the traveler issues by enhancing the experience to the fullest with less effort and time investment.

<img width="660" alt="Captura de Pantalla 2022-09-27 a la(s) 14 23 48" src="https://user-images.githubusercontent.com/102195803/192878345-ec23c00b-df62-4856-af3d-0dc3fd288219.png">

# Name: Foodiefy – Recommendations from your local digital buddy 

Our plattform schema:

![diagram](/Resources/diagram2.png)

We selected first San Francisco, California which is one of the most important touristic places in the world and is also rated as one of the World’s best cities, but due we didn´t get enought data for a machine learning model, we changed to Philadelphia, Pennsilvania. 


### Presentation and Dashboard:

View our presentation here: <a href='https://docs.google.com/presentation/d/1ZlSZUL6SJBcRnLjmMwqcynuWotso9JrDRmxAZ9-IRTA/edit#slide=id.p1{/google_docs'> Recommendation System </a>

View our Dashboard in Tableau Public here:  <a href='https://public.tableau.com/app/profile/karla.plazas/viz/Foofiefy_Dashboard/Foodiefy?publish=yes'> Foodiedy Dashboard </a>

## Description of data processing

* Source of data: Yelp.com

According to Yelp site, it is platform with trusted local business information, photos and review content, that provides a one-stop local platform for consumers to discover, connect and transact with local businesses of all sizes by making it easy to request a quote, join a waitlist or make a reservation, and make an appointment or purchase.

We will use two Yelp Fusion API, which allows to get local content and user reviews from millions of businesses across 32 countries. 

* Data Preprocessing

Kaggle / Yelp Database had 5 databases from were we selected two: 

yelp_academic_dataset_business.json (businesses information)
yelp_academic_dataset_review.json (reviews from businesses)
 
This big data sets (more than 4GB) included information from distinct types of business like SPA`s, GYMS, among others (160,585 registers). It included information from several states of the US and Canada, like: Texas, Florida, Georgia, Ohio, Colorado, Oregon, Vancouver and Pennsylvania.

We chose Pennsylvania more specifically Philadelphia since it had one of the most robust data sets 

From Pennsylvania dataset we used Pyspark to clean the data base filtering only information from Philadelphia and obtain only restaurants as businesses from both databases. From reviews data set we used only use relevant columns of information that allowed us to build the data frame, actually we made the same selection pricess for Business data base. Finally we filtered unique restaurants and obtained the 1,000 restaurants with the higher number of reviews (popularity)

*Feature Engineering and selection

We used 2 Unsupervised Machine Learning Models to solve this  problem: One for the restaurant location and other for the recommendations.

1. First we take the user’s position and number of reviews to search the nearest restaurants around him. For that location restaurant recommendation we selected from all the data set: Lat, Long, No. of reviews, stars, name and Business ID. 

2. With that list, we use a NLP matrix to analyze the reviews that  matches the user’s preferences. For recommendation based on user Phrase, we imported the clean data base (cleaned on database creation), then cleaned reviews words by removeing punctuation and stop words so we obtained one string per user review

3,Finally, we show a map with the best selection.

* Data Split for training and testing

Data for training and testing on our ML model was mainly selected from the word processing. It’s important to take in mind that we are using unsupervised ML algorithms to cluster filter and deliver a recommendation, for that reason test and learn is only a reference.



