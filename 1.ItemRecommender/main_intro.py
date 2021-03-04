#!/usr/bin/env python
# coding: utf-8

# # Recommending Top-N movies - Part 1: Introduction, timings and use cases

# The MovieLens dataset, made by GroupLens, contains movies, tags, genres, and user ratings from real users, contributed voluntarily and gathered over various timeframes. GroupLens offer various version of the data set; this project uses the full 27M dataset which is open and free for educational and research purposes.
# 
# The aim of this project is to use the movie information and user ratings to build recommender systems using different approaches and algorithms. The algorithms are built with a focus on top-N recommendations and an end user application similar to Netflix where movies are shown on pages containing 5 movies per page. The algorithms will be able to take single movies, many movies, or a user's past ratings to achieve this.
# 
# The project comes in two parts. This first part gives an overview of the dataset and the algorithms, how the algorithms work, their use cases, and the various methods present in the objects used to access them contained in the associated python files. Part 2 takes a deeper dive into testing, comparing the accuracy of the algorithms using train-test-validation splits of the dataset, with a look at how the parameters coded into the algorithm affect the metrics used to test them. For further information on how each class works, see the documentation contained in MovieLensData.py, OneMovieCF.py, Algorithms.py, SplitData.py, and Tester.py.

# ------

# ## Importing and analysing the dataset

# First, we import all the packages that will be used, and count the total amounts of data present in the set.

# In[1]:


import warnings
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

import Algorithms
import MovieLensData
import OneMovieCF

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:20,.6f}'.format
sns.set()


# In[2]:


ml = MovieLensData.MovieLensData()

totalUsers = len(ml.ratingsRaw['userId'].unique())
totalRatings = len(ml.ratingsRaw['userId'])
totalMovies = len(ml.moviesRaw['movieId'].unique())
totalMoviesRated = len(ml.ratingsRaw['movieId'].unique())

print("Total # of users:", totalUsers)
print("Total # of ratings:", totalRatings)
print("Total # of movies logged:", totalMovies)
print("Total # of movies rated:", totalMoviesRated)


# Let's take a look at the structure of the two datasets we've imported.

# In[3]:


display(ml.moviesRaw.head(5))
display(ml.ratingsRaw.head(5))


# All 58098 movies and 283228 users have a unique integer ID associated with them. The titles and years for each movie are listed as strings. Genres are listed as pipe delimiter separated strings of which there are a total of 19 genres plus '(no genre listed)'.
# 
# The ratings from each user are listed by user ID and movie ID and a float representing their rating on a 0.5-5.0 scale in increments of 0.5. The timestamp column represents seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
# 
# The full dataset contains ratings for over 280K users with 27.7M ratings in total. These users have rated just under 54K movies, leaving just over 4K movies logged in the dataset with no ratings. Below is a plot the spread of ratings in the data.

# In[4]:


ratingCount = ml.ratingsRaw['rating'].value_counts().sort_index()

plt.figure(figsize=(8, 4))
ax = sns.barplot(x=ratingCount.index, y=ratingCount.values)
ax.set(xlabel='Rating', ylabel='# of Ratings', title='Number of ratings by rating')

totalPossibleRatings = len(ml.ratingsRaw['userId'].unique()) * len(ml.moviesRaw['movieId'].unique())
unrated = totalPossibleRatings - len(ml.ratingsRaw['userId'])
print("Total # of unrated movies for every user:", unrated)


# The most common ratings in order are 4.0, 3.0, and 5.0, indicating that users have a bias toward picking whole numbers for their ratings. We can also see that users are far less likely to give movies a 'perfect' 5.0 rating, preferring to give less-than-perfect 4.0 or 3.0 ratings. We can also see that users are very unlikely to rate movies anything below 3.0 showing a bias towards higher ratings.
# 
# Bias in user ratings is something we will have to consider as one user's 4.0 is not the same as another's i.e., 4.0 may be the highest rating for a harsh critic, but 4.0 could be a relatively low rating for a lenient critic. By plotting user rating means, we can get a sense of these biases present in the ratings.

# In[5]:


means = ml.ratingsRaw.drop(['timestamp', 'movieId'], axis=1).groupby('userId').mean().squeeze().rename('mean')

plt.figure(figsize=(8, 4))
ax = sns.distplot(means, bins=75, kde=False)
ax.set(xlabel='Mean Rating', ylabel='Frequency', title='Count of mean ratings per user')
print("Mean rating for all users:", ml.ratingsRaw['rating'].mean())


# The mean rating for all users is around 3.5 with this plot of 75 bins, showing a mostly normal distribution peaking around 3.5-4.0. The key thing we see is the vast difference of mean ratings per each user. If user A, with a 2.0 mean rating, was found by an algorithm to be similar to user B, with a 4.0 mean rating, we could not simply just assign ratings from user B to user A without considering their mean ratings in some way. It will therefore be crucial to implement a consideration of rating bias in our recommender.
# 
# However, these outlier counts may just be from users with very low movie rating counts. We can plot movie counts against mean ratings to see how movie rating count affects their perceived bias.

# In[6]:


counts = means.reindex(ml.ratingsRaw['userId'].value_counts().rename('Len-Means'))

plt.figure(figsize=(8, 4))
ax = sns.distplot(counts, bins=75, kde=False)
ax.set(xlabel='Mean Rating', ylabel='# of rated movies', title='Amount of rated movies per mean user rating')
plt.show()


# This graph shows mean ratings with each bar now weighted for the sum of the movies rated by all users with that mean rating. We see a clear preference for 4.0 and 3.5 ratings, with these being the average rating for some of the higher rating volume users. The outlier at 5.0 shows a small portion of users whose ratings consist solely of 5.0 ratings. This is also most likely true for those with mean ratings at 1.0 due to the low amount of data surrounding it. Disregarding these anomalies and other low count mean ratings, this leaves most of our higher volume users having biased ratings within the 2.5-4.5 range.
# 
# To gain a better understanding of the spread of user rating counts, we can plot the number of ratings per user. The graph follows a deeply exponential curve so it is therefore better visualised on a log graph.

# In[7]:


userRatingCount = ml.ratingsRaw['userId'].value_counts().reset_index(drop=True)

plt.figure(figsize=(8, 4))
ax = sns.lineplot(x=userRatingCount.index, y=userRatingCount.values)
ax.set(xlabel='User Rating Ranking', ylabel='# of Ratings', title='Number of ratings by user', yscale='log')
plt.show()


# We can see a very small number of users (~<5000) with rating sizes between 1000 and 15000 ratings. These users will have great sway on our predictions, statically skewing recommendations to the movies they have rated highly. Due to this, it may be wise to disregard users with high sizes of ratings to stop their biases skewing recommendations.
# 
# We can also see a great number of users with very low rating sizes of ~<20. These users will be very hard to recommend movies for as there is not much data to go on. Such users may get recommendations widely unrelated to their tastes. Similarly, it may be that these users just have very niche tastes for films with very low ratings themselves. To check whether this is true, we can also look at the sizes of ratings per movie.

# In[8]:


movieRatingCount = ml.ratingsRaw['movieId'].value_counts().reset_index(drop=True)

plt.figure(figsize=(8, 4))
ax = sns.lineplot(x=movieRatingCount.index, y=movieRatingCount.values)
ax.set(xlabel='Movie Popularity', ylabel='# of Ratings', title='Number of ratings by movie', yscale='log')

unratedMovies = totalMovies - len(ml.ratingsRaw['movieId'].unique())
print("Total # of unrated movies:", unratedMovies)


# Over half of all rated movies have ~<10 ratings, showing a great deal of very niche movies in the database. Such movies will usually be missed by basic recommenders due to lack of correlation with other items. This 'long-tail' of items is a common problem in these algorithms with companies aiming to recommend as much of this content as possible, as this can be where the most profitable items lie due to profit margins. However, recommending too many of these items will lead to user distrust in the system, so it is a delicate balance to consider. We will measure this using the novelty metric in part 2 of this project.
# 
# Due to the large amount of data, lack of sophisticated systems available for this project, and potential issues that can arise from the issues discussed above, I will reduce the data for this project. While it will help the quality of recommendations for the purposes of testing, it goes without saying that real world data is, by nature, very sparse and as such, sophisticated recommenders should find ways to take both of these groups into consideration.
# 
# I will sample users with rating counts between 200 and 6000 to avoid biases from outliers. Similarly, I will limit the movies to those with more than 100 ratings.

# In[9]:


userIDs = ml.filterIDs('userId', minRatings=200, maxRatings=6000)
movieIDs = ml.filterIDs('movieId', minRatings=100)
ml.reduce(userIDs, 'userId', 'ratings')
ml.reduce(movieIDs, 'movieId', 'movies')
ratingsData = ml.buildPivot(printStats=True)


# ## Collaborative Filtering
# 
# - Similarities based on ratings, genre and year for single movie.
# - Combining the above for better predictions.
# - Generating correlation matrices for ratings, genre and year for all movies and combining.
# - Generating correlation matrix for all users based on ratings and returning similar movies based on the ratings of the most similar users.

# Collaborative Filtering algorithms aim to calculate the similarity or distance between items or users, and recommend based on a generated similarity matrix. Many methods can be used, such as simple matching coefficient, Jaccard index, or Euclidian distance. This algorithm will use cosine similarity, which represents all items/users as multi-dimensional vectors and calculates the angle between these vectors. It does this by taking the dot product of the two vectors divided by the product of the magnitude of the two vectors. Applying this to all items and users respectively will give us two matrices from which we can calculate similarities to an item or user.
# 
# The first step toward doing this is to build a pivot table of items against users (with missing data filled with nan) from the reduced ratings set built earlier.

# In[10]:


sparsity = ratingsData.isnull().sum() * 100 / len(ratingsData)
print(round(sparsity.sum() / len(sparsity), 2), "% sparsity", sep='')
ratingsData


# The sparsity is the percentage of nan values present in the data frame. A sparsity of around 95.5% is mostly normal, even a bit low, for these system with this kind of real-world data. With this table built, we can now start to recommend some movies.

# ### Single movie recommendations
# 
# - Correlating ratings using cosine similarity
# - Correlating genres using cosine similarity
# - Comparing release years using an exponential function
# - Combining the above and limiting our results

# To recommend movies based on a single movie, we can correlate this movie with the ratings in the pivot table to get the most similar movies to this one according to the cosine similarity formula. 
# 
# The OneMovieCF class initialises with a MovieLensData object and the information for the movie which we wish to recommend movies for. This initialisation returns the movieID for this movie by searching the moviesRaw data frame in the MovieLensData object.
# 
# The getSimilarMovies method then applies cosine similarity the pivot table by calling the corrwith function. This provides a series of movieIDs and their similarity to the correlating movieID. It then gets info about the movies from the moviesRaw data frame contained in the ml object including title, year, popularity etc. and presents this as a data frame.

# In[11]:


movie = ["Lord of the Rings", 2001]
oneMovieCF = OneMovieCF.OneMovieCF(movie, ml)

ratings = oneMovieCF.getSimilarRatings()
ratings.head(10)


# As you may expect for 'The Fellowship of the Ring', two of the movies in the top 10 are simply the sequels showing that the correlation has worked to some degree. However, many of the films deemed most similar to this movie are far less expected. It could indicate that most users who like 'The Fellowship of the Ring' also have very niche tastes as most of these films have very low rating sizes of <100 whereas the two 'Lord of the Rings' sequels have rating sizes close to 20K. However, it's far more likely that simply correlating movies doesn't take into account the validity of the correlations i.e., how low amounts of data available for some movies negatively skew the similarity scores.  The genres of the films recommended further indicate this as all are either comedy or drama, whereas we may associate 'The Fellowship of the Ring' far more with the Adventure and Fantasy genres.
# 
# Given this, it may be worth considering how the genres of movies are correlated. To do this, we will need to represent these genres in a way the correlation function can understand. One way to achieve this is to represent the movies as a bit field. For example, if the database had only 4 genres (Action, Adventure, Drama and Fantasy), then 'The Return of the King' would have a bit field of 1111, whereas 'The Two Towers' would be 0101, and 'Wooden Man's Bride' would be 0010. By splitting up the string given in the genres column of the moviesRaw data frame contained in the ml object, we can build a bit field for all movies in the dataset. The buildGenreBitfield method from the ml object performs this.

# In[12]:


ml.buildGenreBitfield()


# With the 19-bit bit field constructed, we can now perform the cosine similarity function, where each movie is now a 19-dimensional item to correlate. By accessing the bit field for 'The Fellowship of the Ring' and correlating this with all other movies in the bit field, we can generate similarities based on genre alone. The getSimilarGenres method from the OneMovieCF object does this.

# In[13]:


print('The Fellowship of the Ring has genres:', ml.getInfo(4993, get='genres')[0])
genres = oneMovieCF.getSimilarGenres()
genres


# As you would expect, the most similar movies have the exact same genre as the 'The Fellowship of the Ring' showing the correlation to have worked, but there are still a few movies with very low rating sizes that seem disconnected. We do see the least similar movies have genres completely different to the test movie, however all movies in the top 10 have the exact same similarity making it hard to tell the order in which movies are most similar to our test movie. 
# 
# We can also see a vast difference in years with some movies dating as far back as the 1940s. It may be highly unlikely that users who like modern adventure films will like movies released 60 years prior therefore, it could also be wise to consider the correlation between the years movies were released. To do this, we can calculate the absolute integer difference between release years and applying a function to this value to normalise it for the 0-1 range to keep it in line with our previous correlation scores. While we could apply a simple linear function to achieve this, we would ideally like movies released closer to the test movie to correlate far more highly than those released decades away. This can be achieved with an exponential function of negative power, in this case e^(-x/7) where x is the difference and e is Euler's number. The getSimilarYears method performs this for our test movie.

# In[14]:


print('The Fellowship of the Ring was released in', ml.getInfo(4993, get='year')[0])
years = oneMovieCF.getSimilarYears()
years


# All movies in the top 10 were released in the same year as the test movie, with movies with the least similarity released nearly 100 years before the test movie, showing the function to have performed as expected. However, the most similar movies are just a random selection of movies released in 2001, with no consideration of genre or rating similarity to the test movie making this algorithm alone not a very good method for recommending movies.
# 
# Each of these three methods have their merits, but alone they wouldn't be sufficient for a very trust-worthy recommendation algorithm. But we can combine these methods to generate a list of movies from the best elements of each method. This can be achieved simply by multiplying each of their similarities together due to the fact that all similarities are normalised between 0-1. The getAll method achieves this.

# In[15]:


top = oneMovieCF.getAll()
top.head(10)


# Many of these movies are good recommendations; we see the two sequels right at the top, the Harry Potter films and Pirates of the Caribbean. All these movies could be considered good recommendations for 'The Fellowship of the Ring'. However, some lesser-known films are still creeping into the top 10. One last fine-tune we can do is to limit the list to only those of a certain popularity or rating size.

# In[16]:


top[top['rating#'] > 500].head(10)


# By limiting the movies to only those with more than 500 ratings, we get a fairly decent set of movie recommendations. All films seem believable and varied enough to recommend to a user who likes 'The Fellowship of the Ring'. However, it's worth noting that just because a movie has a low rating count does not mean it's niche. Such movies may have only been released recently and as such, not have had the same time to receive ratings as some older films have had. Therefore these newer movies will not have the same amount of data as movies like 'The Fellowship of the Ring' to produce accurate similarity scores. This is called the 'cold start' problem and is a common issue in recommender systems.
# 
# This dataset was generated on the 26/09/18 with ratings taken in various timeframes since 1995. If we were to run the same method on a movie released in 2018 while limiting the final results by rating size, this may cause it to show more dis-similar movies than we might hope for.
# 
# While this method does work, it would be very slow to do this on a movie-by-movie basis for a fast-loading recommendation system like Netflix. As such, a better approach would be to pre-generate similarity scores and store them for fast reference later when loading a page. We will look at this next.

# ### Item-based Collaborative Filtering
# 
# - All item similarity matrices using cosine similarity
# - User-item CF
# - Runtime comparisons
# - Predicting ratings
# - Limitations

# Rather than correlating a given movie to the dataset in real time every time, we can instead generate a correlation matrix of all items using the corr function. This is the basis of Item-based Collaborative Filtering (CF). The code works in much the same way as before: generating cosine similarity scores for all items based on their ratings and genres, generating similarity scores by years by calculation with an exponential function, and multiplying the values for each item together to generate the similarity matrix. The buildModel(modelType='CF') method from the Algorithms class will build all of these for both items and users.

# In[17]:


algo = Algorithms.Algorithms(ml)
CF_itemModel, CF_userModel = algo.buildModel(modelType='CF', printStatus=True)
CF_itemModel


# Here, we can see movie IDs along both rows and columns and how similar to every other movie in the reduced dataset. While this takes much longer to create initially, this will greatly reduce processing time when we look at recommending movies for a user. All we have to do to get similar movies to any movie now is to simply select the row or column of that movieID. This makes this a memory-based algorithm as all similarities are stored in memory for later accessing.
# 
# For this, we can user the getSimilarMovies method on 'The Fellowship of the Ring' once again.

# In[18]:


oneMovieCF.similarRatings = None
oneMovieCF.similarGenres = None
oneMovieCF.similarYears = None
oneMovieCF.similarAll = None

t_start = perf_counter()
oneMovieCF.getAll()
print('\33[94;1m', "Time taken for one movie CF model: ", (perf_counter() - t_start), 's', '\33[0m', sep='')

t_start = perf_counter()
similarMovies = algo.getSimilarMovies(4993, CF_itemModel, 'CF', buildTable=True)
print('\33[94;1m', "Time taken for full item-based CF model: ", (perf_counter() - t_start), 's', '\33[0m', sep='')
similarMovies[:10]


# The movies returned are almost identical to the previous algorithm, with minor differences in similarity causing a different order to the ranking. The time taken for the single movie correlation object to perform a recommendation is close to 11s, whereas this new item-based CF model recommends in a milli-seconds, showing a great improvement on performance. We can now use this matrix to get user recommendations based upon the movies they have already rated. 
# 
# First, we'll add some movie ratings to the ratingsPivot data frame stored in the algo object. This user loves fantasy and action movies, but hates older musical movies.

# In[19]:


mainRatings = {60684: 4, 77561: 3.5, 98809: 5, 165469: 1, 180263: 2, 914: 0.5, 899: 0.5, 85788: 4.5, 102445: 4.5,
               48774: 5}
algo.addTestUser(mainRatings)
algo.getUserRatings(0, buildTable=True)


# The itemBased method picks out all the movies the user has rated from the item-based CF matrix generated earlier, and weights each of the similarities by multiplying each of the sets of similarities by the rating the user has given for that movie.
# 
# This method has two key parameters, neighbours and sample. The sample value limits the itemBased method to only looking at the top number of rated movies given by this parameter. The neighbours parameter is considered once passed to the getPredictedRatings method which sums similarities and predicts ratings from the weighted similarity matrix. Here, the neighbours value limits the ratings the amount of most similar movies to consider when summing similarities. It sums these limited similarities to produce a list of most similar movies to the user. 
# 
# To get the predicted ratings however, a sum of all unweighted similar movies is passed to the getPredictedRatings method with each weighted movie’s sum divided by this unweighted sum to provide some rating predictions. Finally, the user's bias is also considered for the predicted ratings by calculating the user's deviation from the mean of all ratings, and adding this to the rating predictions. To ensure this bias consideration doesn't cause the ratings to go above the maximum possible rating of 5, a limiting factor is place on the ratings. 
# 
# Both series are then passed to the buildTable function to add the extra information and display.

# In[20]:


algo.itemBased(0, CF_itemModel, 'CF', buildTable=True, neighbours=100000, printStatus=True)[:10]


# Given the makeup of our test user, these recommendations seem sensible. We can check to see the impact our algorithm has on recommendations by reversing this user's ratings and re-running.

# In[21]:


oppositeRatings = {movie: 5.5 - ratings for movie, ratings in mainRatings.items()}
algo.addTestUser(oppositeRatings)
display(algo.getUserRatings(0, buildTable=True))
algo.itemBased(0, CF_itemModel, 'CF', neighbours=20, threshold=0, buildTable=True, printStatus=True)[:10]


# Some of these recommendations seem to make sense, however it seems the recommender is struggling to make suggestions for this inverted user. We see some more classic musical movies in our top 10, but it's still dominated by similar action films to those seen before, despite the algorithm thinking it will rate these movies as low as roughly 1.0. This is most likely due to this inverted user having only rated 2 musical number highly, with lots of low ranks for action films from which the algorithm is pulling information to fill the gaps despite their low ratings.
# 
# To combat this, a minimum threshold rating can be passed to the itemBased method to limit the ratings it considers. 

# In[22]:


algo.itemBased(0, CF_itemModel, 'CF', neighbours=20, threshold=2.0, buildTable=True, printStatus=True)[:10]


# Much better, though due to the fact the algorithm is only considering the two movies rated as 5.0, all our ratings our now 5.0. Issues of low rating predictions in the top recommended items can arise from users with very low rating counts, such as this test user, as the algorithm does not have enough data to gauge this users real interests.

# ### User-based Collaborative Filtering
# 
# - User-user CF
# - Runtime comparisons
# - Predicting ratings
# - Limitations

# We can also correlate this user directly with other users by weighting each similar users’ ratings to their similarity to the test user. Once each user is weighted, we also have to consider the similar users rating biases, calculating their deviation from the overall ratings mean and subtracting this from their similarity to the test user. Once this is done, the similarity matrix is passed to the getPredictedRatings to sum similarities and predict ratings as before.
# 
# Here, we're getting recommendations for our original test user who loves fantasy and action movies, but hates older musical movies.

# In[23]:


algo.addTestUser(mainRatings)
algo.userBased(0, CF_userModel, 'CF', threshold=0, buildTable=True, printStatus=True)[:10]


# As the algorithm now has to correlate this user with other users in real time, the runtime has now increased by a few orders of magnitude making this method less useful for fast access to recommendations. We can also see how different these movies are compared to the item-based recommender, with far fewer novel films appearing. A recommender like this may most likely not be great for a company as it doesn't access items in the long-tail, where the largest profit margins exist.
# 
# Most of these films are very popular films showing the algorithm is mostly just pulling the most popular movies from the similar user's ratings. To see the impact a user with more ratings has, let's test user 81.

# In[24]:


userID = 81
print('User', userID, 'has rated', len(algo.getUserRatings(userID)), 'movies')
algo.userBased(userID, CF_userModel, 'CF', threshold=2, buildTable=True, printStatus=True)[:10]


# Here we see much less popular films appearing, however these films are still far more popular than those recommended by the item-based algorithm for the test user. This highlights the unpredictability of relying on user-based recommender system due to the vast differences in tastes user can have. Just because two users happen to rate movies similar, does not necessarily mean they have very similar tastes. Such algorithms are useful to consider, but should be tested rigorously as will happen in the next part of this project.

# ## K-Nearest Neighbours

# K-Nearest Neighbours (KNN), like Collaborative Filtering, is a supervised machine learning algorithm, meaning it relies on labelled data to produce results. The labelled data in this case are the movies and their ratings for item-based methods, or users and their ratings for user-based methods. At a high-level, KNN works by assuming that similar things exist close to one another. In a basic 2-dimensional problem, one could imagine this on a scatter graph with various clear clusters of points. A KNN algorithm would attempt to identify the clusters, and label each point as being associated with that cluster. In contrast to CF, this is a model-based approach as we're querying a model to calculate similarities every time, rather than having all these values readily available in memory like in CF.
# 
# KNN algorithms come in two broad flavours, classification and regression. In classification problems, our data would be a clear binary of choices i.e., a movie was watched or not, whereas a regression problem contains a distribution of values to factor i.e., ratings for the watched movie. Hence we'll need a regression-based algorithm for our recommender.
# 
# Implementing all this from the ground up can be incredibly complex, so I'll be using sklearn's neighbours.NearestNeighbours. This algorithm simply takes a set of data and maps it out in such a way that we can pull the distance of its nearest k neighbours. These distances are our similarity, but are measured from 0 (the movie or user to pull similarities for) up to 1 (the furthest from the test item or user). We will therefore need to subtract these values to get a comparable similarity score to the CF algorithm above.
# 
# While we will be demonstrating the KNN algorithm, it's worth noting that a KNN algorithm is ill-advised for large datasets. This is due to the time it takes the algorithm to calculate distances in larger and sparse datasets. With sufficient resources, KNN could be a useful tool for a recommender system. However, we will see the runtime this algorithm has in comparison to CF.

# ### Item-Based KNN
# 
# - Building a sparse matrix
# - Storing index-ID dictionaries
# - Building the item-based model
# - Runtime and limitations

# One thing we can do to improve runtime is to create a sparse matrix from our ratings pivot data using SciPy's csr_matrix. This throws away any nan data present and just records the index positions of ratings. However, we cannot store the specific userIDs and movieIDs in the matrix and have to make a separate reference for which indices in the matrix are associated with what ID. The buildMatrix method accomplishes this.

# In[25]:


ratingsSparse = algo.buildMatrix()
print(ratingsSparse[0, 0:20])
ratingsSparse


# As we can see, the csr_matrix has dropped the indices relating to movies this user has not rated. This will speed up the KNN algorithm, but now we need to implement a way of switching between these indices and the IDs by using dictionaries. Let's now build our KNN model from this matrix and pull some recommendations for 'The Fellowship of the Ring'.

# In[26]:


KNN_itemModel = algo.buildModel(modelType='KNN', matrix=ratingsSparse.transpose(), printStatus=True)

t_start = perf_counter()
similarMovies = algo.getSimilarMovies(4993, KNN_itemModel, 'KNN', neighbours=10, buildTable=True)
print('\33[94;1m', "Time taken for single movie KNN model: ", (perf_counter() - t_start), 's', '\33[0m', sep='')
similarMovies[:10]


# Slightly different, but still appropriate recommendations for our test movie. We can also see a 4x improvement on timings over the single item CF model. However, the issue of runtime comes with scalability as we'll see shortly. Let's check our regular test user once again.

# In[27]:


algo.addTestUser(mainRatings)
algo.itemBased(0, KNN_itemModel, 'KNN', neighbours=100, buildTable=True, printStatus=True)[:10]


# Again, another set of good-looking recommendations for this user. We can now start to see some of the performance impacts with the KNN model, arising from the constant time it takes to look up the similarities for every movie the user has rated one by one. For this system, that figure is around 0.35s, and with user 0 having rated 8 movies, it takes just over 2s to pull recommendations. Let's see how the item-based KNN model takes a user with a greater number of ratings.

# In[28]:


userID = 81
print('User', userID, 'has rated', len(algo.getUserRatings(userID)), 'movies')
algo.itemBased(userID, KNN_itemModel, 'KNN', neighbours=100, buildTable=True, printStatus=True)[:10]


# Here we can truly see the impracticalities of using KNN on a large data set. The algorithm follows O(n) complexity, increasing runtime linearly as the number of movies the user has rated, n, increases. However, we can improve timings by invoking the sample parameter once again which limits the amount of the movies the algorithm considers from the movies the user has rated. Such tests will be run in part 2.

# ### User-Based KNN
# 
# - Building the user-based model
# - Runtime and appropriateness

# One way the KNN algorithm still performs admirably is with user-based recommendations. As the algorithm only has to pull one set of similarities (the k most similar users), the runtime is greatly improved and less affected by the number of movies the test user has rated. Again, we get distances of 0-1 which must be inverted, biases and sums considered, with all this passed on to the getPredictedRatings method to produce our final recommendations.

# In[29]:


KNN_userModel = algo.buildModel(modelType='KNN', matrix=ratingsSparse)
algo.userBased(81, KNN_userModel, 'KNN', neighbours=100, buildTable=True, printStatus=True)[:10]


# Much faster than item-based KNN. Speed is not the most important consideration though and we will be testing the accuracy of these algorithms in the next part of this project.

# ## SVD 

# One last method we can consider is SVD. SVD is a matrix decomposition method, which separates the matrix into its constituent parts to make calculations on it simpler. The matrices produced are given by the formula, A = U . E . V^T, where A is the matrix to decompose of dimensions m x n (the ratings pivot data frame in our case), U is an m x m matrix, E is a diagonal matrix of m x n, and V^T is the transposed n x n matrix. 
# 
# By finding the dot product of these 3 matrics, we can produce a similarity matrix that can be used to get similarities for a user very quickly due the SVD algorithm effectively filling in the missing values in our ratings pivot.
# 
# However, the drawback is this matrix must be recalculated every time a user adds a new rating, or a new user is added making it a slow model for initialisation and updates.

# In[30]:


algo.addTestUser(mainRatings)
SVD_model = algo.buildModel(modelType='SVD', printStatus=True)


# In[31]:


algo.SVD(0, SVD_model, sample=10, buildTable=True, printStatus=True)[:10]


# While the algorithm is very fast, we see that it also suffers from issues with low rating counts, giving unexpected recommendations due to being unable to draw much conclusion from the limited data. Let's see how it handles a user with more data available.

# In[32]:


algo.SVD(81, SVD_model, sample=10, buildTable=True, printStatus=True)[:10]


# Still fast, and the algorithm has better confidence in the recommendations it has given. Still, we won't know how good these are until we test them.

# ## Summary

# Here we've looked at the MovieLens dataset and the distributions of data within it, making decisions for how best to proceed with building recommender algorithms based upon it. We've looked at 3 algorithms we can use, how they work, some of their quirks and drawbacks, with a brief look at their runtimes. 
# 
# Next we'll dive deeper into testing these algorithms using train-test split data and parameter testing to fine tune each algorithm.
