import random
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

from MovieLensData import MovieLensData


def removeMultiLevel(ratingsPivot):
    """
    Checks for MultiLevel columns in ratingsPivot and removes them.

    :param ratingsPivot: DataFrame. Pivoted DF of columns containing MultiLevel Index with 'title' and 'year' axes.
    :return: ratingsPivot: DataFrame. Pivoted DF with adjusted columns.
    """
    if isinstance(ratingsPivot.columns[0], tuple):
        print("\nRemoving MultiLevel index from pivot df...", end='')
        ratingsPivot = ratingsPivot.droplevel(['title', 'year'], axis=1)  # drop extra indices
        print('\33[92m', "Done", '\33[0m')
    return ratingsPivot


class BuildCF:
    """
    Builds Collaborative Filtering cosine similarity matrices from ratingsPivot DataFrame.

    :attribute corrUsers: DataFrame. DF of rating correlation scores between every user.
    :attribute corrMovieRatings: DataFrame. DF of rating correlation scores between every item.
    :attribute corrGenres: DataFrame. DF of genre correlation scores between every item.
    :attribute corrYears: DataFrame. DF of year correlation scores between every item.
    :attribute corrItems: DataFrame. DF of combination of rating, genres and year correlation scores between every item.
    
    :attribute ml: object. MovieLensData object for function calls.
    :attribute ratingsPivot: DataFrame. Pivoted DF of- columns: movieId, index: userId, values: ratings.

    :methods:
    buildSimilarUsers(update=True, printStatus=False):
        Builds cosine similarity correlation DF of ratings between all users.

    buildSimilarMovieRatings(update=True, printStatus=False):
        Builds cosine similarity correlation DF of ratings between all items.

    buildGenres(update=True, printStatus=False):
        Builds cosine similarity correlation DF of genres between all items.

    buildYears(update=True, printStatus=False):
        Builds cosine similarity correlation DF of years between all items.

    buildAll(get=None, printStatus=False):
        Runs all the functions in the object if self variables are None and combines item rating, genre and year
        correlations.
    """

    # variables for storing correlation data frames
    corrUsers = None
    corrMovieRatings = None
    corrGenres = None
    corrYears = None
    corrItems = None

    def __init__(self, ml=None, trainData=None):
        """
        Initialises attributes, checks for ratingsPivot present, and removes MultiLevel columns if present.

        :param ml: object, default=None. MovieLensData object for function calls
        :param trainData: DataFrame, default=None. Pivoted DF of columns: movieId, index: userId, values: ratings.
                                                    Must have nan values filled with zeros.
        """
        self.ml = MovieLensData() if ml is None else ml  # create new object from MovieLensData class if not given

        if trainData is not None:  # if training data given, remove multilevel column index and fill nan values with 0
            self.ratingsPivot = removeMultiLevel(trainData).fillna(0)
        elif self.ml.ratingsPivot is not None:  # if training data not given, use internally stored pivot table from ml
            # check for multilevel column index and fill nan values with 0
            self.ratingsPivot = removeMultiLevel(self.ml.ratingsPivot).fillna(0)
        else:  # create quick diagnostic pivot table if no table given or found
            print('\33[91m', "Error: Training data not given and ratingsPivot not found.", '\33[0m', sep='')
            print("Building quick pivot df with MinRatings of 1000 for Users and Movies.")
            self.ratingsPivot = self.ml.buildPivot(extraColumns=False, quick=True, update=True).fillna(0)

    def buildSimilarUsers(self, update=True, printStatus=False):
        """
        Builds cosine similarity correlation DF of ratings between all users.

        :param update: bool, default=True. Update self.corrUsers with generated DF
        :param printStatus:  bool, default=False. Print progress.
        :return: DataFrame. Cosine similarity correlation dataframe for user ratings
        """

        print("Correlating ratings for all users...", end='') if printStatus else None
        # convert ratings pivot to numpy array for faster calculation
        # calculate cosine similarity for ratings of all users
        # rebuild data frame from array with userID indices
        corrUsers = pd.DataFrame(np.corrcoef(self.ratingsPivot.to_numpy(), rowvar=True),
                                 index=self.ratingsPivot.index, columns=self.ratingsPivot.index).rename_axis(None)
        print('\33[92m', "Done", '\33[0m') if printStatus else None

        if update:  # update internally stored variable for user correlations
            self.corrUsers = corrUsers
        return corrUsers

    def buildSimilarMovieRatings(self, update=True, printStatus=False):
        """
         Builds cosine similarity correlation DF of ratings between all items.

        :param update: bool, default=True. Update self.corrMovieRatings with generated DF
        :param printStatus:  bool, default=False. Print progress.
        :return: DataFrame. Cosine similarity correlation dataframe for item ratings
        """

        print("Correlating ratings for all movies...", end='') if printStatus else None
        # convert ratings pivot to numpy array for faster calculation
        # calculate cosine similarity for ratings of all movies
        # rebuild data frame from array with movieID indices
        corrMovieRatings = pd.DataFrame(np.corrcoef(self.ratingsPivot.to_numpy(), rowvar=False),
                                        index=self.ratingsPivot.columns, columns=self.ratingsPivot.columns)
        print('\33[92m', "Done", '\33[0m') if printStatus else None

        if update:  # update internally stored variable for movie rating correlations
            self.corrMovieRatings = corrMovieRatings
        return corrMovieRatings

    def buildGenres(self, update=True, printStatus=False):
        """
        Builds cosine similarity correlation DF of genres between all items.

        :param update: bool, default=True. Update self.corrGenres with generated DF
        :param printStatus:  bool, default=False. Print progress.
        :return: DataFrame. Cosine similarity correlation dataframe for item genres
        """
        # build genre bitfield if not already stored in ml object
        allGenres = self.ml.buildGenreBitfield() if self.ml.genreBitfield is None else self.ml.genreBitfield.copy()

        print("Correlating genres for all movies...", end='') if printStatus else None
        # convert ratings pivot to numpy array for faster calculation
        # calculate cosine similarity for genres of all movies
        # rebuild data frame from array with movieID indices
        corrGenres = pd.DataFrame(np.corrcoef(allGenres.to_numpy(), rowvar=True),
                                  index=allGenres.index, columns=allGenres.index)
        print('\33[92m', "Done", '\33[0m') if printStatus else None

        if update:  # update internally stored variable for genre correlations
            self.corrGenres = corrGenres
        return corrGenres

    def buildYears(self, update=True, printStatus=False):
        """
        Builds cosine similarity correlation DF of years between all items.

        :param update: bool, default=True. Update self.corrYears with generated DF
        :param printStatus:  bool, default=False. Print progress.
        :return: DataFrame. Cosine similarity correlation dataframe for item years.
        """
        print("Correlating years for all movies...", end='') if printStatus else None
        # check for reduced data and use raw data if not found
        movies = self.ml.moviesReduced.copy() if self.ml.moviesReduced is not None else self.ml.moviesRaw.copy()
        # get year strings for movies and recast as integers
        allYears = movies.set_index('movieId')['year'].dropna().astype(int)

        # calculate differences in all years and store as numpy array
        diff = np.abs(np.subtract.outer(list(allYears), list(allYears)))
        # calculate similarities according to the exponential function
        similarity = np.exp(-diff / 7.0)

        # build data frame from array with movieID indices
        corrYears = pd.DataFrame(similarity, index=list(allYears.index), columns=list(allYears.index))
        print('\33[92m', "Done", '\33[0m') if printStatus else None

        if update:  # update internally stored variable for year correlations
            self.corrYears = corrYears
        return corrYears

    def buildAll(self, get=None, printStatus=False):
        """
        Runs all the functions in the object if self variables are None and combines item rating, genre and year
        correlations.

        :param get: str, default=None. Define which DataFrame to build and return 'item' or 'user'. None returns both
        :param printStatus:  bool, default=False. Print progress.
        :return: DataFrames. Returns DataFrames defined in get kwarg.
        """
        # build similarities according to get parameter if not already done
        if self.corrUsers is None and get != 'item':
            self.buildSimilarUsers(printStatus=printStatus)

        if self.corrMovieRatings is None and get != 'user':
            self.buildSimilarMovieRatings(printStatus=printStatus)

        if self.corrGenres is None and get != 'user':
            self.buildGenres(printStatus=printStatus)

        if self.corrYears is None and get != 'user':
            self.buildYears(printStatus=printStatus)

        if get != 'user':  # if only item-based requested, multiply similarities together generating combined similarity
            print("Generating combined correlation...", end='') if printStatus else None
            self.corrItems = (self.corrMovieRatings * self.corrGenres * self.corrYears).rename_axis(None, axis=1)
            self.corrItems.dropna(how='all', axis=0, inplace=True)  # drop users with no similarity values
            print('\33[92m', "Done", '\33[0m') if printStatus else None

        # return data frames according to get parameter (item-, user-based or both)
        if get is None:
            return self.corrItems, self.corrUsers
        elif get == 'item':
            return self.corrItems
        elif get == 'user':
            return self.corrUsers
        else:
            print('\33[91m', "Error: 'get' not item or user", '\33[0m', sep='')
            return


class Algorithms:
    """
    Builds algorithm models and generates movie recommendations from these models.
    
    :attribute ratingsMean: DataFrame. DF of movieID, title, year, popularity, ratingSize, ratingMean and genre.
    :attribute userPivot_csr: SciPySparseMatrix. Sparse Matrix of ratingsPivot. User focused. Made movie focused by 
    :attribute userID_lookup: dict. All {userPivot_csr index: userID} pairs
    :attribute movieID_lookup: dict. All {transposed userPivot_csr index: movieID} pairs
    :attribute movieIndex_lookup: dict. {movieID: index} pairs of movieIDs to transposed userPivot_csr indices
    
    :attribute ml: object. MovieLensData object for function calls.
    :attribute moviesInfo: object. Raw data loaded from movies.csv. Columns: movieId, titles+years and genres.
    :attribute trainData: DataFrame. Pivoted trainSet DF of- columns: movieId, index: userId, values: ratings.
    :attribute testData: DataFrame. Pivoted testSet DF of- columns: movieId, index: userId, values: ratings.
    :attribute mean: float. Mean value of all ratings in self.trainData.
    :attribute trainData_filled: DataFrame. Copy of self.trainData with nan values filled with zeros
    
    :methods:
    getInfo(movieID, get=None):
        Returns info as defined by get kwarg from moviesInfo for input movieIDs.

    getMovieID(movie):
        Searches self.moviesInfo for movieID from input movie info arg.

    getUserRatings(userID, data=None, drop=True, buildTable=False):
        Returns all ratings for given userID. Searches either ratingsPivot DF if given in data kwarg, or self.testData.

    addTestUser(ratings):
        Adds ratings to self.testData for userID 0.

    buildMeanRatings(update=True):
        Builds DF of movieID, title, year, popularity, ratingSize, ratingMean and genre for all movies in self.trainData

    buildTable(similarMovies=None, ratingPredictions=None, predict='calc'):
        Adds similarMovies and ratingPredictions series to the ratingsMean DF and sorts by descending similarity.

    buildMatrix(update=True):
        Builds User oriented SciPy CSR SparseMatrix from self.trainData and generates dictionaries for looking up
        IDs from indices and vice versa. Stores these in self variables.
        The CSR matrix can be transposed using [MATRIX].transpose() for item-based modelling.

    buildModel(modelType, matrix=None, printStatus=False):
        Builds models from given matrix.

    getSimilarMovies(movieID, model, modelType, neighbours=100, buildTable=False):
        Gets similar movies from input movieID based on input model.

    getSimilarUsers(userID, model, modelType, neighbours=100):
        Gets similar users from input userID based on input model.

    getPredictedRatings(similarities, userRatings, sums, sample=100):
        Amalgamates similarities together for given user's ratings and predicts ratings for this user.

    itemBased(userID, model, modelType, neighbours=100, sample=None, predict='calc', threshold=2.0,
                buildTable=False, printStatus=False):
        Driver function for ItemBased algorithms. Gets similar items for given user ID based on ItemBased models.

    userBased(userID, model, modelType, neighbours=100, sample=None, predict='calc', threshold=2.0,
                buildTable=False, printStatus=False):
        Driver function for UserBased algorithms. Gets similar items for given user ID based on UserBased models.

    SVD(userID, model, sample=100, predict='calc', buildTable=False, printStatus=False):
        Uses SVD to return similarities and predicted ratings for a given user from a similarity matrix.

    random(_, __, randomRatings=True):
        Generates random recommendations and ratings.
    """

    ratingsMean = None  # data frame of mean ratings, rating sizes, popularity and extra info

    userPivot_csr = None  # compressed sparse row matrix. apply .transpose() for moviePivot
    userID_lookup = None  # find userID from csr index
    movieID_lookup = None  # find movieID from csr index
    movieIndex_lookup = None  # find csr index from movieID

    def __init__(self, ml=None, trainData=None, testData=None, maxRating=5.0):
        """
        Initialises attributes, checks for ratingsPivot present, and removes MultiLevel columns if present.

        :param ml: object, default=None. MovieLensData object for function calls.
        :param trainData: DataFrame, default=None. Pivoted trainSet DF of columns:movieId, index:userId, values:ratings
        :param testData: DataFrame, default=None. Pivoted testSet DF of columns:movieId, index:userId, values:ratings
        :param maxRating: int, default=5.0.
        """
        print("Initialising object...", end='')
        self.ml = MovieLensData() if ml is None else ml  # create new object from MovieLensData class if not given
        self.moviesInfo = self.ml.moviesRaw
        self.maxRating = maxRating

        if trainData is not None:  # if training data given, remove multilevel column index if present
            self.trainData = removeMultiLevel(trainData)
        elif self.ml.ratingsPivot is not None:  # if training data not given, use internally stored pivot table from ml
            # remove multilevel column index if present
            self.trainData = removeMultiLevel(self.ml.ratingsPivot)
        else:  # create quick diagnostic pivot table if no table given or found
            print('\33[91m', "\nError: Training data not given and ratingsPivot not found.", '\33[0m', sep='')
            print("Building quick pivot df with MinRatings of 1000 for Users and Movies.")
            self.trainData = self.ml.buildPivot(quick=True)

        # if no test data given, use training data as test data
        self.testData = testData if testData is not None else self.trainData
        # generate a pivot table copy with nan values filled with 0
        self.trainData_filled = self.trainData.fillna(0)

        # calculate mean rating of all rating values
        self.mean = self.trainData.mean().mean() / self.maxRating
        # calculate each users bias from the total mean
        self.biasFactor = (self.trainData.mean(axis=1) / self.maxRating) - self.mean

        print('\33[92m', "Done", '\33[0m')

    def getInfo(self, movieID, get=None):
        """
        Returns info as defined by get kwarg from moviesInfo for input movieIDs.

        :param movieID: int or list. MovieIDs to search for in self.moviesInfo
        :param get: str, default=None. What info to return. Must be either 'title', 'year', or 'genres'
        :return: list. List of strings of all info found for given IDs
        """
        # method only works if movieIDs are in list form; check if ID given as int and convert to list
        movieID = [movieID] if isinstance(movieID, int) else movieID
        # check for movieID in info data and return df for all IDs found
        movies = self.moviesInfo[self.moviesInfo['movieId'].isin(movieID)].set_index('movieId')
        movies = movies.reindex(movieID)

        if get == 'title':  # returns movie title
            ls = [i for i in movies['title']]
        elif get == 'year':  # returns movie year
            ls = [i for i in movies['year']]
        elif get == 'genres':  # returns movie genres as pipe-delimited string
            ls = [i for i in movies['genres']]
        else:  # if no get property defined, return df of all information
            return movies.set_index('movieId')
        return list(ls)

    def getMovieID(self, movie):
        """
        Searches self.moviesInfo for movieID from input movie info arg.

        :param movie: str or list. Movie to find. Must be title (str) or list of [title (str), year (str or int)]
        :return: list. Returns all movies that match search criteria
        """
        # set movieID as index from movie info data for method to function
        allMovies = self.moviesInfo.set_index('movieId')

        if isinstance(movie, list):  # if movie and year defined, reduce list to only first two items.
            movie = movie[:2]  # check list items are strings
            movie = [str(item) for item in movie]  # search raw data for the title
            df = allMovies[allMovies['title'].str.contains(movie[0])]['year']
            if len(movie) > 1:  # if more than one movie found for title, search data found for the year given
                ls = df[df.str.contains(movie[1])].index
            else:  # if only one movie found, return movieID
                ls = df.index
        elif isinstance(movie, str):  # if only movie title defined, return list of movieIDs for that title name
            ls = allMovies[allMovies['title'].str.contains(movie)].index
        else:
            print('\33[91m', "Error: movie must in form ['title', 'year'] or 'title'.", '\33[0m', sep='')
            return
        return list(ls)

    def getUserRatings(self, userID, data=None, drop=True, buildTable=False):
        """
        Returns all ratings for given userID. Searches either ratingsPivot DF if given in data kwarg, or self.testData.

        :param userID: int. ID to return ratings for
        :param data: DataFrame, default=None. Pivot DF of movies against users to search through
        :param drop: bool, default=True. Drop movies the user has rated that are not in self.trainData
        :param buildTable: bool, default=False. Add title and year to DF and return this DF
        :return: Series or DataFrame. Returns series of movieIDs and ratings unless buildTable=True
        """
        if data is None:  # search test data for user if data not given
            userRatings = self.testData.loc[[userID]].dropna(axis=1).T.rename(columns={userID: 'rating'})
        else:  # search data given
            userRatings = data.loc[[userID]].dropna(axis=1).T.rename(columns={userID: 'rating'})

        if drop:  # check user's rated movies against those in ratings pivot and remove ratings not in pivot table
            diff = list(set(userRatings.index) - set(self.trainData.columns))
            userRatings.drop(diff, inplace=True)

        if buildTable:  # add info for titles and years of movies and return data frame
            userRatings.insert(0, 'title', self.getInfo(userRatings.index, get='title'))
            userRatings.insert(1, 'year', self.getInfo(userRatings.index, get='year'))
            print("User ", userID, "'s ratings:", sep='')
            return userRatings

        return userRatings['rating']  # return series of movieIDs and ratings

    def addTestUser(self, ratings):
        """
        Adds ratings to self.testData for userID 0.

        :param ratings: dict. {ID:rating} pairs for ratings to add.
        """
        self.testData.loc[0] = pd.Series(ratings)  # replace any previous test user with given ratings

    def buildMeanRatings(self, update=True):
        """
        Builds DF of movieID, title, year, popularity, ratingSize, ratingMean and genre for all movies in self.trainData

        :param update: bool, default=True. Update self.ratingsMean with generated DF
        :return: DataFrame. Generated mean ratings DF
        """
        # calculate means and ratings counts for every movie and rename data frames accordingly
        avg = self.trainData.mean().rename('ratingMean')
        size = self.trainData.count().rename('rating#')

        # combine into one data frame and sort
        df = size.to_frame().join(avg).sort_values(['rating#', 'ratingMean'], ascending=False)
        # add information on titles, years, genres and rankings for popularity
        df.insert(0, 'title', self.getInfo(df.index, get='title'))
        df.insert(1, 'year', self.getInfo(df.index, get='year'))
        df['genres'] = self.getInfo(df.index, get='genres')
        df.insert(2, 'popularity', np.arange(len(df)) + 1)  # rankings

        if update:  # update internally stored variable for the mean ratings with a copy
            self.ratingsMean = df.copy()
        return df

    def buildTable(self, similarMovies=None, ratingPredictions=None, predict='calc'):
        """
        Adds similarMovies and ratingPredictions series to the ratingsMean DF and sorts by descending similarity.

        :param similarMovies: Series, default=None. MovieIDs and their similarities to reduce ratingsMean DF by
        :param ratingPredictions: Series, default=None. MovieIDs and their predicted ratings to reduce ratingsMean DF by
        :param predict: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings
        :return: DataFrame. Generated mean ratings, similarities and predicted ratings DF
        """
        # check if mean ratings table has been stored, build if not
        ratingsMean = self.buildMeanRatings() if self.ratingsMean is None else self.ratingsMean.copy()
        ratingsMean['similarity'] = similarMovies  # add similarity values

        # add predicted ratings based on predict parameter
        if predict == 'calc':  # predict with calculated
            ratingsMean['rating'] = ratingPredictions
        elif predict == 'mean':  # predict with mean rating
            ratingsMean['rating'] = ratingsMean['ratingMean']
        elif predict == 'sims':  # predict with similarity multiplied by max rating
            ratingsMean['rating'] = ratingsMean['similarity'] * self.maxRating
        elif predict == 'norm_sims':  # predict with similarity normalised then multiplied by max rating
            ratingsMean['rating'] = ratingsMean['similarity'] * self.maxRating / ratingsMean['similarity'].max()
        elif predict == 'rand':  # predict with random ratings
            ratingsMean['rating'] = pd.Series({ID: random.uniform(0, 1)
                                               for ID in ratingsMean.index}).multiply(self.maxRating)

        # drop movies with missing similarity values and sort by descending similarity
        return ratingsMean.dropna(subset=['similarity']).sort_values('similarity', ascending=False)

    def buildMatrix(self, update=True):
        """
        Builds User oriented SciPy CSR SparseMatrix from self.trainData and generates dictionaries for looking up
        IDs from indices and vice versa. Stores these in self variables.
        The CSR matrix can be transposed using [MATRIX].transpose() for item-based modelling.

        :param update: bool, default=True. Update self stored matrix and dictionaries.
        :return: SciPySparseMatrix. Sparse Matrix of ratingsPivot. User focused.
        """
        print("Building sparse matrix...", end='')
        # convert ratings pivot table to numpy array and convert to compressed sparse row matrix
        userPivot_csr = csr_matrix(self.trainData_filled.to_numpy())

        # store key-value pairs of the indices of the csr matrix to movieIDs and userIDs
        user_dict = dict(zip([i for i in range(userPivot_csr.shape[0])], list(self.trainData.index)))
        movie_dict = dict(zip([i for i in range(userPivot_csr.shape[1])], list(self.trainData.columns)))
        print('\33[92m', "Done", '\33[0m')

        if update:  # update internally stored variables
            self.userPivot_csr = userPivot_csr
            self.userID_lookup = user_dict  # find userID from csr index
            self.movieID_lookup = movie_dict  # find movieID from csr index
            self.movieIndex_lookup = {value: key for key, value in movie_dict.items()}  # find csr index from movieID
        return userPivot_csr

    def buildModel(self, modelType, matrix=None, printStatus=False):
        """
        Builds models from given matrix.

        :param modelType: str. Model to build. Must be 'CF', 'SVD, or 'KNN'.
        :param matrix: object or str, default=None. Used to build the KNN model or determine which CF model to return.
        :param printStatus:  bool, default=False. Print progress and time taken to build the model.
        :return: DataFrames*2 or object. Model for given algorithm.
        """
        if printStatus:
            print('\33[1m', "\nBuilding ", modelType, " model", '\33[0m', sep='')
            t_start = perf_counter()  # start timer

        if modelType == 'CF':  # build collaborative filtering models from BuildCF class
            if matrix not in ['item', 'user']:  # build both item and user models
                itemModel, userModel = BuildCF(ml=self.ml,
                                               trainData=self.trainData_filled).buildAll(printStatus=printStatus)
                if printStatus:
                    print('\33[92m', "Done. Time: ", (perf_counter() - t_start), 's', '\33[0m', sep='')
                return itemModel, userModel
            else:  # build only item or user model depending on matrix parameter value
                model = BuildCF(ml=self.ml, trainData=self.trainData_filled).buildAll(get=matrix,
                                                                                      printStatus=printStatus)
        elif modelType == 'KNN':  # build KNN model and fit
            model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
            model.fit(matrix)
        elif modelType == 'SVD':  # build SVD model
            # check differences between ID in test and train data and build data frame containing all IDs
            newIDs = list(set(self.testData.index) - set(self.trainData.index))
            df = self.trainData.append(self.testData.loc[newIDs])

            arr = df.to_numpy(na_value=0)  # convert data frame to numpy array, filling missing values with 0
            trainData_subMeans = arr - np.mean(arr, axis=1).reshape(-1, 1)  # subtract means for SVD analysis
            u, s, vt = svds(trainData_subMeans, k=50)  # perform SVD

            # build model from matrices and normalise
            model = pd.DataFrame(u @ np.diag(s) @ vt, columns=df.columns, index=df.index)
            model /= model.max()
        else:
            return

        if printStatus:
            print('\33[92m', "Done. Time: ", (perf_counter() - t_start), 's', '\33[0m', sep='')
        return model

    def getSimilarMovies(self, movieID, model, modelType, neighbours=100, buildTable=False):
        """
        Gets similar movies from input movieID based on input model. If modelType='CF' or buildTables=True,
        generates predicted ratings from similarities and combines with self.ratingsMean DF.

        :param movieID: int. Movie to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF', 'KNN',
        :param neighbours: int, default=100. The number of top most similar movies to return.
        :param buildTable: bool, default=False. Join with self.ratingsMean DF and return this.
        :return: DataFrame or Series. if modelType='CF' or buildTable=True: Full DF containing movieID, title, year,
                                    popularity, ratingSize, ratingMean, similarities and predicted ratings.
                                    Else: series of indices and distances from sklearn models.
        """
        if modelType == 'CF':  # collaborative filtering algorithm
            # find similarities for the movie in the collaborative filtering model
            # drop input movieID and limit by most similar neighbours parameter
            similarMovies = model[movieID].drop(movieID, errors='ignore').dropna().nlargest(neighbours)
        elif modelType == 'KNN':  # k-nearest neighbours algorithm
            # look up the the index for the movieID and return the ratings from the csr matrix
            ratings = self.userPivot_csr.transpose()[self.movieIndex_lookup[movieID]]
            # determine the the k nearest movies and their distances
            distances, indices = model.kneighbors(ratings, n_neighbors=neighbours + 1)
            # reduce distances and indices to lists from arrays and create series
            similarMovies = pd.Series(distances.squeeze(), indices.squeeze())

            if buildTable:  # replace indices with movieIDs and reverse similarities (1 - distance)
                similarMovies = similarMovies.rename(index=self.movieID_lookup).drop(movieID).rsub(1)
        else:
            print('\33[91;3;1m', "ModelType definition Error", '\33[0m')
            return

        if buildTable:  # build data frame of title, year, genres, mean ratings etc. and drop predicted ratings column
            return self.buildTable(similarMovies, 0.0).drop(columns='rating')

        return similarMovies

    def getSimilarUsers(self, userID, model, modelType, userRatings, neighbours=100):
        """
        Gets similar users from input userID based on input model.

        :param userID: int. User to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF' or 'KNN'.
        :param userRatings: Series. MovieIDs and ratings for userID being tested.
        :param neighbours: int, default=100. The number of top most similar users to return.
        :return: Series. UserIDs and similarities to tested user.
        """
        if modelType == 'CF':  # collaborative filtering algorithm
            if model is None or userID not in model.index:  # if ID not in model, correlate user ratings with train data
                similarUsers = self.trainData.dropna(axis='columns', thresh=2).corrwith(userRatings, axis=1)
            else:  # if ID in model, return similarities
                similarUsers = model[userID]
        elif modelType == 'KNN':  # k-nearest neighbours algorithm
            # create data frame from user ratings and convert to csr matrix
            df = pd.DataFrame(columns=self.movieID_lookup.values()).append(userRatings, ignore_index=True)
            userRatings_csr = csr_matrix(df.to_numpy(na_value=0))
            userRatings /= self.maxRating  # normalise ratings

            # determine the the k nearest users and their distances
            distances, indices = model.kneighbors(userRatings_csr, n_neighbors=neighbours + 1)
            # create series, replace indices with movieIDs, and reverse similarities (1 - distance)
            similarUsers = pd.Series(distances.squeeze(), indices.squeeze()).rename(index=self.userID_lookup).rsub(1)
        else:
            print('\33[91;3;1m', "ModelType definition Error", '\33[0m')
            return

        # drop input userID and limit by most similar neighbours parameter
        return similarUsers.drop(userID, errors='ignore').dropna().nlargest(neighbours)

    def getPredictedRatings(self, similarities, userRatings, sums, sample=100):
        """
        Amalgamates similarities together for given user's ratings and predicts ratings for this user.

        :param similarities: DataFrame or Series. Similarity scores for similar users or movies
        :param userRatings: Series. MovieIDs and ratings for userID being tested.
        :param sums: Series or float. Sums with which to normalise the predicted ratings.
        :param sample: int, default=100. Top most similar movies to sample from item-based
                                        or highest rated movies from users to sample from user-based.
        :return: Series | Series. MovieIDs and similarities to tested user | Movies IDs and predicted ratings
        """

        similarMovies = pd.Series()  # empty series to add similarities to
        userBias = mean(userRatings) - self.mean  # user bias = mean user rating difference from the total ratings mean

        for ID in similarities:  # sample most similar movies to movie/top rated movies from user and add similarities
            reduced = similarities[ID].nlargest(sample)
            similarMovies = similarMovies.add(reduced, fill_value=0)

        similarMovies /= similarMovies.max()  # normalise similarities

        # sum weighted similarities for each similar movie given
        predictedRatings = similarities.sum(axis=1).dropna()
        # divide weighted similarity sums by sum of unweighted similarities per move and multiply by max rating
        predictedRatings = predictedRatings.divide(sums).multiply(self.maxRating)
        # apply user bias to predicted ratings and limit ratings to max rating for user with high positive biases
        predictedRatings += userBias
        predictedRatings[predictedRatings > self.maxRating] = self.maxRating

        return similarMovies, predictedRatings

    def itemBased(self, userID, model, modelType, neighbours=100, sample=None, threshold=2.0, predict='calc',
                  buildTable=False, printStatus=False):
        """
        Driver function for ItemBased algorithms. Gets similar items for given user ID based on ItemBased models.

        :param userID: int. User to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF', 'KNN',
        :param neighbours: int, default=100. The number of top most similar movies to return.
        :param sample: int, default=None. The number of top rated movies to sample from the input userID.
        :param threshold: float, default=2.0. The min user rating for a movie to be considered by the algorithm.
        :param predict: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings

        :param buildTable: bool, default=False. Return similarities and predicted ratings joined with self.ratingsMean
        :param printStatus: bool, default=False. Print current progress and time taken at each key stage
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings. Returns series of only similarities if buildTables=False.
        """
        if self.testData is not None and userID not in list(self.testData.index):  # check if userID in test data
            print('\33[91m', "Error: userID ", userID, " not in testData", '\33[0m', sep='')
            return

        # set sample size to max possible value if not given
        sample = len(self.testData.columns) if sample is None else sample
        # get user ratings, drop rated movies not in train data, sample top rated movies, and normalise
        userRatings = self.getUserRatings(userID, data=self.testData, drop=True).nlargest(sample) / self.maxRating
        # list of movieIDs the user has rated
        rated = list(userRatings.index)
        # reduce user ratings to only movies rated above the threshold rating
        userRatings = userRatings[userRatings > (threshold / self.maxRating)]

        if userRatings.empty:  # return if threshold rating condition returns no movies for the user
            return

        if printStatus:
            print("Getting similar movies for all of user ", userID, "'s rated movies...", sep='', end=' ')
            t_start = perf_counter()  # start timer

        similarities = pd.DataFrame()
        if modelType == 'CF':  # collaborative filtering algorithm
            similarities = model[userRatings.index]  # similarities to the user's rated movies
        elif modelType == 'KNN':  # k-nearest neighbours algorithm
            for movieID in list(userRatings.index):  # for each movie in userRatings, get similarities and add to df
                sims = self.getSimilarMovies(movieID, model, modelType, neighbours=neighbours)
                similarities = similarities.join(sims.rename(movieID), how='outer')
            # replace indices with movieIDs and reverse similarities (1 - distance)
            similarities = similarities.rename(index=self.movieID_lookup).rsub(1)

        sums = similarities.sum(axis=1)  # sum unweighted similarities for predicted ratings algorithm
        weighted = similarities.multiply(userRatings, axis=1)  # weight similarities by the user's ratings

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_start) / len(userRatings),
                  's /rated movie', '\33[0m', sep='')
            print("Weighting similarities by rating and predicting ratings...", end=' ')
            t_predict = perf_counter()  # start timer

        # sum similarities, predict ratings, and drop already rated movies
        similarMovies, predictedRatings = self.getPredictedRatings(weighted, userRatings, sums, sample=neighbours)
        similarMovies.drop(rated, errors='ignore', axis=0, inplace=True)

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_predict) / len(similarities.columns),
                  's /movie', '\33[0m', sep='')
            print('\33[94;1m', "Total Time: ", (perf_counter() - t_start), "s", '\33[0m', sep='')

        # build data frame of title, year, genres, mean ratings etc. or return similar movies series
        return self.buildTable(similarMovies, predictedRatings, predict) if buildTable else similarMovies

    def userBased(self, userID, model, modelType, neighbours=100, sample=None, threshold=2.0, predict='calc',
                  buildTable=False, printStatus=False):
        """
        Driver function for UserBased algorithms. Gets similar items for given user ID based on UserBased models.

        :param userID: int. User to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF', 'KNN',
        :param neighbours: int, default=100. The number of top most similar users to return.
        :param sample: int, default=None. The number of top rated movies to sample from the similar users.
        :param threshold: float, default=2.0. The min user rating for a movie to be considered by the algorithm.
        :param predict: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings.

        :param buildTable: bool, default=False. Return similarities and predicted ratings joined with self.ratingsMean
        :param printStatus: bool, default=False. Print current progress and time taken at each key stage
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings. Returns series of only similarities if buildTables=False.
        """
        if self.testData is not None and userID not in list(self.testData.index):  # check if userID in test data
            print('\33[91m', "Error: userID ", userID, " not in testData", '\33[0m', sep='')
            return

        # set sample size to max possible value if not given
        sample = len(self.testData.columns) if sample is None else sample
        # get user ratings and drop rated movies not in train data
        userRatings = self.getUserRatings(userID, data=self.testData, drop=True)
        # list of movieIDs the user has rated
        rated = list(userRatings.index)
        # reduce user ratings to only movies rated above the threshold rating
        userRatings = userRatings[userRatings > threshold]

        if userRatings.empty:  # return if threshold rating condition returns no movies for the user
            return

        if printStatus:
            print("Getting similar users for user ", userID, "...", sep='', end=' ')
            t_start = perf_counter()  # start timer

        # get similar users, get the ratings for these users and normalise the ratings
        similarUsers = self.getSimilarUsers(userID, model, modelType, userRatings, neighbours=neighbours)
        similarities = self.trainData.loc[similarUsers.index].divide(self.maxRating)

        weighted = similarities.multiply(similarUsers, axis=0)  # weight similar user's ratings by similarities
        sums = weighted.sum()  # sum weighted similarities for predicted ratings algorithm
        # apply each user's bias to similarity scores and transpose for summing
        unbiased = weighted.subtract(self.biasFactor, axis=0).dropna(how='all', axis=0).T

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_start) / len(similarUsers),
                  "s /similar user", '\33[0m', sep='')
            print("Weighting similarities by rating and predicting ratings...", end=' ')
            t_predict = perf_counter()  # start timer

        # sum similarities, predict ratings, and drop already rated movies
        similarMovies, predictedRatings = self.getPredictedRatings(unbiased, userRatings / self.maxRating, sums, sample)
        similarMovies.drop(rated, errors='ignore', axis=0, inplace=True)

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_predict) / len(similarUsers),
                  "s /similar user", '\33[0m', sep='')
            print('\33[94;1m', "Total Time: ", (perf_counter() - t_start), "s", '\33[0m', sep='')

        # build data frame of title, year, genres, mean ratings etc. or return similar movies series
        return self.buildTable(similarMovies, predictedRatings, predict) if buildTable else similarMovies

    def SVD(self, userID, model, sample=100, predict='calc', buildTable=False, printStatus=False):
        """
        Uses SVD to return similarities and predicted ratings for a given user from a similarity matrix.

        :param userID: int. User to find similarities for.
        :param model: object. Similarity matrix.
        :param sample: int, default=100. Sample of most similar movies to the user to return.
        :param predict: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings.
        :param buildTable: bool, default=False. Return similarities and predicted ratings joined with self.ratingsMean
        :param printStatus: bool, default=False. Print current progress and time taken at each key stage
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings. Returns series of only similarities if buildTables=False.
        """
        if userID not in list(model.index):  # check if userID in test data
            print('\33[91m', "Error: userID ", userID, " not in SVD matrix", '\33[0m', sep='')
            return

        if printStatus:
            print("Getting similar movies for user ", userID, "...", sep='', end=' ')
            t_start = perf_counter()  # start timer

        similarMovies = model.loc[userID].nlargest(sample)  # return similarities from and sample most similar
        userMean = mean(self.getUserRatings(userID, data=self.testData, drop=True))  # calculate user's mean rating
        predictedRatings = similarMovies + userMean  # add user's mean to similarities to get predicted ratings

        if printStatus:
            print('\33[94;1m', "Total Time: ", (perf_counter() - t_start), "s", '\33[0m', sep='')

        # build data frame of title, year, genres, mean ratings etc. or return similar movies series
        return self.buildTable(similarMovies, predictedRatings, predict) if buildTable else similarMovies

    def random(self, _, __, randomRatings=True, buildTable=False):
        """
        Generates random recommendations and ratings.

        :param _: dummy parameter. Nullifies error raised when inputting ID from Tester object.
        :param __: dummy parameter. Nullifies error raised when inputting model from Tester object.
        :param randomRatings: bool. Determine whether ratings generated are uniquely random or normalised multiple of
                                    random similarities
        :param buildTable: bool, default=False. Return similarities and predicted ratings joined with self.ratingsMean
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings
        """
        # generate random similarity scores for all movies
        similarMovies = pd.Series({ID: random.uniform(0, 1) for ID in self.trainData})

        if randomRatings:  # generate random ratings for all movies
            predictedRatings = pd.Series({ID: random.uniform(0, 1) for ID in self.trainData}).multiply(self.maxRating)
        else:  # multiply similarities by max rating for rating predictions
            predictedRatings = similarMovies.multiply(self.maxRating)

        # build data frame of title, year, genres, mean ratings etc. or return similar movies series
        return self.buildTable(similarMovies, predictedRatings) if buildTable else similarMovies
