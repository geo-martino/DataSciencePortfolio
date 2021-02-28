import random
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from MovieLensData import MovieLensData


def removeMultiLevel(ratingsPivot):
    """
    Checks for MultiLevel columns in ratingsPivot and removes them.

    :param ratingsPivot: DataFrame. Pivoted DF of columns containing MultiLevel Index with 'title' and 'year' axes.
    :return: ratingsPivot: DataFrame. Pivoted DF with adjusted columns.
    """
    if isinstance(ratingsPivot.columns[0], tuple):
        print("\nRemoving MultiLevel index from pivot df...", end='')
        ratingsPivot = ratingsPivot.droplevel(['title', 'year'], axis=1)
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
    buildSimilarUsers(update=True):
        Builds cosine similarity correlation DF of ratings between all users.

    buildSimilarMovieRatings(update=True):
        Builds cosine similarity correlation DF of ratings between all items.

    buildGenres(update=True):
        Builds cosine similarity correlation DF of genres between all items.

    buildYears(update=True):
        Builds cosine similarity correlation DF of years between all items.

    buildAll(get=None):
        Runs all the functions in the object if self variables are None and combines item rating, genre and year
        correlations.
    """

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
        self.ml = MovieLensData() if ml is None else ml

        if trainData is not None:
            self.ratingsPivot = removeMultiLevel(trainData).fillna(0)
        elif self.ml.ratingsPivot is not None:
            self.ratingsPivot = removeMultiLevel(self.ml.ratingsPivot).fillna(0)
        else:
            print('\33[91m', "Error: Training data not given and ratingsPivot not found.", '\33[0m', sep='')
            print("Building quick pivot df with MinRatings of 1000 for Users and Movies.")
            self.ratingsPivot = self.ml.buildPivot(extraColumns=False, quick=True, update=True).fillna(0)

    def buildSimilarUsers(self, update=True):
        """
        Builds cosine similarity correlation DF of ratings between all users.

        :param update: bool, default=True. Update self.corrUsers with generated DF
        :return: DataFrame. Cosine similarity correlation dataframe for user ratings
        """

        print("Correlating ratings for all users...", end='')
        corrUsers = pd.DataFrame(np.corrcoef(self.ratingsPivot.to_numpy(), rowvar=True),
                                 index=self.ratingsPivot.index, columns=self.ratingsPivot.index).rename_axis(None)
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.corrUsers = corrUsers
        return corrUsers

    def buildSimilarMovieRatings(self, update=True):
        """
         Builds cosine similarity correlation DF of ratings between all items.

        :param update: bool, default=True. Update self.corrMovieRatings with generated DF
        :return: DataFrame. Cosine similarity correlation dataframe for item ratings
        """

        print("Correlating ratings for all movies...", end='')
        corrMovieRatings = pd.DataFrame(np.corrcoef(self.ratingsPivot.to_numpy(), rowvar=False),
                                        index=self.ratingsPivot.columns, columns=self.ratingsPivot.columns)
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.corrMovieRatings = corrMovieRatings
        return corrMovieRatings

    def buildGenres(self, update=True):
        """
        Builds cosine similarity correlation DF of genres between all items.

        :param update: bool, default=True. Update self.corrGenres with generated DF
        :return: DataFrame. Cosine similarity correlation dataframe for item genres
        """
        allGenres = self.ml.buildGenreBitfield() if self.ml.genreBitfield is None else self.ml.genreBitfield.copy()

        print("Correlating genres for all movies...", end='')
        corrGenres = pd.DataFrame(np.corrcoef(allGenres.to_numpy(), rowvar=True),
                                  index=allGenres.index, columns=allGenres.index)
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.corrGenres = corrGenres
        return corrGenres

    def buildYears(self, update=True):
        """
        Builds cosine similarity correlation DF of years between all items.

        :param update: bool, default=True. Update self.corrYears with generated DF
        :return: DataFrame. Cosine similarity correlation dataframe for item years.
        """
        print("Correlating years for all movies...", end='')
        movies = self.ml.moviesReduced.copy() if self.ml.moviesReduced is not None else self.ml.moviesRaw.copy()
        allYears = movies.set_index('movieId')['year'].dropna().astype(int)
        diff = np.abs(np.subtract.outer(list(allYears), list(allYears)))
        similarity = np.exp(-diff / 7.0)
        corrYears = pd.DataFrame(similarity, index=list(allYears.index), columns=list(allYears.index))
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.corrYears = corrYears
        return corrYears

    def buildAll(self, get=None):
        """
        Runs all the functions in the object if self variables are None and combines item rating, genre and year
        correlations.

        :param get: str, default=None. Define which DataFrame to build and return 'item' or 'user'. None returns both
        :return: DataFrames. Returns DataFrames defined in get kwarg.
        """
        if self.corrUsers is None and get != 'item':
            self.buildSimilarUsers()

        if self.corrMovieRatings is None and get != 'user':
            self.buildSimilarMovieRatings()

        if self.corrGenres is None and get != 'user':
            self.buildGenres()

        if self.corrYears is None and get != 'user':
            self.buildYears()

        if get != 'user':
            print("Generating combined correlation...", end='')
            self.corrItems = (self.corrMovieRatings * self.corrGenres * self.corrYears).rename_axis(None, axis=1)
            self.corrItems.dropna(how='all', axis=0, inplace=True)
            self.corrItems.dropna(how='all', axis=1, inplace=True)
            print('\33[92m', "Done", '\33[0m')

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

    buildTable(similarMovies=None, ratingPredictions=None, pred='calc'):
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

    itemBased(userID, model, modelType, neighbours=100, sample=None, pred='calc', threshold=2.0,
                buildTable=False, printStatus=False):
        Driver function for ItemBased algorithms. Gets similar items for given user ID based on ItemBased models.

    userBased(userID, model, modelType, neighbours=100, sample=None, pred='calc', threshold=2.0,
                buildTable=False, printStatus=False):
        Driver function for UserBased algorithms. Gets similar items for given user ID based on UserBased models.

    random(_, __, randomRatings=True):
        Generates random recommendations and ratings.
    """

    ratingsMean = None

    userPivot_csr = None
    userID_lookup = None
    movieID_lookup = None
    movieIndex_lookup = None

    def __init__(self, ml=None, trainData=None, testData=None, maxRating=5.0):
        """
        Initialises attributes, checks for ratingsPivot present, and removes MultiLevel columns if present.

        :param ml: object, default=None. MovieLensData object for function calls.
        :param trainData: DataFrame, default=None. Pivoted trainSet DF of columns:movieId, index:userId, values:ratings
        :param testData: DataFrame, default=None. Pivoted testSet DF of columns:movieId, index:userId, values:ratings
        :param maxRating: int, default=5.0.
        """
        print("Initialising object...", end='')
        self.ml = MovieLensData() if ml is None else ml
        self.moviesInfo = self.ml.moviesRaw
        self.maxRating = maxRating

        if trainData is not None:
            self.trainData = removeMultiLevel(trainData)
        elif self.ml.ratingsPivot is not None:
            self.trainData = removeMultiLevel(self.ml.ratingsPivot)
        else:
            print('\33[91m', "\nError: Training data not given and ratingsPivot not found.", '\33[0m', sep='')
            print("Building quick pivot df with MinRatings of 1000 for Users and Movies.")
            self.trainData = self.ml.buildPivot(quick=True)

        self.testData = testData if testData is not None else self.trainData
        self.mean = self.trainData.mean().mean() / self.maxRating
        self.trainData_filled = self.trainData.fillna(0)
        print('\33[92m', "Done", '\33[0m')

    def getInfo(self, movieID, get=None):
        """
        Returns info as defined by get kwarg from moviesInfo for input movieIDs.

        :param movieID: int or list. MovieIDs to search for in self.moviesInfo
        :param get: str, default=None. What info to return. Must be either 'title', 'year', or 'genres'
        :return: list. List of strings of all info found for given IDs
        """
        movieID = [movieID] if isinstance(movieID, int) else movieID
        movies = self.moviesInfo[self.moviesInfo['movieId'].isin(movieID)].set_index('movieId')
        movies = movies.reindex(movieID)

        if get == 'title':
            ls = [i for i in movies['title']]
        elif get == 'year':
            ls = [i for i in movies['year']]
        elif get == 'genres':
            ls = [i for i in movies['genres']]
        else:
            return movies.set_index('movieId')
        return list(ls)

    def getMovieID(self, movie):
        """
        Searches self.moviesInfo for movieID from input movie info arg.

        :param movie: str or list. Movie to find. Must be title (str) or list of [title (str), year (str or int)]
        :return: list. Returns all movies that match search criteria
        """
        allMovies = self.moviesInfo.set_index('movieId')

        if isinstance(movie, list):
            movie = movie[:2]
            movie = [str(item) for item in movie]
            df = allMovies[allMovies['title'].str.contains(movie[0])]['year']
            if len(movie) > 1:
                ls = df[df.str.contains(movie[1])].index
            else:
                ls = df.index
        elif isinstance(movie, str):
            ls = allMovies[allMovies['title'].str.contains(movie)].index
        else:
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
        if data is None:
            userRatings = self.testData.loc[[userID]].dropna(axis=1).T.rename(columns={userID: 'rating'})
        else:
            userRatings = data.loc[[userID]].dropna(axis=1).T.rename(columns={userID: 'rating'})

        if drop:
            diff = list(set(userRatings.index) - set(self.trainData.columns))
            userRatings.drop(diff, inplace=True)

        if buildTable:
            userRatings.insert(0, 'title', self.getInfo(userRatings.index, get='title'))
            userRatings.insert(1, 'year', self.getInfo(userRatings.index, get='year'))
            print("User ", userID, "'s ratings:", sep='')
            return userRatings
        return userRatings['rating']

    def addTestUser(self, ratings):
        """
        Adds ratings to self.testData for userID 0.

        :param ratings: dict. {ID:rating} pairs for ratings to add.
        """
        self.testData.loc[0] = pd.Series(ratings)

    def buildMeanRatings(self, update=True):
        """
        Builds DF of movieID, title, year, popularity, ratingSize, ratingMean and genre for all movies in self.trainData

        :param update: bool, default=True. Update self.ratingsMean with generated DF
        :return: DataFrame. Generated mean ratings DF
        """
        avg = self.trainData.mean().rename('ratingMean')
        size = self.trainData.count().rename('rating#')
        df = size.to_frame().join(avg).sort_values(['rating#', 'ratingMean'], ascending=False)
        df.insert(0, 'title', self.getInfo(df.index, get='title'))
        df.insert(1, 'year', self.getInfo(df.index, get='year'))
        df.insert(2, 'popularity', np.arange(len(df)) + 1)
        df['genres'] = self.getInfo(df.index, get='genres')
        if update:
            self.ratingsMean = df.copy()
        return df

    def buildTable(self, similarMovies=None, ratingPredictions=None, pred='calc'):
        """
        Adds similarMovies and ratingPredictions series to the ratingsMean DF and sorts by descending similarity.

        :param similarMovies: Series, default=None. MovieIDs and their similarities to reduce ratingsMean DF by
        :param ratingPredictions: Series, default=None. MovieIDs and their predicted ratings to reduce ratingsMean DF by
        :param pred: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings
        :return: DataFrame. Generated mean ratings, similarities and predicted ratings DF
        """
        ratingsMean = self.buildMeanRatings() if self.ratingsMean is None else self.ratingsMean.copy()
        ratingsMean['similarity'] = similarMovies

        if pred == 'calc':
            ratingsMean['rating'] = ratingPredictions
        elif pred == 'mean':
            ratingsMean['rating'] = ratingsMean['ratingMean']
        elif pred == 'sims':
            ratingsMean['rating'] = ratingsMean['similarity'] * self.maxRating
        elif pred == 'norm_sims':
            ratingsMean['rating'] = ratingsMean['similarity'] * self.maxRating / ratingsMean['similarity'].max()
        elif pred == 'rand':
            ratingsMean['rating'] = pd.Series({ID: random.uniform(0, 1)
                                               for ID in ratingsMean.index}).multiply(self.maxRating)

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
        trainData_filled = self.trainData_filled.to_numpy()
        userPivot_csr = csr_matrix(trainData_filled)
        user_dict = dict(zip([i for i in range(userPivot_csr.shape[0])], list(self.trainData.index)))
        movie_dict = dict(zip([i for i in range(userPivot_csr.shape[1])], list(self.trainData.columns)))
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.userPivot_csr = userPivot_csr
            self.userID_lookup = user_dict
            self.movieID_lookup = movie_dict
            self.movieIndex_lookup = {value: key for key, value in movie_dict.items()}
        return userPivot_csr

    def buildModel(self, modelType, matrix=None, printStatus=False):
        """
        Builds models from given matrix.

        :param modelType: str. Model to build. Must be 'CF', 'KNN',
        :param matrix: object, default=None.
        :param printStatus:  bool, default=False. Print progress and time taken to build the model.
        :return: DataFrames*2 or object. Model for given algorithm.
        """
        if printStatus:
            print('\33[1m', "\nBuilding ", modelType, " model", '\33[0m', sep='')
            t_start = perf_counter()

        if modelType == 'CF':
            itemModel, userModel = BuildCF(ml=self.ml, trainData=self.trainData_filled).buildAll()
            if printStatus:
                print('\33[92m', "Done. Time: ", (perf_counter() - t_start), 's', '\33[0m', sep='')
            return itemModel, userModel
        elif modelType == 'KNN':
            model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
            model.fit(matrix)
        elif modelType == 'SVD':
            newIDs = list(set(self.testData.index) - set(self.trainData.index))
            df = self.trainData.append(self.testData.loc[newIDs])
            arr = df.to_numpy(na_value=0)
            trainData_subMeans = arr - np.mean(arr, axis=1).reshape(-1, 1)
            u, s, vt = svds(trainData_subMeans, k=50)
            model = pd.DataFrame(u @ np.diag(s) @ vt, columns=df.columns, index=df.index)
            model /= model.max()
        elif modelType == 'KMeans':
            model = KMeans(n_jobs=-1)
        elif modelType == 'LR':
            model = LinearRegression(n_jobs=-1)
        elif modelType == 'DT':
            model = DecisionTreeRegressor()
        elif modelType == 'RF':
            model = RandomForestRegressor(n_jobs=-1)
        elif modelType == 'SVC':
            model = SVC()
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
        if modelType == 'CF' or modelType == 'SVD':
            similarMovies = model[movieID].drop(movieID,
                                                errors='ignore').dropna().sort_values(ascending=False)[:neighbours]
            if buildTable:
                return self.buildTable(similarMovies, 0.0).drop(columns='rating')
            else:
                return similarMovies
        elif modelType == 'KNN':
            distances, indices = model.kneighbors(self.userPivot_csr.transpose()[self.movieIndex_lookup[movieID]],
                                                  n_neighbors=neighbours + 1)
        else:
            print('\33[91;3;1m', "ModelType definition Error", '\33[0m')
            return

        d_i = pd.Series(distances.squeeze(), indices.squeeze())

        if buildTable:
            similarMovies = d_i.rename(index=self.movieID_lookup).drop(movieID).rsub(1)
            predictedRatings = similarMovies * self.maxRating / similarMovies.max()
            return self.buildTable(similarMovies, predictedRatings)
        else:
            return d_i

    def getSimilarUsers(self, userID, model, modelType, neighbours=100):
        """
        Gets similar users from input userID based on input model.

        :param userID: int. User to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF', 'KNN',
        :param neighbours: int, default=100. The number of top most similar users to return.
        :return: Series. UserIDs and similarities to tested user.
        """
        userRatings = self.getUserRatings(userID, data=self.testData, drop=True)

        if modelType == 'CF' or modelType == 'SVD':
            if userID not in model.index:
                similarUsers = self.trainData.T.dropna(axis='columns', thresh=2).corrwith(userRatings)
            else:
                similarUsers = model[userID]
            return similarUsers.drop(userID, errors='ignore').dropna().nlargest(neighbours)
        else:
            df = pd.DataFrame(columns=self.movieID_lookup.values()).append(userRatings, ignore_index=True)
            userRatings_csr = csr_matrix(df.to_numpy(na_value=0))
            userRatings /= self.maxRating

        if modelType == 'KNN':
            distances, indices = model.kneighbors(userRatings_csr, n_neighbors=neighbours + 1)
        else:
            print('\33[91;3;1m', "ModelType definition Error", '\33[0m')
            return

        similarUsers = pd.Series(distances.squeeze(), indices.squeeze()).rename(index=self.userID_lookup).rsub(1)
        return similarUsers.drop(userID, errors='ignore')

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
        similarMovies = pd.Series()
        userBias = mean(userRatings) - self.mean

        for ID in similarities:
            reduced = similarities[ID].nlargest(sample)
            similarMovies = similarMovies.add(reduced, fill_value=0)

        similarMovies /= similarMovies.max()

        predictedRatings = similarities.sum(axis=1).dropna()
        predictedRatings = predictedRatings.divide(sums).multiply(self.maxRating)
        predictedRatings += userBias
        predictedRatings[predictedRatings > self.maxRating] = self.maxRating

        return similarMovies, predictedRatings

    def itemBased(self, userID, model, modelType, neighbours=100, sample=None, pred='calc', threshold=2.0,
                  buildTable=False, printStatus=False):
        """
        Driver function for ItemBased algorithms. Gets similar items for given user ID based on ItemBased models.

        :param userID: int. User to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF', 'KNN',
        :param neighbours: int, default=100. The number of top most similar movies to return.
        :param sample: int, default=None. The number of top rated movies to sample from the input userID.
        :param pred: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings
        :param threshold: float, default=2.0. The min user rating for a movie to be considered by the algorithm.
        :param buildTable: bool, default=False. Return similarities and predicted ratings joined with self.ratingsMean
        :param printStatus: bool, default=False. Print current progress and time taken at each key stage
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings. Returns series of only similarities if buildTables=False.
        """
        if self.testData is not None and userID not in list(self.testData.index):
            print('\33[91m', "Error: userID ", userID, " not in testData", '\33[0m', sep='')
            return

        sample = len(self.testData.columns) if sample is None else sample
        userRatings = self.getUserRatings(userID, data=self.testData, drop=True).nlargest(sample) / self.maxRating
        rated = list(userRatings.index)
        userRatings = userRatings[userRatings > (threshold / self.maxRating)]

        if printStatus:
            print("Getting similar movies for all of user ", userID, "'s rated movies...", sep='', end=' ')
            t_start = perf_counter()

        if modelType == 'KNN':
            similarities = pd.DataFrame()
            for movieID in list(userRatings.index):
                sims = self.getSimilarMovies(movieID, model, modelType, neighbours=neighbours)
                similarities = similarities.join(sims.rename(movieID), how='outer')
            similarities = similarities.rename(index=self.movieID_lookup).rsub(1)
        else:
            similarities = model[userRatings.index]

        sums = similarities.sum(axis=1)
        weighted = similarities.multiply(userRatings, axis=1)

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_start) / len(userRatings),
                  's /rated movie', '\33[0m', sep='')
            print("Weighting similarities by rating and predicting ratings...", end=' ')
            t_pred = perf_counter()

        similarMovies, predictedRatings = self.getPredictedRatings(weighted, userRatings, sums, sample=neighbours)
        similarMovies.drop(rated, errors='ignore', axis=0, inplace=True)

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_pred) / len(similarities.columns),
                  's /movie', '\33[0m', sep='')
            print('\33[94;1m', "Total Time: ", (perf_counter() - t_start), "s", '\33[0m', sep='')

        return self.buildTable(similarMovies, predictedRatings, pred) if buildTable else similarMovies

    def userBased(self, userID, model, modelType, neighbours=100, sample=None, pred='calc', threshold=2.0,
                  buildTable=False, printStatus=False):
        """
        Driver function for UserBased algorithms. Gets similar items for given user ID based on UserBased models.

        :param userID: int. User to find similarities for.
        :param model: object. Model to use to generate similarities.
        :param modelType: str. String representing the model type given. Must be 'CF', 'KNN',
        :param neighbours: int, default=100. The number of top most similar users to return.
        :param sample: int, default=None. The number of top rated movies to sample from the similar users.
        :param pred: str, default='calc'. Rating prediction algorithm to use.
                                        'calc': formulaically calculated value,
                                        'mean': use mean values from trainData,
                                        'sims': multiply similarity scores by max possible rating,
                                        'norm_sims': normalise similarity scores then multiply by max possible rating,
                                        'rand': randomly generate ratings
        :param threshold: float, default=2.0. The min user rating for a movie to be considered by the algorithm.
        :param buildTable: bool, default=False. Return similarities and predicted ratings joined with self.ratingsMean
        :param printStatus: bool, default=False. Print current progress and time taken at each key stage
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings. Returns series of only similarities if buildTables=False.
        """
        if self.testData is not None and userID not in list(self.testData.index):
            print('\33[91m', "Error: userID ", userID, " not in testData", '\33[0m', sep='')
            return

        sample = len(self.testData.columns) if sample is None else sample
        userRatings = self.getUserRatings(userID, data=self.testData, drop=True) / self.maxRating
        rated = list(userRatings.index)
        userRatings = userRatings[userRatings > (threshold / self.maxRating)]

        if printStatus:
            print("Getting similar users for user ", userID, "...", sep='', end=' ')
            t_start = perf_counter()

        similarUsers = self.getSimilarUsers(userID, model, modelType, neighbours=neighbours)
        similarities = self.trainData.loc[similarUsers.index].divide(self.maxRating)
        # sums = similarities.sum(axis=0)
        sums = similarUsers.sum()

        weighted = similarities.multiply(similarUsers, axis=0)
        biasFactor = (self.trainData.mean(axis=1) / self.maxRating) - self.mean
        unbiased = weighted.subtract(biasFactor, axis=0).dropna(how='all', axis=0).T

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_start) / len(similarUsers),
                  "s /similar user", '\33[0m', sep='')
            print("Weighting similarities by rating and predicting ratings...", end=' ')
            t_pred = perf_counter()

        similarMovies, predictedRatings = self.getPredictedRatings(unbiased, userRatings, sums, sample)
        similarMovies.drop(rated, errors='ignore', axis=0, inplace=True)

        if printStatus:
            print('\33[92m', "Done. Average time: ", (perf_counter() - t_pred) / len(similarUsers),
                  "s /similar user", '\33[0m', sep='')
            print('\33[94;1m', "Total Time: ", (perf_counter() - t_start), "s", '\33[0m', sep='')

        return self.buildTable(similarMovies, predictedRatings, pred) if buildTable else similarMovies

    def SVD(self, userID, model, sample=100, pred='calc', buildTable=False, printStatus=False):
        if userID not in list(model.index):
            print('\33[91m', "Error: userID ", userID, " not in SVD matrix", '\33[0m', sep='')
            return

        if printStatus:
            print("Getting similar movies for user ", userID, "...", sep='', end=' ')
            t_start = perf_counter()

        similarMovies = model.loc[userID].nlargest(sample)
        predictedRatings = similarMovies * self.maxRating

        if printStatus:
            print('\33[94;1m', "Total Time: ", (perf_counter() - t_start), "s", '\33[0m', sep='')

        return self.buildTable(similarMovies, predictedRatings, pred) if buildTable else similarMovies

    def random(self, _, __, randomRatings=True):
        """
        Generates random recommendations and ratings.

        :param _: dummy parameter. Nullifies error raised when inputting ID from Tester object.
        :param __: dummy parameter. Nullifies error raised when inputting model from Tester object.
        :param randomRatings: bool. Determine whether ratings generated are uniquely random or normalised multiple of
                                    random similarities
        :return: DataFrame. Full DF containing movieID, title, year, popularity, ratingSize, ratingMean, similarities
                            and predicted ratings
        """
        similarMovies = pd.Series({ID: random.uniform(0, 1) for ID in self.trainData})
        if randomRatings:
            predictedRatings = pd.Series({ID: random.uniform(0, 1) for ID in self.trainData}).multiply(self.maxRating)
        else:
            predictedRatings = similarMovies.multiply(self.maxRating)
        return self.buildTable(similarMovies, predictedRatings)
