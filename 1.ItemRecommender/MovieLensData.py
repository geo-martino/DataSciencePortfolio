import random
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split


class MovieLensData:
    """
    Updates, loads and arranges data from the MovieLens dataset.

    :attribute moviesReduced: DataFrame. Reduced DF of moviesRaw (see method: reduce)
    :attribute ratingsReduced: DataFrame. Reduced DF of ratingsRaw (see method: reduce)

    :attribute ratingsPivot: DataFrame. Pivoted DF of- columns: movieId, index: userId, values: ratings
                            (see method: buildPivot)
    :attribute ratingsMean: DataFrame. DF of movieID, title, year, popularity, ratingSize, ratingMean and genre.
    :attribute genreBitfield: DataFrame. DF of movieIDs and respective genre bitfield (see method: buildGenreBitfield)

    :attribute moviesRaw: DataFrame. Raw data loaded from movies.csv. Columns: movieId, titles+years and genres.
    :attribute ratingsRaw: DataFrame. Raw data loaded from ratings.csv. Columns: userId, movieId, ratings and timestamp.

    :methods:
    resetAll():
        Resets all self variables bar moviesRaw or ratingsReduced to clear memory.

    filterIDs(idType, minRatings=400, maxRatings=1000000, fraction=0.15):
        Filters and returns list of IDs in moviesRaw and ratingsRaw based on min and max rating counts, or a random
        sample.

    reduce(IDs, idType, dfType):
        Update self.moviesReduced and self.ratingsReduced by input IDs.

    buildPivot(extraColumns=False, quick=False, update=True):
        Builds pivot DF from moviesReduced & ratingsReduced if not None, else uses self.moviesRaw & self.ratingsRaw.
        Pivots movieIDs as columns, userIDs as index, ratings as values filling missing ratings with nan.

    buildGenreBitfield(IDs=None, update=True):
        Builds DF of genres as a bit field for all movieIDs in moviesReduced if not None, else uses self.moviesRaw.

    buildMeanRatings(update=True, extraInfo=False):
        Builds DF of movieID, title, year, popularity, ratingSize, ratingMean and genre for all movies in
        ratingsReduced if not None, else uses self.ratingsRaw.

    buildTable(similarMovies=None, ratingPredictions=None):
        Adds similarMovies and ratingPredictions series to the ratingsMean DF and sorts by descending similarity.

    getInfo(movieID=None, get=None):
        Returns info as defined by get kwarg from moviesRaw for input movieIDs.

    getMovieID(movie):
        Searches moviesRaw for movieID from input movie info kwarg.

    getMovieRatingsFromMultiLevel(movie, data=None):
        Searches columns for input movie info/ID & returns DF. Searches data kwarg if given, or self.ratingsPivot.

    getUserRatings(userID, data=None, drop=True, buildTable=False):
        Returns all ratings for given userID. Searches either ratingsPivot DF if given in data kwarg, or self.ratingsRaw

    addTestUser(ratings):
        Adds ratings to self.ratingsRaw for userID 0.
    """

    # variables for storing reduced data
    moviesReduced = None
    ratingsReduced = None

    # variables for storing generated data frames
    ratingsPivot = None
    ratingsMean = None
    genreBitfield = None

    def __init__(self, update=False):
        """
        Updates and loads raw data form MovieLens dataset.

        :param update: bool. Determine whether to download and update stored csv files
        """
        if update:
            datasetUrl = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"

            # Download current 27M MovieLens dataset from GroupLens
            print("Downloading...", end='')
            r = requests.get(datasetUrl)
            with open("input/ml-latest.zip", 'wb') as output:
                output.write(r.content)
            print('\33[92m', "Done", '\33[0m')

            # Unzip
            print("Extracting...", end='')
            with zipfile.ZipFile("input/ml-latest.zip", 'r') as z:
                z.extractall("input")
            print('\33[92m', "Done", '\33[0m')

        # Import movies and ratings data to data frames
        print("Loading data...", end='')
        with open("input/ml-latest/movies.csv", encoding="UTF-8") as file:
            self.moviesRaw = pd.read_csv(file)
        with open("input/ml-latest/ratings.csv", encoding="UTF-8") as file:
            self.ratingsRaw = pd.read_csv(file)

        # Split titles and years into separate columns and rearrange column order
        self.moviesRaw['title'] = self.moviesRaw['title'].apply(lambda x: x.strip())
        self.moviesRaw['title'] = self.moviesRaw['title'].apply(lambda x: x.replace('))', ')'))
        self.moviesRaw['year'] = self.moviesRaw['title'].apply(lambda x: x[-5:-1] if x[-5:-1].isdecimal()
                                                                                     and len(x[-5:-1]) == 4 else None)
        self.moviesRaw['title'] = self.moviesRaw['title'].apply(lambda x: x[:-7] if x[-5:-1].isdecimal()
                                                                                    and len(x[-5:-1]) == 4 else x)
        self.moviesRaw = self.moviesRaw[['movieId', 'title', 'year', 'genres']]
        print('\33[92m', "Done", '\33[0m')

    def resetAll(self):
        """
        Resets all self variables bar moviesRaw or ratingsReduced to clear memory.
        """
        self.ratingsReduced = None
        self.moviesReduced = None
        self.ratingsPivot = None
        self.ratingsMean = None
        self.genreBitfield = None

    def filterIDs(self, idType, minRatings=400, maxRatings=1000000, fraction=0.15):
        """
        Filters and returns list of IDs in moviesRaw and ratingsRaw based on min and max rating counts, or a random
        sample.

        :param idType: str. Define which idType to filter. Must be either 'userId', 'movieId',
                            'randomRatings' or 'randomMovies'
        :param minRatings: int, default=400. Min ratings to keep for input idType. Only applies to 'userId'
                                            and 'movieId' idTypes
        :param maxRatings: int, default=1000000. Max ratings to keep for input idType. Only applies to 'userId'
                                                and 'movieId' idTypes
        :param fraction: float, default=0.15. Fraction of random IDs keep. Only applies to 'randomRatings' or
                                            'randomMovies' idTypes
        :return: list. List of IDs that match criteria
        """

        if idType in ['userId', 'movieId']:  # reduce movieIDs or UserIDs to only those between min/maxRatings
            counts = self.ratingsRaw[idType].value_counts()
            counts = counts[counts.between(minRatings, maxRatings, inclusive=True)]
            return list(counts.index)
        elif idType == 'randomRatings':  # randomly select and return a fraction of userIDs present in ratingsRaw
            users = self.ratingsRaw.sample(frac=fraction, random_state=20)['userId']
            return list(users.index)
        elif idType == 'randomMovies':  # randomly select and return a fraction of movieIDs present in ratingsRaw
            movies = self.moviesRaw.sample(frac=fraction, random_state=20)['movieId']
            return list(movies)
        else:
            print('\33[91m', "idType error", '\33[0m', sep='')

    def reduce(self, IDs, idType, dfType):
        """
        Update self.moviesReduced or self.ratingsReduced by input IDs.

        :param IDs: list. List of IDs to keep.
        :param idType: str. Define which column to reduce. Must be either 'userId' or 'movieId'.
        :param dfType: str. Define which DF to reduce. Must be either 'ratings' or 'movies'.
        """
        if idType in ['userId', 'movieId']:  # reduce ratings/movies dataframes with input IDs
            if dfType == 'ratings':
                self.ratingsReduced = self.ratingsRaw[self.ratingsRaw[idType].isin(IDs)].copy()
            elif dfType == 'movies':
                self.moviesReduced = self.moviesRaw[self.moviesRaw[idType].isin(IDs)].copy()
        else:
            print('\33[91m', "idType error", '\33[0m', sep='')

    def buildPivot(self, extraColumns=False, quick=False, printStats=False, update=True):
        """
        Builds pivot DF from moviesReduced & ratingsReduced if not None, else uses self.moviesRaw & self.ratingsRaw.
        Pivots movieIDs as columns, userIDs as index, ratings as values filling missing ratings with nan.

        :param extraColumns: bool, default=False. Build DF with MultiLevel columns for movieID, title, and year.
                                                Columns=movieID only if False
        :param quick: bool, default=False. Build quick pivot DF keeping only IDs with minimum 1000 ratings for
                                            quick diagnostics
        :param printStats: bool, default=False. Print stats on % of IDs retained and sparsity of pivot table
        :param update: bool, default=True. Update self.ratingsPivot with generated DF
        :return: DataFrame. Generated pivot DF
        """
        if quick:  # for diagnostic or debugging tests
            # produces pivot table of users with rating counts between 1000-6000 and movies with min 1000 ratings
            userIDs = self.filterIDs('userId', minRatings=1000, maxRatings=6000)
            movieIDs = self.filterIDs('movieId', minRatings=1000)
            self.reduce(userIDs, 'userId', 'ratings')
            self.reduce(movieIDs, 'movieId', 'movies')

        print("Building movies/ratings pivot df...", end='')
        # check if reduced data frames have been produced, use raw data if not
        movies = self.moviesReduced.copy() if self.moviesReduced is not None else self.moviesRaw.copy()
        ratings = self.ratingsReduced.copy() if self.ratingsReduced is not None else self.ratingsRaw.copy()

        # merge data from movies and ratings data frames and create pivot table of movieIDs against userIDs
        merge = pd.merge(movies.drop(columns=['genres', 'title']), ratings.drop(columns='timestamp'))
        df = merge.pivot_table(index='userId', columns='movieId', values='rating')

        if extraColumns:  # add MultiLevel indexing to the columns to show movieIDs, title, and years
            movieIDs = df.columns
            titles = self.getInfo(movieIDs, get='title')
            years = self.getInfo(movieIDs, get='year')
            columns = [list(df.columns), titles, years]
            df.columns = pd.MultiIndex.from_arrays(columns, names=('movieId', 'title', 'year'))
        print('\33[92m', "Done", '\33[0m')

        if printStats:  # display info on the ratings pivot for users/movies/ratings retained from raw data and sparsity
            totalUsers = len(self.ratingsRaw['userId'].unique())
            totalMovies = len(self.moviesRaw['movieId'].unique())
            totalRatings = len(self.ratingsRaw)
            sparsity = df.isnull().sum() * 100 / len(df)
            print('\33[94m', len(df.index), ' / ', totalUsers, ' users retained (',
                  round(len(df.index) * 100 / totalUsers, 2), '%)', '\33[0m', sep='')
            print('\33[94m', len(df.columns), ' / ', totalMovies, ' movies retained (',
                  round(len(df.columns) * 100 / totalMovies, 2), '%)', '\33[0m', sep='')
            print('\33[94m', df.count().sum(), ' / ', totalRatings, ' ratings retained (',
                  round(df.count().sum() * 100 / totalRatings, 2), '%)', '\33[0m', sep='')
            print('\33[94m', round(sparsity.sum() / len(sparsity), 2), "% sparsity", '\33[0m', sep='')

        if update:  # update internally stored variable for the pivot table with a copy
            self.ratingsPivot = df.copy()
        return df

    def buildGenreBitfield(self, movieIDs=None, update=True):
        """
        Builds DF of genres as a bit field for all movieIDs in moviesReduced if not None, else uses self.moviesRaw.

        :param movieIDs: list, default=None. MoviesIDs to build bit field for
        :param update: bool, default=True. Update self.genreBitfield with generated DF
        :return: DataFrame. Generated genre bit field DF
        """
        # check if reduced data frames have been produced, use raw data if not
        movies = self.moviesReduced.copy() if self.moviesReduced is not None else self.moviesRaw.copy()
        # check if IDs to build bit field for has been passed, use above data frames IDs if not
        movieIDs = list(movies['movieId']) if movieIDs is None else movieIDs

        # define genre column names for bit field
        genreNames = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                      'Mystery', 'IMAX', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
        genreDict = dict(zip(movieIDs, self.getInfo(movieIDs, get='genres')))  # build dictionary of movieIDs and genres
        movieGenres = {}  # create dict to store genre bit fields per movieID

        for movieID in movieIDs:
            genreBits = [0] * len(genreNames)  # initialise an all 0 bit field of length of all genres available
            genres = genreDict[movieID].split('|')  # split pipe-delimited string into individual genres present
            for genre in genres:
                genreBits[genreNames.index(genre)] = 1  # change the corresponding bit in the genreBits bitfield to 1
            movieGenres[movieID] = genreBits  # add bit field to dictionary

        # create data frame from dict with genreNames as column names and drop movies with no genre listed
        genreBitfield = pd.DataFrame.from_dict(movieGenres, orient='index')
        genreBitfield.columns = genreNames
        genreBitfield = genreBitfield[genreBitfield['(no genres listed)'] != 1]
        genreBitfield.drop(columns='(no genres listed)', inplace=True)

        if update:  # update internally stored variable for the genre bitfield with a copy
            self.genreBitfield = genreBitfield.copy()
        return genreBitfield

    def buildMeanRatings(self, update=True, extraInfo=False):
        """
        Builds DF of movieID, title, year, popularity, ratingSize, ratingMean and genre for all movies in
         ratingsReduced if not None, else uses self.ratingsRaw.

        :param update: bool, default=True. Update self.ratingsMean with generated DF
        :param extraInfo: bool, default=False. Add title, year, and genres to DF, or just return movieIDs, popularity,
                                                ratingSize and ratingMean
        :return: DataFrame. Generated mean ratings DF
        """
        # check if reduced data frames have been produced, use raw data if not
        ratings = self.ratingsReduced.copy() if self.ratingsReduced is not None else self.ratingsRaw.copy()
        ratings.drop(['userId', 'timestamp'], axis=1, inplace=True)

        # calculate means and ratings counts for every movie and rename data frames accordingly
        avg = ratings.groupby('movieId').mean().rename({'rating': 'ratingMean'}, axis=1)
        size = ratings.groupby('movieId').count().rename({'rating': 'rating#'}, axis=1)

        # combine into one data frame and sort
        df = size.join(avg)
        df.sort_values(['rating#', 'ratingMean'], ascending=False, inplace=True)
        df.insert(0, 'popularity', np.arange(len(df)) + 1)  # add rankings for popularity

        if extraInfo:  # add information on titles, years and genres
            df.insert(0, 'title', self.getInfo(df.index, get='title'))
            df.insert(1, 'year', self.getInfo(df.index, get='year'))
            df['genres'] = self.getInfo(df.index, get='genres')

        if update:  # update internally stored variable for the mean ratings with a copy
            self.ratingsMean = df.copy()
        return df

    def buildTable(self, similarMovies=None, ratingPredictions=None):
        """
        Adds similarMovies and ratingPredictions series to the ratingsMean DF and sorts by descending similarity.

        :param similarMovies: Series, default=None. MovieIDs and their similarities to reduce ratingsMean DF by
        :param ratingPredictions: Series, default=None. MovieIDs and their predicted ratings to reduce ratingsMean DF by
        :return: DataFrame. Generated mean ratings, similarities and predicted ratings DF
        """
        # check if mean ratings table has been stored, build if not
        ratingsMean = self.buildMeanRatings(extraInfo=True) if self.ratingsMean is None else self.ratingsMean.copy()
        ratingsMean['similarity'] = similarMovies  # append similarities
        ratingsMean['rating'] = ratingPredictions  # append ratings

        # drop movies with missing similarity values and sort by descending similarity
        return ratingsMean.dropna(subset=['similarity']).sort_values('similarity', ascending=False)

    def getInfo(self, movieID=None, get=None):
        """
        Returns info as defined by get kwarg from moviesRaw for input movieIDs.

        :param movieID: int or list, default=None. MovieIDs to search for. If none, uses IDs from self.moviesRaw
        :param get: str, default=None. What info to return. Must be either 'title', 'year', or 'genres'
        :return: list. List of strings of all info found for given IDs
        """
        if movieID is None:  # if no ID given, get all movieIDs from raw data
            movies = self.moviesRaw.set_index('movieId')
        else:  # method only works if movieIDs are in list form; check if ID given as int and convert to list
            movieID = [movieID] if isinstance(movieID, int) else movieID
            # check for movieID in raw data and return df for all IDs found
            movies = self.moviesRaw[self.moviesRaw['movieId'].isin(movieID)].set_index('movieId')
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
        Searches self.moviesRaw for movieID from input movie info arg.

        :param movie: str or list. Movie to find. Must be title (str) or list of [title (str), year (str or int)]
        :return: list. Returns all movies that match search criteria
        """
        allMovies = self.moviesRaw.set_index('movieId')  # set movieID as index from raw data for method to function

        if isinstance(movie, list):  # if movie and/or year defined
            movie = movie[:2]  # reduce list to only first two items.
            movie = [str(item) for item in movie]  # check list items are strings
            df = allMovies[allMovies['title'].str.contains(movie[0])]['year']  # search raw data for the title
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

    def getMovieRatingsFromMultilevel(self, movie, data=None):
        """
        Searches columns for input movie info/ID & returns DF. Searches data kwarg if given, or self.ratingsPivot.

        :param movie: int or list. Movie to find. Must be movieID int, or list of [title (str), year (str or int)]
        :param data: DataFrame, default=None. DF to search. Will search self.ratingsPivot if None
        :return: DataFrame. One column dataframe for the movie. Returns None if movie not found
        """
        data = self.ratingsPivot if data is None else data  # use internally stored ratings pivot table if not given
        df = None

        if isinstance(movie, list):  # if movie title and/or year given, search multilevel index for movie
            title = movie[0]
            year = str(movie[1]) if len(movie) > 1 else ''
            df = data.loc[:, data.columns.get_level_values(1).str.contains(title) &
                             data.columns.get_level_values(2).str.contains(year)]
        elif isinstance(movie, int):  # if movie ID given, search multilevel index for movie
            df = data.loc[:, data.columns.get_level_values(0) == movie]
        return df

    def getUserRatings(self, userID, data=None, drop=True, buildTable=False):
        """
        Returns all ratings for given userID. Searches either ratingsPivot DF if given in data kwarg, or self.ratingsRaw

        :param userID: int. ID to return ratings for
        :param data: DataFrame, default=None. Pivot DF of movies against users to search through
        :param drop: bool, default=True. Drop movies the user has rated that are not in self.ratingsPivot
        :param buildTable: bool, default=False. Add title and year to DF and return this DF
        :return: Series or DataFrame. Returns series of movieIDs and ratings unless buildTable=True
        """
        if data is None:  # search raw data for user if data not given
            userRatings = self.ratingsRaw.loc[self.ratingsRaw['userId'] == userID].set_index('movieId')
            userRatings.drop(['userId', 'timestamp'], axis=1, inplace=True)
        else:  # search data given
            userRatings = data.loc[[userID]].dropna(axis=1).T.rename(columns={userID: 'rating'})

        if drop:  # check user's rated movies against those in ratings pivot and remove ratings not in pivot table
            diff = list(set(userRatings.index) - set(self.ratingsPivot.columns))
            userRatings.drop(diff, inplace=True)

        if buildTable:  # add info for titles and years of movies and return data frame
            userRatings.insert(0, 'title', self.getInfo(userRatings.index, get='title'))
            userRatings.insert(1, 'year', self.getInfo(userRatings.index, get='year'))
            print("User ", userID, "'s ratings:", sep='')
            return userRatings

        return userRatings['rating']  # return series of movieIDs and ratings

    def addTestUser(self, ratings):
        """
        Adds ratings to self.ratingsRaw for userID 0.

        :param ratings: dict. {ID:rating} pairs for ratings to add.
        """
        # create data frame from ratings, rename columns, and add empty columns for append to function correctly
        df = pd.DataFrame.from_dict(ratings, orient='index').reset_index()
        df.columns = ['movieId', 'rating']
        df['userId'] = 0
        df['timestamp'] = 0

        # remove any previous test user ratings and append new test ratings
        self.ratingsRaw = self.ratingsRaw[self.ratingsRaw['userId'] != 0].append(df, ignore_index=True)


class SplitData:
    """
    Splits ratingsPivot into train, test, validation, and LeaveOneOut-CrossValidation sets

    :attribute ratingsPivot: DataFrame. Pivoted DF of- columns: movieId, index: userId, values: ratings.
    :attribute trainSet: DataFrame. Fraction of users from ratingsPivot for training algorithms.
    :attribute testSet_full: DataFrame. The users removed from trainSet.
    :attribute testSet: DataFrame. testSet_full with fraction of ratings removed per user.
    :attribute validationSet: DataFrame. The ratings removed from testSet for validation.
    :attribute LOO_testSet: DataFrame. testSet_full with one rating removed per user for Cross-Validation.
    :attribute LOO_droppedRatings: DataFrame. The ratings removed from LOO_test
    :attribute LOO_droppedMovies: dict. {userID: movieID} pairs for movies removed from LOO_test

    :methods:
    buildTrainTest(test_size=0.2, random_state=20):
        Generate dataframes trainSet, testSet_full, testSet, and validationSet.

    buildLOO():
        Generate LeaveOneOut-CrossValidation sets LOO_test, LOO_droppedRatings, LOO_droppedMovies.

    buildAll():
        Runs buildTrainTest and buildLOO methods and returns self.trainSet and self.testSet.
    """

    def __init__(self, ratingsPivot):
        """
        Stores and initialises self attributes.

        :param ratingsPivot: DataFrame. Pivoted DF of- columns: movieId, index: userId, values: ratings.
        """
        self.ratingsPivot = ratingsPivot
        self.trainSet = None
        self.testSet_full = None
        self.testSet = None
        self.validationSet = None
        self.LOO_testSet = None
        self.LOO_droppedRatings = None
        self.LOO_droppedMovies = {}

    def buildTrainTest(self, testSize=0.2, validationSize=50, randomState=20):
        """
        Generate dataframes trainSet, testSet_full, testSet, and validationSet.

        :param testSize: float or int, default=0.2. Fraction or absolute amount of users to store as testSet
        :param validationSize float or int, default=20. Fraction or absolute amount of ratings to store as validationSet
        :param randomState: int, default=20. Sets the random state of test_train_split for consistent results.
        :return: DatFrames. self.trainSet, self.testSet, self.validationSet
        """
        print("Building train/test/validation split by row...", end='')
        # split data frame by row
        self.trainSet, self.testSet_full = train_test_split(self.ratingsPivot, test_size=testSize,
                                                            random_state=randomState)

        # create and store empty data frames with movieIDs as columns for building test/validation sets
        self.testSet = pd.DataFrame(columns=self.testSet_full.columns)
        self.validationSet = pd.DataFrame(columns=self.testSet_full.columns)

        for i, row in self.testSet_full.iterrows():
            # iterate through each row of testSet_full and split ratings into test and validation sets
            test, validation = train_test_split(row, test_size=validationSize, random_state=randomState)
            self.testSet = self.testSet.append(test)
            self.validationSet = self.validationSet.append(validation)

        print('\33[92m', "Done", '\33[0m')
        return self.trainSet, self.testSet, self.validationSet

    def buildLOO(self):
        """
        Generate LeaveOneOut-CrossValidation sets LOO_test, LOO_droppedRatings, LOO_droppedMovies.

        :return: DataFrames/dict. self.LOO_test, self.LOO_droppedRatings, self.LOO_droppedMovies (dict)
        """
        print("Building LeaveOneOut-CrossValidation data...", end='')
        # store copy of full, un-split test set and create dataframe to store left out ratings
        test = self.testSet_full.copy()
        dropped = pd.DataFrame(np.nan, columns=test.columns, index=test.index)

        for i in test.index:  # iterate through rows of full test set
            ratedMovies = test.loc[i][test.loc[i].notna()].index  # get IDs for user's rated movies
            drop = random.choice(ratedMovies)  # randomly select one ID
            dropped.at[i, drop] = test.at[i, drop]  # add dropped rating to dropped data frame
            self.LOO_droppedMovies[i] = drop  # update internally stored dict with userID, dropped movieID pair
            test.at[i, drop] = np.nan  # drop rating from test set

        # update internally stored data frames with LOO-CV test/validation sets
        self.LOO_testSet = test
        self.LOO_droppedRatings = dropped

        print('\33[92m', "Done", '\33[0m')
        return self.LOO_testSet, self.LOO_droppedRatings, self.LOO_droppedMovies

    def buildAll(self, testSize=0.2, validationSize=50, randomState=20):
        """
        Runs buildTrainTest and buildLOO methods and returns self.trainSet and self.testSet.

        :param testSize: float or int, default=0.2. Fraction or absolute amount of users to store as testSet
        :param validationSize float or int, default=20. Fraction or absolute amount of ratings to store as validationSet
        :param randomState: int, default=20. Sets the random state of test_train_split for consistent results.
        :return: DataFrames. self.trainSet, self.testSet
        """
        # run all split methods and return
        self.buildTrainTest(testSize=testSize, validationSize=validationSize, randomState=randomState)
        self.buildLOO()
        return self.trainSet, self.testSet, self.validationSet, self.LOO_testSet, self.LOO_droppedMovies
