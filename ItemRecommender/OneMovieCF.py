import numpy as np
import pandas as pd

from MovieLensData import MovieLensData


class OneMovieCF:
    """
    Builds Collaborative Filtering cosine similarity matrices from ratingsPivot DataFrame.

    :attribute similarRatings: DataFrame. Series of rating correlation scores for the test item.
    :attribute similarGenres: DataFrame. Series of genre correlation scores for the test item.
    :attribute similarYears: DataFrame. Series of year correlation scores for the test item.
    :attribute similarAll: DataFrame. Series of combination of rating, genres and year correlation scores.

    ;attribute movie: list of int. List containing the title and/or year of the movie to test, or int of the movie's ID.
    :attribute ml: object. MovieLensData object for function calls.

    :methods:
    getSimilarRatings(update=True):
        Correlates movie's ratings with ratings of all other movies in self.ratingsPivot.

    getSimilarGenres(update=True):
        Correlates movie's genres with genres of all other movies in self.ml.movieReduced.

    getSimilarYears(update=True):
        Correlates movie's year with years of all other movies in self.ml.movieReduced.

    getAll(update=True):
        Builds and multiplies the similarities scores from all the other methods.
    """

    # variables for storing correlation values as series/data frames
    similarRatings = None
    similarGenres = None
    similarYears = None
    similarAll = None

    def __init__(self, movie, ml=None):
        """
        Initialises objects and searches for the movie to test in the MovieLensData object passed.

        :param movie: list of int. List containing the title and/or year of the movie to test, or int of the movie's ID.
        :param ml: object. MovieLensData object for function calls.
        """
        print("Initialising object...", end='')
        self.ml = MovieLensData() if ml is None else ml  # create new object from MovieLensData class if not given

        if self.ml.ratingsPivot is not None:  # check for multilevel indexing on columns and remove if present
            if isinstance(self.ml.ratingsPivot.columns[0], tuple):
                print("Removing MultiLevel index from pivot df...", end='')
                self.ratingsPivot = self.ml.ratingsPivot.droplevel(['title', 'year'], axis=1)  # drop extra indices
                print('\33[92m', "Done", '\33[0m')
            else:
                self.ratingsPivot = self.ml.ratingsPivot
        else:  # create quick diagnostic pivot table if no ratings pivot table stored
            print("Pivot df not found. Building quick pivot df with MinRatings of 1000 for Users and Movies.")
            self.ratingsPivot = self.ml.buildPivot(extraColumns=False, quick=True, update=True)

        try:
            if isinstance(movie[0], int):  # if movieID given
                self.movieID = movie[0]
            else:  # if movie information given, get movieID
                self.movieID = self.ml.getMovieID(movie)[0]

            self.year = int(self.ml.getInfo([self.movieID], get='year')[0])  # get movie title
            title = self.ml.getInfo([self.movieID], get='title')[0]  # get movie year

            print('\33[92m', "Done", '\33[0m')
            print("Finding recommendations for: ", title, " (", self.year, ") - ID:[", self.movieID, "]", sep='')
        except IndexError:
            print("Error: Movie not found. Try again.")

    def getSimilarRatings(self, update=True):
        """
        Correlates movie's ratings with ratings of all other movies in self.ratingsPivot.

        :param update: bool, default=True. Update similarRatings with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        movieRatings = self.ratingsPivot[self.movieID].dropna()  # find ratings for the test movie

        print("Correlating ratings...", end='')
        # calculate cosine similarity of the movie's ratings with the ratings of all other movies
        # drop test movieID from results
        similarMovies = self.ratingsPivot.dropna(axis='columns', thresh=2).corrwith(movieRatings).dropna()
        similarMovies.drop(self.movieID, errors='ignore', inplace=True)
        print('\33[92m', "Done", '\33[0m')

        if update:  # update internally stored variable for movie rating correlations
            self.similarRatings = similarMovies

        # build data frame of information for similar movies and remove predicted rating column
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')

    def getSimilarGenres(self, update=True):
        """
        Correlates movie's genres with genres of all other movies in self.ml.movieReduced.

        :param update: bool, default=True. Update similarGenres with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        # build genre bitfield if not already stored in ml object and get test movie's genre bit field
        allGenres = self.ml.buildGenreBitfield() if self.ml.genreBitfield is None else self.ml.genreBitfield
        testGenres = allGenres.loc[self.movieID]

        similarities = {}  # for storing movieID and similarity scores

        print("Correlating genres...", end='')
        for movieID, movieGenres in allGenres.iterrows():  # iterate through each movie's genre bit field and correlate
            sum_xx, sum_xy, sum_yy = 0, 0, 0
            for j in range(len(movieGenres)):  # iterate through each genre type and sum parts of correlation equation
                x = movieGenres[j]
                y = testGenres[j]
                sum_xx += x * x
                sum_yy += y * y
                sum_xy += x * y

            # calculate correlation to test movie and update similarities dictionary
            similarities[movieID] = sum_xy / ((sum_xx * sum_yy) ** 0.5)

        # convert similarities dictionary to series and drop test movie
        similarMovies = pd.Series(similarities).drop(self.movieID, errors='ignore')
        print('\33[92m', "Done", '\33[0m')

        if update:  # update internally stored variable for genre correlations
            self.similarGenres = similarMovies

        # build data frame of information for similar movies and remove predicted rating column
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')

    def getSimilarYears(self, update=True):
        """
        Correlates movie's year with years of all other movies in self.ml.movieReduced.

        :param update: bool, default=True. Update similarYears with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        # check for reduced data and use raw data if not found
        movies = self.ml.moviesReduced.copy() if self.ml.moviesReduced is not None else self.ml.moviesRaw.copy()
        # get year strings for movies and recast as integers, dropping the test movieID
        allYears = movies.set_index('movieId')['year'].dropna().drop(self.movieID).astype(int)

        similarities = {}  # for storing movieID and similarity scores

        print("Correlating years...", end='')
        for movieID, year in allYears.items():  # calculate similarities according to the exponential function
            similarities[movieID] = np.exp(-abs(year - self.year) / 7.0)
        similarMovies = pd.Series(similarities)  # convert dictionary to series
        print('\33[92m', "Done", '\33[0m')

        if update:  # update internally stored variable for year correlations
            self.similarYears = similarMovies

        # build data frame of information for similar movies and remove predicted rating column
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')

    def getAll(self, update=True):
        """
        Builds and multiplies the similarities scores from all the other methods.

        :param update: bool, default=True. Update similarAll with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        # build similarities if not already done
        if self.similarRatings is None:
            self.getSimilarRatings()

        if self.similarGenres is None:
            self.getSimilarGenres()

        if self.similarYears is None:
            self.getSimilarYears()

        # multiply similarities together to generate combined similarity scores
        similarMovies = self.similarRatings * self.similarGenres * self.similarYears

        if update:  # update internally stored variable for combined similarities
            self.similarAll = similarMovies

        # build data frame of information for similar movies and remove predicted rating column
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')
