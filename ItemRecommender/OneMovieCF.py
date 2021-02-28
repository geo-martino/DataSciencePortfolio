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

        self.ml = MovieLensData() if ml is None else ml

        if self.ml.ratingsPivot is not None:
            if isinstance(self.ml.ratingsPivot.columns[0], tuple):
                print("Removing MultiLevel index from pivot df...", end='')
                self.ratingsPivot = self.ml.ratingsPivot.droplevel(['title', 'year'], axis=1)
                print('\33[92m', "Done", '\33[0m')
            else:
                self.ratingsPivot = self.ml.ratingsPivot
        else:
            print("Pivot df not found. Building quick pivot df with MinRatings of 1000 for Users and Movies.")
            self.ratingsPivot = self.ml.buildPivot(extraColumns=False, quick=True, update=True)

        try:
            if isinstance(movie[0], int):
                self.movieID = movie[0]
            else:
                self.movieID = self.ml.getMovieID(movie)[0]

            self.year = int(self.ml.getInfo([self.movieID], get='year')[0])
            title = self.ml.getInfo([self.movieID], get='title')[0]

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
        movieRatings = self.ratingsPivot[self.movieID].dropna()

        print("Correlating ratings...", end='')
        similarMovies = self.ratingsPivot.dropna(axis='columns', thresh=2).corrwith(movieRatings).dropna()
        similarMovies.drop(self.movieID, errors='ignore', inplace=True)
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.similarRatings = similarMovies
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')

    def getSimilarGenres(self, update=True):
        """
        Correlates movie's genres with genres of all other movies in self.ml.movieReduced.

        :param update: bool, default=True. Update similarGenres with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        allGenres = self.ml.buildGenreBitfield() if self.ml.genreBitfield is None else self.ml.genreBitfield
        testGenres = allGenres.loc[self.movieID]

        similarities = {}

        print("Correlating genres...", end='')
        for movieID, movieGenres in allGenres.iterrows():
            sumxx, sumxy, sumyy = 0, 0, 0
            for j in range(len(movieGenres)):
                x = movieGenres[j]
                y = testGenres[j]
                sumxx += x * x
                sumyy += y * y
                sumxy += x * y
            similarities[movieID] = sumxy / ((sumxx * sumyy) ** 0.5)
        similarMovies = pd.Series(similarities).drop(self.movieID, errors='ignore')
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.similarGenres = similarMovies
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')

    def getSimilarYears(self, update=True):
        """
        Correlates movie's year with years of all other movies in self.ml.movieReduced.

        :param update: bool, default=True. Update similarYears with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        movies = self.ml.moviesReduced.copy() if self.ml.moviesReduced is not None else self.ml.moviesRaw.copy()
        allYears = movies.set_index('movieId')['year'].dropna().drop(self.movieID).astype(int)
        similarities = {}

        print("Correlating years...", end='')
        for movieID, year in allYears.items():
            similarities[movieID] = np.exp(-abs(year - self.year) / 7.0)
        similarMovies = pd.Series(similarities)
        print('\33[92m', "Done", '\33[0m')

        if update:
            self.similarYears = similarMovies
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')

    def getAll(self, update=True):
        """
        Builds and multiplies the similarities scores from all the other methods.

        :param update: bool, default=True. Update similarAll with generated series.
        :return: Series of movieIDs and similarity scores.
        """
        if self.similarRatings is None:
            self.getSimilarRatings()

        if self.similarGenres is None:
            self.getSimilarGenres()

        if self.similarYears is None:
            self.getSimilarYears()

        similarMovies = self.similarRatings * self.similarGenres * self.similarYears

        if update:
            self.similarAll = similarMovies
        return self.ml.buildTable(similarMovies, 0.0).drop(columns='rating')
