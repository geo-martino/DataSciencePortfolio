import csv
import math
import os
import sys
import time
from statistics import mean

import pandas as pd
from tqdm.auto import tqdm


class Metrics:
    """
    Stores results and analyses these results with various metrics.

    :attribute csvName: str. Csv filename to store results under.
    :attribute validationSet: DataFrame. The ratings removed from testSet.
    :attribute LOO_droppedMovies: dict. {userID: movieID} pairs for movies removed from LOO_testSet

    :attribute topN: int. Top N movies to sample from results
    :attribute moviesPerPage: int. Movies per page present in final format for meanReciprocalHR metric.

    :attribute ratings: dict. {userID: ratings} results
    :attribute movies: dict. {userID: movies} results
    :attribute similarity: dict. {userID: similarity} results
    :attribute popularity: dict. {userID: popularity} results
    :attribute total: dict. {userID: total number of movies} results

    :attribute metricResults: dict. {metric: value} store for the current algorithm being tested.

    :methods:
    readCSV(csvName, update=False):
        Reads the stored csv with name=csvName and returns it Pandas DataFrame format.

    updateCSV(name, totalIDs=None, time=None, param=None, pValue=None, allParams=None):
        Add results to the csv file with the filename given in initialisation.

    resetResults():
        Resets the dictionaries storing results.

    addResults(allResults, userID):
        Updates the dictionaries storing results.

    allMetrics(name='unknown', dp=10, normalise=True, printResults=False):
        Runs all metrics and stores the values in self.metricResults.

    MAE():
        Mean Absolute Error metric for predicted ratings.

    RMSE():
        Root Mean Squared Error metric for predicted ratings.

    coverage():
        Total movies recommended / Total movies present in train data.

    diversity():
        How diverse are the TopN movies i.e. how similar are the TopN movies to each other.

    novelty():
        How novel are the TopN movies i.e. how much of the 'long-tail' does the algorithm access for its TopN movies.

    LOOHR():
        How many times did the algorithm recommend the LOO-CV movie in its TopN results.

    validationHR():
        How many movies from the validation set of movies left out did the algorithm recommend in its TopN results.

    cumulativeHR():
        ValidationHR but movies are only counted with a rating above the threshold rating defined in initialisation.

    meanReciprocalHR():
        ValidationHR with the movie weighted by a rank of movies per page, i.e. appears on page 1: HR/1, page 2: HR/2,
        page 3: HR/3 etc.

    actualRatingHR(normalise=True):
        Measuring validationHR by rank the user gave that movie from the validation HR.
        Can be normalised by dividing HR by the movies present in the validation set for each user.
    """

    def __init__(self, validation, LOO, topN=None, moviesPerPage=5, thresholdRating=3.0, csvName=None):
        """
        Initialises attributes and creates new csv file for storing results. Checks if one already exists and queries
        user if already exists.

        :param validation: DataFrame. The ratings removed from testSet for validation.
        :param LOO: dict. {userID: movieID} pairs for movies removed from LOO_test
        :param topN: int, default=10. TopN movies to sample in metrics.
        :param moviesPerPage: int, default=5. Movies that will appear every page in final customer view for mrHR metric.
        :param thresholdRating: int or float, default=3.0. Minimum acceptable rating for cumulativeHR metric.
        :param csvName: str, default=None. Csv filename to store results under.
        """
        self.csvName = csvName
        if csvName is not None:  # for creating csv file
            overwrite = True
            if os.path.isfile('output/' + csvName + '.csv'):  # check if file exists and prompt user
                overwrite = input(csvName + '.csv already exists. Update this file (u) or overwrite (o)? ') == 'o'

            if overwrite:  # create csv file
                with open('output/' + csvName + '.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Algorithm', 'Top-N', 'MoviesPerPage', 'ThresholdRating', 'TotalIDs',
                                     'ParameterTest', 'ParameterValue', 'OtherParameters', 'TotalTime/s', 'MeanTime/s',
                                     'MAE', 'RMSE', 'Coverage', 'Diversity', 'Novelty', 'LOOCV_HR', 'Validation_HR',
                                     'Cumulative_HR', 'MeanReciprocal_HR', 'ActualRating_HR'])

        self.validationSet = validation
        self.LOO_droppedMovies = LOO

        self.topN = topN
        self.moviesPerPage = moviesPerPage
        self.thresholdRating = thresholdRating

        self.ratings = {}
        self.movies = {}
        self.similarities = {}
        self.popularities = {}
        self.total = {}

        self.metricResults = {'MAE': 0, 'RMSE': 0, 'coverage': 0, 'diversity': 0, 'novelty': 0, 'looHR': 0,
                              'vHR': 0, 'cHR': 0, 'mrHR': 0, 'arHR': {}}

    def readCSV(self, csvName, update=False):
        """
        Reads the stored csv with name=csvName and returns it Pandas DataFrame format.

        :param csvName: str. Csv filename to read from output/[csvName].csv
        :param update: bool, default=False. Update the filename that the updateCsv method stores results under.
        :return: DataFrame. DF of csv file
        """
        if not os.path.isfile('output/' + csvName + '.csv'):  # if file doesn't exist
            print('\33[91m', "Error: ", csvName, ".csv not found.", '\33[0m', sep='')
            return

        with open('output/' + csvName + '.csv', encoding="UTF-8") as file:  # read into data frame format
            data = pd.read_csv(file)

        if update:  # update stored csv file name
            self.csvName = csvName
        return data

    def updateCSV(self, name, totalTime=None, param=None, pValue=None, allParams=None):
        """
        Add results to the csv file with the filename given in initialisation.

        :param name: str. Name of algorithm being tested.
        :param totalTime: float, default=None. Total test time.
        :param param: str, default=None. Parameter being tested
        :param pValue: int, float or str, default=None. Parameter value of test.
        :param allParams: dict, default=None. All other parameters used not including ID or model.
        """
        if self.csvName is not None:
            r = self.metricResults  # results
            totalIDs = len(self.similarities.keys())  # total number of tests
            meanTime = totalTime / totalIDs if totalIDs != 0 else None  # average time per test

            with open('output/' + self.csvName + '.csv', 'a', newline='') as file:  # add results
                writer = csv.writer(file)
                writer.writerow([name, self.topN, self.moviesPerPage, self.thresholdRating, totalIDs, param, pValue,
                                 allParams, totalTime, meanTime, r['MAE'], r['RMSE'], r['coverage'], r['diversity'],
                                 r['novelty'], r['looHR'], r['vHR'], r['cHR'], r['mrHR'], r['arHR']])

    def resetResults(self):
        """
        Resets the dictionaries storing results.
        """
        self.ratings = {}
        self.movies = {}
        self.similarities = {}
        self.popularities = {}
        self.total = {}

        self.metricResults = {'MAE': 0, 'RMSE': 0, 'coverage': 0, 'diversity': 0, 'novelty': 0, 'looHR': 0,
                              'vHR': 0, 'cHR': 0, 'mrHR': 0, 'arHR': {}}

    def addResults(self, allResults, userID):
        """
        Updates the dictionaries storing results.

        :param allResults: DataFrame. DataFrame of results. Must have MovieIDs as index and columns for
                                        'popularity', 'similarity', and 'rating'
        :param userID: int. UserID to add results for.
        """
        if allResults is None or allResults.empty:  # if no results, do not add
            return

        results = allResults[:self.topN]  # limit results to top-N only
        self.ratings[userID] = list(results['rating'])
        self.movies[userID] = list(results.index)
        self.similarities[userID] = list(results['similarity'])
        self.popularities[userID] = list(results['popularity'])
        self.total[userID] = len(allResults)  # total amount of movies recommended

    def allMetrics(self, normalise=True, printResults=False, name='unknown', dp=10):
        """
        Runs all metrics and stores the values in self.metricResults.

        :param normalise: bool, default=True. Passes to arHR method. Normalises results by dividing by movies present
                                            in validation set per user
        :param printResults: bool, default=False. Show print out of results in console.
        :param name: str, default='unknown'. Name of current algorithm to print if printResults=True
        :param dp: int, default=10. Decimal point to round results to in console if printResults=True.
        """
        if len(self.movies) == 0:  # if no results, return
            return

        # run metrics
        arHR = self.actualRatingHR(normalise=normalise)
        self.metricResults = {
            'MAE': self.MAE(),
            'RMSE': self.RMSE(),
            'coverage': self.coverage(),
            'diversity': self.diversity(),
            'novelty': self.novelty(),
            'looHR': self.LOOHR(),
            'vHR': self.validationHR(),
            'cHR': self.cumulativeHR(),
            'mrHR': self.meanReciprocalHR(),
            'arHR': arHR.to_dict()
        }

        if printResults:
            r = [round(x, dp) for x in list(self.metricResults.values())[:-1]]  # round results

            print('\n', '\33[92;1;4m', 'Results for ', name, ' algorithm', '\n', '\33[0m', sep='')
            print('MAE = {}'.format(r[0]),
                  'RMSE = {}'.format(r[1]),
                  '-----------------',
                  'Coverage = {}'.format(r[2]),
                  'Diversity = {}'.format(r[3]),
                  'Novelty = {}'.format(r[4]),
                  '-----------------',
                  'LOO-CV HitRate = {}'.format(r[5]),
                  'Validation Set HitRate = {}'.format(r[6]),
                  'Cumulative HitRate = {}'.format(r[7]),
                  'Mean Reciprocal HitRate = {}'.format(r[8]), sep='\n')
            print('Actual Rating HitRate:', end='')
            print(round(arHR, dp))

    def MAE(self):
        """
        Mean Absolute Error metric for predicted ratings.

        :return: int. Mean Absolute Error
        """
        MAE = 0
        for userID, recommendedMovies in self.movies.items():
            predictedRatings = pd.Series(self.ratings[userID], recommendedMovies)  # movies recommended
            actualRatings = self.validationSet.loc[userID].dropna()  # movies in validation set
            hits = list(set(recommendedMovies).intersection(actualRatings.index))

            # add mean of the difference between predicted rating and actual rating if there are any hits
            MAE += mean(abs(predictedRatings[hits] - actualRatings[hits])) if hits else 0
        return MAE / len(self.movies)  # normalise with total movies rated

    def RMSE(self):
        """
        Root Mean Squared Error metric for predicted ratings.

        :return: int. Root Mean Squared Error
        """
        RMSE = 0
        for userID, recommendedMovies in self.movies.items():
            predictedRatings = pd.Series(self.ratings[userID], recommendedMovies)  # movies recommended
            actualRatings = self.validationSet.loc[userID].dropna()  # movies in validation set
            hits = list(set(recommendedMovies).intersection(actualRatings.index))

            # difference of predicted rating and actual rating, squared, mean, square root and add if there are any hits
            RMSE += (mean((predictedRatings[hits] - actualRatings[hits]) ** 2)) ** 0.5 if hits else 0
        return RMSE / len(self.movies)  # normalise with total movies rated

    def coverage(self):
        """
        Total movies recommended / Total movies present in train data.

        :return: int. Coverage
        """
        return mean(self.total.values()) / len(self.validationSet.columns)

    def diversity(self):
        """
        How diverse are the TopN movies i.e. how similar are the TopN movies to each other.

        :return: int. Diversity
        """
        diversity = 0
        for similarities in self.similarities.values():
            diversity += 1 - mean(similarities)  # calculate reverse mean of similarities and add
        return diversity / len(self.similarities)  # normalise with total similar movies

    def novelty(self):
        """
        How novel are the TopN movies i.e. how much of the 'long-tail' does the algorithm access for its TopN movies.

        :return: int. Novelty
        """
        novelty = 0
        for popularity in self.popularities.values():
            novelty += mean(popularity) / len(self.validationSet.columns)  # calculate mean popularity per total movies
        return novelty / len(self.popularities)  # normalise with total recommendations

    def LOOHR(self):
        """
        How many times did the algorithm recommend the LOO-CV movie in its TopN results.

        :return: int. LeaveOneOut-CrossValidation Hit Rate
        """
        HR = 0
        for userID, recommendedMovies in self.movies.items():
            HR += 1 if self.LOO_droppedMovies[userID] in recommendedMovies else 0
        return HR / len(self.movies)  # normalise with total movies rated

    def validationHR(self):
        """
        How many movies from the validation set of movies left out did the algorithm recommend in its TopN results.

        :return: int. Validation Set Hit Rate
        """
        HR = 0
        for userID, recommendedMovies in self.movies.items():
            actualMovies = self.validationSet.loc[userID].dropna().index  # movies in validation set
            # no. of hits / total movies recommended
            HR += len(set(actualMovies).intersection(recommendedMovies)) / recommendedMovies
        return HR / len(self.movies)  # normalise with total movies rated

    def cumulativeHR(self):
        """
        ValidationHR but movies are only counted with a rating above the threshold rating defined in initialisation.

        :return: int. Cumulative Hit Rate
        """
        cHR = 0
        for userID, predictedRatings in self.ratings.items():
            # reduce results to only those above threshold rating
            results = [self.movies[userID][i] for i in range(len(predictedRatings))
                       if predictedRatings[i] >= self.thresholdRating]
            actualMovies = self.validationSet.loc[userID].dropna().index  # movies in validation set
            cHR += len(set(actualMovies).intersection(results)) / self.topN  # no. of hits / total movies recommended
        return cHR / len(self.movies)  # normalise with total movies rated

    def meanReciprocalHR(self):
        """
        ValidationHR with the movie weighted by a rank of movies per page, i.e. appears on page 1: HR/1, page 2: HR/2,
        page 3: HR/3 etc.

        :return: int. Mean Reciprocal Hit Rate
        """
        mrHR = 0
        for userID, recommendedMovies in self.movies.items():
            pages = math.ceil(len(recommendedMovies) / self.moviesPerPage)  # total number of pages
            actualMovies = self.validationSet.loc[userID].dropna().index  # movies in validation set

            for page in range(pages):
                minRank = page * self.moviesPerPage  # minimum rank for this page
                thisPage = recommendedMovies[minRank: minRank + self.moviesPerPage]  # movies in this page
                # hit rate over page number and top-N
                mrHR += len(set(actualMovies).intersection(thisPage)) / ((page + 1) * self.topN)
        return mrHR / len(self.movies)  # normalise with total movies rated

    def actualRatingHR(self, normalise=True):
        """
        Measuring validationHR by rank the user gave that movie from the validation HR.
        Can be normalised by dividing HR by the movies present in the validation set for each user.

        :param normalise: bool, default=True. Normalise results by dividing by movies present in validation set per user
        :return: DataFrame. Actual Rating Hit Rate DF
        """
        arHR = pd.Series()  # to store results
        for userID, recommendedMovies in self.movies.items():
            actualRatings = self.validationSet.loc[userID].dropna()  # movies in validation set
            hits = list(set(recommendedMovies).intersection(actualRatings.index))  # recommended movies in validation
            reducedRatings = actualRatings[hits]  # ratings of hits
            weighted = reducedRatings.value_counts()  # count ratings of hits
            weighted /= len(actualRatings) if normalise else 1  # normalise with number of movies in validation set
            arHR = arHR.add(weighted, fill_value=0)
        return arHR.rename('arHR').sort_index().divide(len(self.movies))  # sort and normalise with total movies rated


class Tester:
    """
    Stores algorithms to test and runs Metrics object for each algorithm.

    :attribute algorithms: dict. {algorithm name: [algorithm method, **kwargs]} for storing algorithms to be tested.
    :attribute evaluator: object. Evaluator for calculating metrics and storing/saving results.

    :methods:
    addAlgorithm(name, algorithm, **kwargs):
        Adds an algorithms from the self.algorithms dictionary.

    removeAlgorithm(name):
        Removes an algorithms from the self.algorithms dictionary.

    runParameterTest(testData, param, pRange, testAlgo=None, sampleTest=None, printResults=False):
        Run tests for all algorithms in self.algorithms (unless testAlgo is not None) across all users in testData
        (unless sampleTest is not None) for a given range of values of a parameter.

    runBasicTest(testData, param=None, pValue=None, testAlgo=None, sampleTest=None, printResults=False):
        Run tests for all algorithms in self.algorithms (unless testAlgo is not None) across all users in testData
        (unless sampleTest is not None).
    """

    algorithms = {}  # stores algorithm names, functions, models, and kwargs

    def __init__(self, evaluator):
        """
        Initialises evaluator attribute for calculating metrics and storing/saving results.

        :param evaluator: object. For evaluating algorithms
        """
        self.evaluator = evaluator

    def addAlgorithm(self, name, algorithm, **kwargs):
        """
        Adds an algorithms from the self.algorithms dictionary.

        :param name: str. Algorithm name to add to the self.algorithms dictionary.
        :param algorithm: method. The algorithm method to be used for this algorithm
        :param kwargs: dict. Keyword arguments to be passed to the algorithm method during testing.
        """
        self.algorithms[name] = [algorithm, kwargs]
        print("Added", name)

    def removeAlgorithm(self, name):
        """
        Removes an algorithms from the self.algorithms dictionary.

        :param name: str. Algorithm name to remove from the self.algorithms dictionary.
        """
        del self.algorithms[name]
        print("Removed", name, "- Remaining algorithms:", list(self.algorithms.keys()))

    def runParameterTest(self, testData, param, pRange, testAlgo=None, sampleTest=None, printResults=False):
        """
        Run tests for all algorithms in self.algorithms (unless testAlgo is not None) across all users in testData
        (unless sampleTest is not None) for a given range of values of a parameter.

        :param testData: DataFrame. Pivoted DF of- columns: movieId, index: userId, values: ratings to test.
        :param param: str, default=None. Parameter to test.
        :param pRange: iterator or list, default=None. Range of parameter values to test.
        :param testAlgo: str, default=None. Name of algorithm from self.algorithms to test.
        :param sampleTest: int, default=None. Sample a random number of users from testData.
        :param printResults: bool, default=False. Show print out of results and leave tqdm bar in console.
        """
        bar = tqdm(pRange, desc='Parameter Testing', unit='Parameter', leave=False, file=sys.stdout)  # progress bar
        for pValue in bar:  # run tests
            self.runBasicTest(testData, param, pValue, testAlgo, sampleTest, printResults)

    def runBasicTest(self, testData, param=None, pValue=None, testAlgo=None, sampleTest=None, printResults=False):
        """
        Run tests for all algorithms in self.algorithms (unless testAlgo is not None) across all users in testData
        (unless sampleTest is not None).

        :param testData: DataFrame. Pivoted DF of- columns: movieId, index: userId, values: ratings to test.
        :param param: str, default=None. Enter a parameter to test or to adjust after adding an algorithm.
        :param pValue: int, float or str, default=None. Parameter value if param is not None.
        :param testAlgo: str, default=None. Name of algorithm from self.algorithms to test.
        :param sampleTest: int, default=None. Sample a random number of users from testData.
        :param printResults: bool, default=False. Show print out of results and leave tqdm bar in console.
        :return:
        """
        algorithms = list(self.algorithms.keys()) if testAlgo is None else testAlgo  # algorithm names to be tested
        algorithms = algorithms if isinstance(algorithms, list) else [algorithms]  # check names are in list form

        for nameAlgo in algorithms:
            algorithm = self.algorithms[nameAlgo][0]  # algorithm function
            model = self.algorithms[nameAlgo][1].get('model')  # algorithm model
            kwargs = {k: v for k, v in self.algorithms[nameAlgo][1].items() if k != 'model'}  # drop model from kwargs
            kwargs.pop(param, None)  # drop testing parameter from kwargs

            IDs = testData.index if sampleTest is None else testData.sample(n=sampleTest).index  # all IDs to be tested
            bar = tqdm(IDs, desc='Testing ' + nameAlgo, unit=' Users', leave=printResults, file=sys.stdout)  # progress
            self.evaluator.resetResults()

            t_start = time.perf_counter()  # start timer
            for ID in bar:
                if param is None or 'random' in str(algorithm):  # if no parameter test or algorithm is random control
                    self.evaluator.addResults(algorithm(ID, model, buildTable=True, **kwargs), ID)
                else:  # if parameter test
                    self.evaluator.addResults(algorithm(ID, model, buildTable=True, **kwargs, **{param: pValue}), ID)

            totalTime = time.perf_counter() - t_start
            kwargs.pop('modelType', None)  # drop model type, not needed in csv
            self.evaluator.allMetrics(name=nameAlgo, printResults=printResults)  # run metrics
            self.evaluator.updateCSV(nameAlgo, totalTime, param, pValue, kwargs)  # add results to csv
        return
