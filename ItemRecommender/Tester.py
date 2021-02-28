import csv
import math
import os
import sys
import time
from statistics import mean

import pandas as pd
from IPython.display import display
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
        if csvName is not None:
            overwrite = True
            if os.path.isfile('output/' + csvName + '.csv'):
                overwrite = input(csvName + '.csv already exists. Update this file (u) or overwrite (o)? ') == 'o'

            if overwrite:
                with open('output/' + csvName + '.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Algorithm', 'TotalIDs', 'MeanTime/s', 'ParameterTest', 'ParameterValue',
                                     'OtherParameters', 'MAE', 'RMSE', 'Coverage', 'Diversity', 'Novelty',
                                     'LOOCV_HR', 'Validation_HR', 'Cumulative_HR', 'MeanReciprocal_HR',
                                     'ActualRating_HR'])

        self.validationSet = validation
        self.LOO_droppedMovies = LOO

        self.topN = topN
        self.moviesPerPage = moviesPerPage
        self.thresholdRating = thresholdRating

        self.ratings = {}
        self.movies = {}
        self.similarity = {}
        self.popularity = {}
        self.total = {}

        self.metricResults = {}

    def readCSV(self, csvName, update=False):
        """
        Reads the stored csv with name=csvName and returns it Pandas DataFrame format.

        :param csvName: str. Csv filename to read from output/[csvName].csv
        :param update: bool, default=False. Update the filename that the updateCsv method stores results under.
        :return: DataFrame. DF of csv file
        """
        if not os.path.isfile('output/' + csvName + '.csv'):
            print('\33[91m', "Error: ", csvName, ".csv not found.", '\33[0m', sep='')
            return
        with open('output/' + csvName + '.csv', encoding="UTF-8") as file:
            data = pd.read_csv(file)
        if update:
            self.csvName = csvName
        return data

    def updateCSV(self, name, totalIDs=None, meanTime=None, param=None, pValue=None, allParams=None):
        """
        Add results to the csv file with the filename given in initialisation.

        :param name: str. Name of algorithm being tested.
        :param totalIDs: int, default=None. Total number of IDs tested.
        :param meanTime: float, default=None. Average time taken per ID in test.
        :param param: str, default=None. Parameter being tested
        :param pValue: int, float or str, default=None. Parameter value of test.
        :param allParams: dict, default=None. All other parameters used not including ID or model.
        """
        if self.csvName is not None:
            r = self.metricResults
            with open('output/' + self.csvName + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, totalIDs, meanTime, param, pValue, allParams,
                                 r['MAE'], r['RMSE'], r['coverage'], r['diversity'], r['novelty'],
                                 r['looHR'], r['vHR'], r['cHR'], r['mrHR'], r['arHR']])

    def resetResults(self):
        """
        Resets the dictionaries storing results.
        """
        self.ratings = {}
        self.movies = {}
        self.similarity = {}
        self.popularity = {}
        self.total = {}

    def addResults(self, allResults, userID):
        """
        Updates the dictionaries storing results.

        :param allResults: DataFrame. DataFrame of results. Must have MovieIDs as index and columns for
                                        'popularity', 'similarity', and 'rating'
        :param userID: int. UserID to add results for.
        """
        results = allResults[:self.topN]
        self.ratings[userID] = list(results['rating'])
        self.movies[userID] = list(results.index)
        self.similarity[userID] = list(results['similarity'])
        self.popularity[userID] = list(results['popularity'])
        self.total[userID] = len(allResults)

    def allMetrics(self, normalise=True, printResults=False, name='unknown', dp=10):
        """
        Runs all metrics and stores the values in self.metricResults.

        :param normalise: bool, default=True. Passes to arHR method. Normalises results by dividing by movies present
                                            in validation set per user
        :param printResults: bool, default=False. Show print out of results in console.
        :param name: str, default='unknown'. Name of current algorithm to print if printResults=True
        :param dp: int, default=10. Decimal point to round results to in console if printResults=True.
        """
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
            'arHR': arHR.to_dict()['arHR']
        }

        if printResults:
            r = [round(x, dp) for x in list(self.metricResults.values())[:-1]]

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
            display(round(arHR, dp))

    def MAE(self):
        """
        Mean Absolute Error metric for predicted ratings.

        :return: int. Mean Absolute Error
        """
        MAE = 0
        for userID, recommendedMovies in self.movies.items():
            predictedRatings = pd.Series(self.ratings[userID], recommendedMovies)
            actualRatings = self.validationSet.loc[userID].dropna()
            hits = list(set(recommendedMovies).intersection(actualRatings.index))
            MAE += mean(abs(predictedRatings[hits] - actualRatings[hits])) if hits else 0
        return MAE / len(self.movies)

    def RMSE(self):
        """
        Root Mean Squared Error metric for predicted ratings.

        :return: int. Root Mean Squared Error
        """
        RMSE = 0
        for userID, recommendedMovies in self.movies.items():
            predictedRatings = pd.Series(self.ratings[userID], recommendedMovies)
            actualRatings = self.validationSet.loc[userID].dropna()
            hits = list(set(recommendedMovies).intersection(actualRatings.index))
            RMSE += (mean((predictedRatings[hits] - actualRatings[hits]) ** 2)) ** 0.5 if hits else 0
        return RMSE / len(self.movies)

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
        for similarities in self.similarity.values():
            diversity += 1 - mean(similarities)
        return diversity / len(self.similarity)

    def novelty(self):
        """
        How novel are the TopN movies i.e. how much of the 'long-tail' does the algorithm access for its TopN movies.

        :return: int. Novelty
        """
        novelty = 0
        for popularity in self.popularity.values():
            novelty += mean(popularity) / len(self.validationSet.columns)
        return novelty / len(self.popularity)

    def LOOHR(self):
        """
        How many times did the algorithm recommend the LOO-CV movie in its TopN results.

        :return: int. LeaveOneOut-CrossValidation Hit Rate
        """
        HR = 0
        for userID, recommendedMovies in self.movies.items():
            HR += 1 if self.LOO_droppedMovies[userID] in recommendedMovies else 0
        return HR / len(self.movies)

    def validationHR(self):
        """
        How many movies from the validation set of movies left out did the algorithm recommend in its TopN results.

        :return: int. Validation Set Hit Rate
        """
        HR = 0
        for userID, recommendedMovies in self.movies.items():
            actualMovies = self.validationSet.loc[userID].dropna().index
            HR += len(set(actualMovies).intersection(recommendedMovies)) / self.topN
        return HR / len(self.movies)

    def cumulativeHR(self):
        """
        ValidationHR but movies are only counted with a rating above the threshold rating defined in initialisation.

        :return: int. Cumulative Hit Rate
        """
        cHR = 0
        for userID, predictedRatings in self.ratings.items():
            results = [self.movies[userID][i] for i in range(len(predictedRatings))
                       if predictedRatings[i] >= self.thresholdRating]
            actualMovies = self.validationSet.loc[userID].dropna().index
            cHR += len(set(actualMovies).intersection(results)) / self.topN
        return cHR / len(self.movies)

    def meanReciprocalHR(self):
        """
        ValidationHR with the movie weighted by a rank of movies per page, i.e. appears on page 1: HR/1, page 2: HR/2,
        page 3: HR/3 etc.

        :return: int. Mean Reciprocal Hit Rate
        """
        mrHR = 0
        for userID, recommendedMovies in self.movies.items():
            pages = math.ceil(len(recommendedMovies) / self.moviesPerPage)
            for page in range(pages):
                minRank = page * self.moviesPerPage
                actualMovies = self.validationSet.loc[userID].dropna().index
                thisPage = recommendedMovies[minRank: minRank + self.moviesPerPage]
                mrHR += len(set(actualMovies).intersection(thisPage)) / ((page + 1) * self.topN)
        return mrHR / len(self.movies)

    def actualRatingHR(self, normalise=True):
        """
        Measuring validationHR by rank the user gave that movie from the validation HR.
        Can be normalised by dividing HR by the movies present in the validation set for each user.

        :param normalise: bool, default=True. Normalise results by dividing by movies present in validation set per user
        :return: DataFrame. Actual Rating Hit Rate DF
        """
        arHR = pd.Series()
        for userID, recommendedMovies in self.movies.items():
            actualRatings = self.validationSet.loc[userID].dropna()
            hits = list(set(recommendedMovies).intersection(actualRatings.index))
            reducedRatings = actualRatings[hits]
            weighted = reducedRatings.value_counts()
            weighted /= len(actualRatings) if normalise else 1
            arHR = arHR.add(weighted, fill_value=0)
        return pd.DataFrame(arHR / len(self.movies), columns=['arHR']).rename_axis(index='rating').sort_index()


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

    algorithms = {}

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
        bar = tqdm(pRange, desc='Parameter Testing', unit='Parameter', leave=False, file=sys.stdout)
        for pValue in bar:
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
        if testAlgo is None:
            for nameAlgo in self.algorithms:
                algorithm = self.algorithms[nameAlgo][0]
                model = self.algorithms[nameAlgo][1].get('model')
                kwargs = {k: v for k, v in self.algorithms[nameAlgo][1].items() if k != 'model'}
                kwargs.pop(param, None)

                IDs = testData.index if sampleTest is None else testData.sample(n=sampleTest).index
                bar = tqdm(IDs, desc='Testing ' + nameAlgo, unit=' Users', leave=printResults, file=sys.stdout)
                self.evaluator.resetResults()

                t_start = time.perf_counter()
                for ID in bar:
                    if param is None or 'random' in str(algorithm):
                        self.evaluator.addResults(algorithm(ID, model, **kwargs), ID)
                    else:
                        self.evaluator.addResults(algorithm(ID, model, **kwargs, **{param: pValue}), ID)

                meanTime = (time.perf_counter() - t_start) / len(IDs)
                self.evaluator.allMetrics(name=nameAlgo, printResults=printResults)
                self.evaluator.updateCSV(nameAlgo, len(IDs), meanTime, param, pValue, kwargs)
            return
        else:
            algorithm = self.algorithms[testAlgo][0]
            model = self.algorithms[testAlgo][1].get('model')
            kwargs = {k: v for k, v in self.algorithms[testAlgo][1].items() if k != 'model'}
            kwargs.pop(param, None)

            IDs = testData.index if sampleTest is None else testData.sample(n=sampleTest).index
            bar = tqdm(IDs, desc='Testing ' + testAlgo, unit=' Users', leave=printResults, file=sys.stdout)
            self.evaluator.resetResults()

            t_start = time.perf_counter()
            for ID in bar:
                if param is None or 'random' in str(algorithm):
                    self.evaluator.addResults(algorithm(ID, model, **kwargs), ID)
                else:
                    self.evaluator.addResults(algorithm(ID, model, **kwargs, **{param: pValue}), ID)

            meanTime = (time.perf_counter() - t_start) / len(IDs)
            self.evaluator.allMetrics(name=testAlgo, printResults=printResults)
            self.evaluator.updateCSV(testAlgo, len(IDs), meanTime, param, pValue, kwargs)
            return
