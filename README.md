# Data Science Portfolio

A collection of small projects covering Data Science and Analytical techniques using Python and the Pandas package.

This readme covers the aims of each project and the purposes of the files associated with them. In-depth documentation can be found in each of the files listed below.

## Projects
- [Item Recommender](#item-recommender)
- [Reinforcement Learning](#reinforcement-learning)
- [Other Projects](#other-projects)

## Item Recommender
Designing and testing algorithms to recommend movies for a 'Netflix' style recommendation system using the MovieLens data set from GroupLens. Draws on algorithmic models and statistical methods to design 4 models: item- and user-based collaborative filtering, and item- and user-based k-nearest neighbors. The algorithms generate Top-N recommendations through similarity scores and predicted ratings. These recommendations are then tested using metrics including MAE, RMSE, Coverage, Diversity, Novelty and various types of HitRate.

>The project is described by Jupyter Notebooks in 2 parts: 
>- **main_intro.ipynb** - Covers analysis of the data set to make informed decisions about how best to design the algorithms. Then demonstrates the inner workings of the algorithms and touches on some of their differences.
>- **main_test.ipynb** *(WIP)* - Tests the algorithms using various metrics and discusses which parameter values are most appropriate for the end-user use case of a 'Netflix' style system, and which would be most appropriate for a business looking to increase engagement.
>
>>These notebooks draw on 4 python files which contain the various classes they need to run. Filenames and classes contained within them follow.
>>- **MovieLensData.py**: 
>>	- *MovieLensData*: Downloads, updates, and loads MovieLens data set. Contains methods for reducing the dataset, and building pivot tables, genre bitfields, and mean ratings tables. Also contains methods for getting info on movies or users from the raw or reduced data.
>>	- *SplitData*: Splits the data into train-test-validation splits and LeaveOneOut cross validation for testing quality of the algorithms.
>>- **OneMovieCF.py**:
>>  - *OneMovieCF*: Implements simple, but slow, item-based collaborative filtering for one movie using cosine similarity.
>>- **Algorithms.py**: 
>>	- *BuildCF*: Builds the item- and user- based collaborative filtering similarity matrices, drawing on movie/user ratings, genres, and years to generate similarity scores for all movies
>>	- *Algorithms*: Builds algorithmic models and generates movie recommendations from these models. Contains separate methods for single item, user-item and user-user based recommendations. Also contains methods for getting info on movies or users from the data.
>>- **Tester.py**:
>>	- *Metrics*: Stores results and analyses these results with various metrics. Contains methods for reading and updating results for csv files, and calculating metrics including MAE, RMSE, Coverage, Diversity, Novelty, and various types of HitRate.
>>	- *Tester*: Stores the algorithms to test and runs Metrics object for each algorithm. Contains methods for basic and parameter testing from a test set.

	
## Reinforcement Learning
Using reinforcement learning (or Q-Learning) to teach a program how to win TicTacToe style games of any grid size. Contains classes for playing a game, and building and testing the various parameters of the model builder.

>The tictactoe_main.ipynb Jupyter Notebook demonstrates the classes and tests the parameters of the model builder. This notebook draws on the tictactoe_classes.py file which contains 3 classes and 1 function:
>- **Base**: Creates a base game of TicTacToe for any board size of greater than 3. Capable of playing games with combinations of human and random players. Considers the current game state, determines winners, and presents board states in the console.
>- **Learner**: Trains a model on how to play the game as both player 1 and player 2 with Q-Learning. Also contains a method for playing the game as humans or random players against the trained models.
>- **Tester**: Contains methods for testing the Learner class by the wins, losses, and draws of the tested model against a random player. 
>- **parameter_test** *(function)*: Builds multiple models for varying parameters, and tests the results of these models. Used to assess the impact of the 3 main Q-Learning parameters on the quality of the models.
	
## Other Projects
Smaller projects on data analysis using web data or otherwise.

>- **heart_disease.py** *(WIP)* - Testing accuracy of machine learning classification methods to predict presence of heart disease from patient health data.
>- **covid_daily.py** - Producing graphs on daily cases for any given rolling day rate and list of countries from ECDC COVID-19 data.
>- **jorj_web.py** - Analysing web traffic from jorjmakesmusic.com
>- **spotify_API.py** *(WIP)* - Analysing spotify tags for music.