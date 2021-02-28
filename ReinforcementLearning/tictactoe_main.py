#!/usr/bin/env python
# coding: utf-8

# # TicTacToe Reinforcement Learning
# 
# Reinforcement Learning is an unsupervised method of machine learning which aims to effectively teach a program how to accomplish a simple task. The program accomplishes a task over many trials or epochs, with a certain percentage of these epochs accomplished at random, with the rest accomplished by looking up the best outcome from a table, dictionary or list of some kind. This table is initially empty, with the program updating the stored values after each epoch according to a learning formula. This formula factors in some kind of numerical reward or punishment for accomplishing the task well or badly respectively, and updates the table. After a large number of epochs, the program should have generated a set of values for the best possible outcomes in every situation it has seen, and avoid taking actions that will lead to negative outcomes. Such learning works very well in very limited user cases, such as the game of TicTacToe.
# 
# This notebook demonstrates 3 main classes designed for the above purpose: Base, which runs a simple game of TicTacToe containing various methods to perform moves and report on the winner of a game; Learner, which contains methods to make random and learned moves in the Base class, whilst learning from the moves it has for both players 1 and 2; and Tester, which accesses the Learner class to perform this learning over many epochs. Finally, the function parameter_test can be used to tune the parameters of the Q-Learning formula and other parameters.
# 
# This notebook accesses and demonstrates the objects defined in tictactoe_classes.py to teach a program how to win games of TicTacToe using reinforcement learning or Q-learning. More documentation on how this program works can found in tictactoe_classes.py. Here, we primarily access the Tester class and parameter_test function to produce results for varying parameters of the learning process.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline

from tictactoe_classes import Base, Tester, parameter_test

sns.set()


# ## Base game demonstration
# 
# The Base object is set up with various functions that can run depending on the players present and the parameters
# set. The object can take two arguments: one for size, and one for a dictionary of players present. I have introduced
# a grid size lower than the minimum to demonstrate the error management of the object.
# 
# The following demonstrates a game against a human player and random player of grid size 3x3 with error management.

# In[2]:


basicGame_3 = Base(2)
basicGame_3.basic_play()


# The object can be run for any grid size, and be simulated by random moves.

# In[3]:


basicGame_4 = Base(4, {1: 'r', -1: 'r'})
basicGame_4.basic_play()


# ## Initial model test
# 
# The model builder can be accessed by running the Tester class which sets up a game of the size parsed using the Base object,
# passing parameters for exploration rate, learn rate, discount rate, and the rewards for winning, drawing and
# losing. The values below are the same as those set by default. We build our model by parsing how many games (epochs)
# we want the builder to run for.
# 
# The program starts by exploring random moves for the first 100 games. It builds a dictionary of all the states of
# moves it has played so far and the Q-value associated with it. If a Q-value is already present in the dictionary,
# it will factor this in by a factor of (1 - learn rate). A discount factor is applied to the Q-value generated from
# the current epoch, and added to the reward rates given earlier. The new Q-value is therefore calculated as:
# 
# new_q = (1 - learn_rate) * prev_q + learn_rate * (reward + discount_factor * next_max_q)
# 
# This new Q-value is then updated in the dictionary for later referencing.
# 
# After 100 epochs, the program then factors in the exploration rate i.e. an exploration of 0.3 = 30% random moves.
# The rest are taken by either finding the next move with the highest Q-value from the dictionary, or, if one cannot
# be found, taking a random move.

# In[4]:


exploration = 0.3
learn = 0.7
discount = 0.2
r_win = 2
r_draw = -1
r_lose = -3

tester = Tester(N=3, exp=exploration, lrn=learn, dsc=discount, r_win=r_win, r_draw=r_draw, r_lose=r_lose)


# In[5]:


epochs = 30000

tester.build_model(epochs);


# The learner stores data on the wins, draws and losses (w/d/l), and after how many turns these conditions occurred to
# gauge how well the program has learned to play the game. First we run a control test of random moves for both players.
# 
# Both tests require a player list in the form ['b', 'r'] to run and take various other inputs as arguments. More
# information is present in the docstrings of tictactoe.py.

# In[6]:


trials = 10
games_per_trial = 1000

P1_control_wdl, P1_control_turns = tester.test_control(player_list=['b', 'r'], moves_list=[6, 8])
P2_control_wdl, P2_control_turns = tester.test_control(player_list=['r', 'b'], moves_list=[5, 7])


# Now we test the program for both player 1 and 2 bots. 

# In[7]:


P1_test_wdl, P1_test_turns = tester.test_model(player_list=['b', 'r'], moves_list=[6, 8])
P2_test_wdl, P2_test_turns = tester.test_model(player_list=['r', 'b'], moves_list=[5, 7])


# We can compare the percentages of wins for each test to see how much the bot improves compared to random moves.

# In[8]:


P1_control_win = (sum(P1_control_wdl['wins'])/(games_per_trial * trials)) * 100
P1_test_win = (sum(P1_test_wdl['wins'])/(games_per_trial * trials)) * 100
P2_control_win = (sum(P2_control_wdl['wins'])/(games_per_trial * trials)) * 100
P2_test_win = (sum(P2_test_wdl['wins'])/(games_per_trial * trials)) * 100

print("Player 1: Control Win % =", P1_control_win, ", Player 1: Test Win % =", P1_test_win)
print("Player 2: Control Win % =", P2_control_win, ", Player 2: Test Win % =", P2_test_win)


# Extracting the w/d/l stats from the model we can gauge the statistical significance of the test by conducting a
# simple A/B test.

# In[9]:


print(stats.ttest_ind(P1_test_wdl['wins'], P1_control_wdl['wins']))
print(stats.ttest_ind(P2_test_wdl['wins'], P1_control_wdl['wins']))


# A high t-value and low p-value (<0.05 or even <0.01) there is less than a 5% or 1% probability the null hypothesis
# is correct i.e. that the results are random. Our results are therefore not random.

# ## Parameter testing
# 
# In the previous run we used 30000 epochs, but we can find a 'sweet spot' for the epoch number by seeing how the
# amounts of moves learned changes with varying epoch numbers, and comparing this to the wins.
# 
# tictactoe.py also contains a parameter_test function which allows us to iterate over the tester function for
# increasing epochs. This tests the effect of changing the exploration rate, learn rate, and discount
# factor has on the moves learned and wins.
# 
# First, define the interpolate function to use which will return 3 data frames: one for moves learned un-interpolated,
# and 2 for the interpolated values of P1 and P2 wins. As there is a great deal of data generated, a graph plotted with this raw data will make it very hard to distinguish trends. As such, we interpolate the data to 'smooth out' the data plotted.

# In[10]:


def interpolate_df(start_epc, max_epc, inter_epc, moves_learned, P1_wins, P2_wins):
    x_values = list(range(start_epc, max_epc+1, inter_epc))

    moves_df = pd.DataFrame.from_dict(moves_learned)
    moves_df.index = x_values

    xnew = np.linspace(start, max_epochs, 400)
    ynew_P1 = {}
    ynew_P2 = {}

    for i in range(0, 11, 1):
        i = i/10
        spl_P1 = make_interp_spline(x_values, P1_wins[i], k=2)
        spl_P2 = make_interp_spline(x_values, P2_wins[i], k=2)
        ynew_P1[i] = spl_P1(xnew)
        ynew_P2[i] = spl_P2(xnew)

    P1_df = pd.DataFrame.from_dict(ynew_P1)
    P2_df = pd.DataFrame.from_dict(ynew_P2)
    P1_df.index = xnew
    P2_df.index = xnew
    
    return moves_df, P1_df, P2_df


# ### Testing the exploration rate
# 
# The exploration rate governs the ratio of random moves to learned moves the program takes in the learning phase. Low exploration rates cause the game to take very little chances in its learning, whereas high rates make the program focus less on testing the Q-values it has already stored for a given situation.
# 
# Here we set the default values for each test and begin testing for the exploration rate. 

# In[11]:


start = 500
max_epochs = 20000
interval = 500
exp = 0.9
lrn = 0.1
dsc = 0.9

test = 'e'

moves_e, P1_e, P2_e = parameter_test(test=test, lrn=lrn, dsc=dsc, start_epc=start, max_epc=max_epochs, inter_epc=interval)
moves_edf, P1_edf, P2_edf = interpolate_df(start, max_epochs, interval, moves_e, P1_e, P2_e)


# In[12]:


ax = moves_edf.plot(figsize=(16, 8))
ax.set_title('Exploration Rates: Moves Learned (lr=' + str(lrn) + ', df=' + str(dsc) + ')', fontsize= 16)
ax.set_xlabel("Epochs", fontsize=14)
ax.set_ylabel("Moves Learned", fontsize=14)
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

fig = ax.get_figure()
fig.savefig('tictactoe_graphs/Exploration Rate - Moves.jpg', format='jpeg', dpi=200, bbox_inches='tight')
plt.show()


# In[13]:


fig, ax = plt.subplots(4,2, figsize=(20, 15), sharex=True, sharey=True)
ax[0][0] = P1_edf.iloc[:,0:3].plot(title='Exploration Rates: P1 Wins (lr=' + str(lrn) + ', df=' + str(dsc) + ')', ax=ax[0][0])
ax[1][0] = P1_edf.iloc[:,3:6].plot(ax=ax[1][0])
ax[2][0] = P1_edf.iloc[:,6:9].plot(ax=ax[2][0])
ax[2][0] = P1_edf.iloc[:,9:11].plot(ax=ax[3][0])

ax[0][1] = P2_edf.iloc[:,0:3].plot(title='Exploration Rates: P2 Wins (lr=' + str(lrn) + ', df=' + str(dsc) + ')', ax=ax[0][1])
ax[1][1] = P2_edf.iloc[:,3:6].plot( ax=ax[1][1])
ax[2][1] = P2_edf.iloc[:,6:9].plot(ax=ax[2][1])
ax[2][0] = P2_edf.iloc[:,9:11].plot(ax=ax[3][1])

ax[1][0].set_ylabel("Wins", fontsize=14)
ax[3][0].set_xlabel("Epochs", fontsize=14)
ax[3][1].set_xlabel("Epochs", fontsize=14)

plt.show()
fig.savefig('tictactoe_graphs/Exploration Rate - Wins.jpg', format='jpeg', dpi=200, bbox_inches='tight')


# ### Testing the learn rate
# 
# The learning rate governs the balance of the consideration of the new Q-value vs. the old Q-value. Higher learning rates shift the balance to the new Q-value, and lower learning rates cause the program to update the stored Q-values by smaller fractions of the new value. Low learning rates cause the program to learn from new epochs very slowly, whereas very high values cause the program to effectively only consider the current epochs Q-values.

# In[14]:


test = 'l'

moves_l, P1_l, P2_l = parameter_test(test=test, exp=exp, dsc=dsc, start_epc=start, max_epc=max_epochs, inter_epc=interval)
moves_ldf, P1_ldf, P2_ldf = interpolate_df(start, max_epochs, interval, moves_l, P1_l, P2_l)


# In[15]:


ax = moves_ldf.plot(figsize=(16, 8))
ax.set_title('Learn Rates: Moves Learned (er=' + str(exp) + ', df=' + str(dsc) + ')', fontsize= 16)
ax.set_xlabel("Epochs", fontsize=14)
ax.set_ylabel("Moves Learned", fontsize=14)
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

fig = ax.get_figure()
fig.savefig('tictactoe_graphs/Learn Rate - Moves.jpg', format='jpeg', dpi=200, bbox_inches='tight')
plt.show()


# In[16]:


fig, ax = plt.subplots(4,2, figsize=(20, 15), sharex=True, sharey=True)
ax[0][0] = P1_ldf.iloc[:,0:3].plot(title='Learn Rates: P1 Wins (er=' + str(exp) + ', df=' + str(dsc) + ')', ax=ax[0][0])
ax[1][0] = P1_ldf.iloc[:,3:6].plot(ax=ax[1][0])
ax[2][0] = P1_ldf.iloc[:,6:9].plot(ax=ax[2][0])
ax[2][0] = P1_ldf.iloc[:,9:11].plot(ax=ax[3][0])

ax[0][1] = P2_ldf.iloc[:,0:3].plot(title='Learn Rates: P2 Wins (er=' + str(exp) + ', df=' + str(dsc) + ')', ax=ax[0][1])
ax[1][1] = P2_ldf.iloc[:,3:6].plot( ax=ax[1][1])
ax[2][1] = P2_ldf.iloc[:,6:9].plot(ax=ax[2][1])
ax[2][0] = P2_ldf.iloc[:,9:11].plot(ax=ax[3][1])

ax[1][0].set_ylabel("Wins", fontsize=14)
ax[3][0].set_xlabel("Epochs", fontsize=14)
ax[3][1].set_xlabel("Epochs", fontsize=14)

plt.show()
fig.savefig('tictactoe_graphs/Learn Rate - Wins.jpg', format='jpeg', dpi=200, bbox_inches='tight')


# ### Testing the discount factor
# 
# The discount factor affects the percentage of the next move's maximum Q-value is considered for the current move's Q-value.

# In[17]:


test = 'd'

moves_d, P1_d, P2_d = parameter_test(test=test, exp=exp, lrn=lrn, start_epc=start, max_epc=max_epochs, inter_epc=interval)
moves_ddf, P1_ddf, P2_ddf = interpolate_df(start, max_epochs, interval, moves_d, P1_d, P2_d)


# In[18]:


ax = moves_ddf.plot(figsize=(16, 8))
ax.set_title('Discount Factors: Moves Learned (er=' + str(exp) + ', lr=' + str(lrn) + ')', fontsize= 16)
ax.set_xlabel("Epochs", fontsize=14)
ax.set_ylabel("Moves Learned", fontsize=14)
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

fig = ax.get_figure()
fig.savefig('tictactoe_graphs/Discount Rate - Moves.jpg', format='jpeg', dpi=200, bbox_inches='tight')
plt.show()


# In[19]:


fig, ax = plt.subplots(4,2, figsize=(20, 15), sharex=True, sharey=True)
ax[0][0] = P1_ddf.iloc[:,0:3].plot(title='Discount Factors: P1 Wins (er=' + str(exp) + ', lr=' + str(lrn) + ')', ax=ax[0][0])
ax[1][0] = P1_ddf.iloc[:,3:6].plot(ax=ax[1][0])
ax[2][0] = P1_ddf.iloc[:,6:9].plot(ax=ax[2][0])
ax[2][0] = P1_ddf.iloc[:,9:11].plot(ax=ax[3][0])

ax[0][1] = P2_ddf.iloc[:,0:3].plot(title='Discount Factors: P2 Wins (er=' + str(exp) + ', lr=' + str(lrn) + ')', ax=ax[0][1])
ax[1][1] = P2_ddf.iloc[:,3:6].plot( ax=ax[1][1])
ax[2][1] = P2_ddf.iloc[:,6:9].plot(ax=ax[2][1])
ax[2][0] = P2_ddf.iloc[:,9:11].plot(ax=ax[3][1])

ax[1][0].set_ylabel("Wins", fontsize=14)
ax[3][0].set_xlabel("Epochs", fontsize=14)
ax[3][1].set_xlabel("Epochs", fontsize=14)

plt.show()
fig.savefig('tictactoe_graphs/Discount Factor - Wins.jpg', format='jpeg', dpi=200, bbox_inches='tight')


# From these results we can see that higher exploration rates and lower learning rates are preferred for the model. The 
# discount factor appears to have little impact on the accuracy of the model however, higher values marginally improve 
# it. Good values for the 3 parameters appear to be: 
# - exploration rate = 0.9
# - learn rate = 0.1
# - discount factor = 1.0
# 
# Given these values, the accuracy of the tests stops being significantly affected by further epochs at around 15000 epochs. 

# ### Re-running the model test
# 
# Using these values, we can run our tests again to get the maximum accuracy capable for this model.

# In[20]:


exploration = 0.9
learn = 0.1
discount = 1.0
epochs = 15000
trials = 10
games_per_trial = 1000

tester = Tester(N=3, exp=exploration, lrn=learn, dsc=discount)
tester.build_model(epochs)

P1_test_wdl, P1_test_turns = tester.test_model(player_list=['b', 'r'], moves_list=[6, 8])
P2_test_wdl, P2_test_turns = tester.test_model(player_list=['r', 'b'], moves_list=[5, 7])

P1_control_win = (sum(P1_control_wdl['wins'])/(games_per_trial * trials)) * 100
P1_test_win = (sum(P1_test_wdl['wins'])/(games_per_trial * trials)) * 100
P2_control_win = (sum(P2_control_wdl['wins'])/(games_per_trial * trials)) * 100
P2_test_win = (sum(P2_test_wdl['wins'])/(games_per_trial * trials)) * 100

print("Player 1: Control Win % =", P1_control_win, ", Player 1: Test Win % =", P1_test_win)
print("Player 2: Control Win % =", P2_control_win, ", Player 2: Test Win % =", P2_test_win)


# ## Conclusion

# In this notebook, we've seen an implementation of a human/random/bot playable TicTacToe game for board sizes n x n via the Base class. We then called the Tester class to create an object with the purpose of learning how to play a 3 x 3 TicTacToe game using reinforcement learning. This process involves playing a set number of games (or epochs) with a certain exploration governing the ratio of games with random moves and games that use the best move from a set of stored moves. These stored moves are built up in a dictionary after every epoch with an associated Q-value and updated using the earlier discussed formula by the 'reinforce' method. Hence this process is known as Q-Learning or Reinforcement Learning. We then tested the parameters coded into the Learner class governing the exploration rate, learning rate, and discount factor to find an optimal set of parameters for fast, but accurate, learning.
# 
# The results showed that learning can happen over less epochs, reducing runtime, whilst relative accuracy as measured by the games won. However, these tests were basic and did not account for how each parameter affects one another. Hence further tests can be done which increment the dependent variables individually over the tests i.e., test exploration rate incrementally from 0-1 for every incremental value of learning rate/discount factor from 0-1. This would give a better understanding of how each parameter affects the other and help better fine-tune these parameters.
# 
# Furthermore, upon further examination of the game through many human P1 vs. bot P2 games, the bot demonstrates an inability to notice when to make a move that blocks P1 from wining which potentially leads to the roughly 10% loss ratio result from the last model test. This exhibits itself as such: P1 lines up 2Xs in a row with a clear 3rd move to win, however the P2 bot does not place an O in the 3rd wining space. This could be due to not having had enough epochs to have learned the optimal move in this case, however experiments with higher epoch counts were conducted with the bot still demonstrating this issue. As such, improvements to the 'best_move' move method in the Learner object could be made such as adding to the read Q-value from the stored states for the position which blocks P1 wins. Such a move would add an element of supervision to the unsupervised algorithm.
