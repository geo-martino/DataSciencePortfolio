import itertools
import sys
from statistics import mean
from time import sleep

import numpy as np
from tqdm.auto import tqdm
from tqdm.auto import trange


class Base:
    """
    Builds a base game of TicTacToe.
    ...
    :attribute size: int. Size of board being played on. Default: 3
    :attribute players: dict. A dict of the two player present in the form {1: 'x', -1: 'x'} where x can be
        h, r or b for human, random or trained bot. Default: None
    :attribute winner: int. Records the current game's winner as an integer.
    :attribute draw: bool. Records whether the game board is full with no winner.
    :attribute turn: int. Records the turns of the current game.
    :attribute board: array. An array of the current board state.
    ...
    :methods
    reset_state(): Resets the parameters for winner, draw, turn and board state.

    print_grid(c=""): Prints a visual representation of the current board state including Xs and Os.

    print_win(winner, name="?"): Prints the winner and a visual representation of the current board state
        and colours it depending on the win or draw state.

    check_win(self): Checks if there is a winner in the current board state.

    move(player, x, y): Updates the current board state with a move based on x and y.

    human(player): Gets a move from the user.

    basic_play(): Plays 1 game based on the input players for testing.

    get_values(): Gets input from the user on grid size and players
        if grid size is below 3 or the players dict is not present.
    """

    def __init__(self, N=3, players=None):
        """
        Builds a base game of TicTacToe.
        ...
        :param N: int. Size of board being played on. Default: 3
        :param players: dict. A dict of the two player present in the form {1: 'x', -1: 'x'} where x can be
            h, r or b for human, random or trained bot.
        """

        self.size = N  # board dimensions (n x n)
        self.players = players  # dictionary of players for this game

        if players is None:  # if no players, get input from user
            self.get_values()
        if self.size < 3:  # check grid size is 3 or more
            sys.exit("Error: Grid size too small. Enter a number that is 3 or higher.")

        self.winner = 0  # winner state
        self.draw = False  # draw state
        self.turn = 0  # current turn
        self.board = np.zeros(shape=(self.size, self.size)).astype('int')  # array for current board state

    def reset_state(self):
        """
        Resets the parameters for winner, draw, turn and board state.
        """
        self.winner = 0
        self.draw = False
        self.turn = 0
        self.board = np.zeros(shape=(self.size, self.size)).astype('int')

    def print_grid(self, winner_colour=""):
        """
        Prints a visual representation of the current board state including Xs and Os.
        ...
        :param winner_colour: str. A string to add to the print function to colour the board. e.g. '\33[3m' Default: ""
        """
        if winner_colour == "1":  # colour for player 1
            winner_colour = '\33[92;1m'
        elif winner_colour == "-1":  # colour for player 2
            winner_colour = '\33[93;1m'
        elif winner_colour == "0":  # colour for draw state
            winner_colour = '\33[94;1m'

        print(winner_colour, "  ", "+---" * self.size, "+", '\33[0m', sep="")  # first line
        y = self.size + 1  # y-coordinates to print on left of board
        for row in self.board:  # iterate through each row of board
            y -= 1
            print(y, winner_colour, " ", '\33[0m', sep="", end="")  # y-coordinates on left side

            for space in row:  # for each number in this row
                print(winner_colour, "| ", '\33[0m', sep="", end='')  # left wall of each space
                if space == 0:  # empty space if no player occupies space
                    print("  ", sep="", end='')
                elif space == 1:  # X if player 1 in space
                    print('\33[92;1m', "X ", '\33[0m', sep="", end="")
                elif space == -1:  # O if player 2 in space
                    print('\33[93;1m', "O ", '\33[0m', sep="", end="")

            print(winner_colour, "|", '\33[0m', sep="")  # right wall of board
            print(winner_colour, "  ", "+---" * self.size, "+", '\33[0m', sep="")  # next line
        print(" ", end="", sep="")

        for x in range(self.size):  # x-coordinates on bottom
            print("   ", x + 1, sep="", end="")
        print()

    def print_win(self, winner, name="?"):
        """
        Prints the winner and a visual representation of the current board state
        and colours it depending on the win or draw state.
        ...
        :param winner: int. An integer representing the game winner.
        :param name: str. The name of the game winner Default: "?"
        """
        self.print_grid(str(winner))  # print coloured version of grid
        if self.winner == 0:  # draw state
            print('\33[94m', "Draw!", '\33[0m')
        else:  # print winner text
            print('\33[92;1m', name, "wins!", '\33[0m')
        print('\33[91;3m', "Turns taken:", self.turn, '\33[0m', '\n')

    def check_win(self):
        """
        Checks if there is a winner in the current board state.
        """
        diagonal_LR = sum(self.board.diagonal())  # sum of diagonal from upper left to lower right
        diagonal_RL = sum(np.fliplr(self.board).diagonal())  # sum of diagonal from upper right to lower left

        if self.size in (diagonal_LR, diagonal_RL):  # check P1 win in diagonal
            self.winner = 1
        elif -self.size in (diagonal_LR, diagonal_RL):  # check P2 win in diagonal
            self.winner = -1
        else:  # check win in rows
            for (i, j) in zip(self.board, np.transpose(self.board)):  # iterate through rows and columns simultaneously
                row = sum(i)
                col = sum(j)
                if self.size in (row, col):  # check P1 win row or column
                    self.winner = 1
                elif -self.size in (row, col):  # check P2 win row or column
                    self.winner = -1

    def move(self, player, x, y):
        """
        Updates the current board state with a move based on x and y.
        ...
        :param player: int. The player to make a move for.
        :param x: int. Column index of the move to be taken.
        :param y: int. Row index of the move to be taken.
        """
        self.turn += 1  # increase turn
        self.board[y][x] = player  # change board state at position to current player number identifier

        if self.turn >= self.size * 2 - 1:  # if required amount of turns for a possible to happen have occurred
            self.check_win()  # check for winner
            self.draw = False if np.count_nonzero(self.board) != self.size ** 2 else True  # draw if board is full

        return

    def human(self, player):
        """
        Gets a move from the user.
        ...
        :param player: int. The player to make a move for.
        :return: Row and column indices of the move to be taken.
        """
        self.print_grid()  # print current board state as grid
        x = -1  # initialise x-coordinate
        y = -1  # initialise y-coordinate

        # print instructions
        if player == 1:
            print('\33[92;1m', "Player 1's move - X", '\33[0m', sep="", end="")
        else:
            print('\33[93;1m', "Player 2's move - O", '\33[0m', sep="", end="")
        print('\33[94m', "   Enter X and Y coordinates of move from 1 to ", self.size, " in the form:", '\33[0m',
              '\33[91;1m', " x,y", '\33[0m', sep="")

        while x < 0 or y < 0:
            try:
                user_input = input("Your move: ")  # get input from user
                if user_input == 'q':  # quit game
                    sys.exit("User quit the program")
                else:  # split coordinates into x and y
                    x, y = user_input.split(',')
            except ValueError:  # input not in correct form
                print('\33[91;3;1m', "Error: Enter two numbers in the form: x,y", '\33[0m')
                print('\33[91;3m', "e.g: 2,1", '\33[0m')
                print('\33[91m', "(Enter 'q' to end program)", '\33[0m')
                continue

            try:  # convert coordinates into array indices
                x = int(x) - 1
                y = self.size - int(y)
            except ValueError:  # user has not input integers
                print('\33[91;3;1m', "Error: Enter integers between 1 and", self.size, "only", '\33[0m')
                x = -1
                y = -1
                continue

            if x >= self.size or y >= self.size or x < 0 or y < 0:  # integers out of range of board size
                print('\33[91;3;1m', "Error: Only enter numbers between 1 and", self.size, '\33[0m')
                x = -1
                y = -1
            elif self.board[y][x] != 0:  # input space is taken
                print('\33[91;3;1m', "Space taken. Try another space!", '\33[0m')
                x = -1
                y = -1

        return y, x

    def basic_play(self):
        """
        Plays 1 game based on the input players for testing.
        """
        self.reset_state()  # reset board
        p = 1  # current player

        while self.winner == 0 and not self.draw:  # play game until winner or draw
            if self.players[p] == "h":  # human player move
                name = 'Human Player ' + str(p) if p == 1 else 'Human Player 2'  # set player name
                y, x = self.human(p)  # get move indices from user
            else:  # random move
                name = 'Random Player ' + str(p) if p == 1 else 'Random Player 2'  # set player name
                positions = np.argwhere(self.board == 0)  # get available spaces to make a move
                y, x = positions[np.random.randint(0, len(positions), size=None)]  # randomly select space
                print('\33[34;1m', "Random Player ", p if p == 1 else 2, "'s move: (",
                      x + 1, ",", self.size - y, ")", '\33[0m', sep="")

            self.move(p, x, y)  # make the move

            if 'h' not in self.players.values() and self.winner == 0 and not self.draw:  # print board if no winner
                self.print_grid()
                sleep(1)
            elif self.winner != 0 or self.draw:  # print coloured winner board if winner or draw
                self.print_win(self.winner, name)

            p = -p  # switch players

    def get_values(self):
        """
        Gets input from the user on grid size and players if grid size is below 3 or the players dict is not present.
        """
        # get the player types from user
        self.players = {1: input("Is player 1 human (h), bot (b) or random (r)?: "),
                        -1: input("Is player 2 human (h), bot (b) or random (r)?: ")}

        # check types were defined correctly and re-prompt if necessary
        while self.players[1] not in ('h', 'b', 'r') and self.players[-1] not in ('h', 'b', 'r'):
            print('\33[91;3;1m', "Error: Enter only 'h' or 'b' or 'r'", '\33[0m')
            self.players = {1: input("Is player 1 human (h), bot (b) or random (r)?: "),
                            -1: input("Is player 2 human (h), bot (b) or random (r)?: ")}

        while self.size < 3:  # loop until size conditions satisfied
            try:
                self.size = int(input("Grid size (enter a number that is 3 or higher): "))  # get grid size from user
            except ValueError:  # input is not an integer
                print('\33[91;3;1m', "Error: Enter only integers", '\33[0m')
                continue

            if self.size < 3:  # input integer is below 3
                print('\33[91;3;1m', "Error: Grid size too small. Enter a number that is 3 or higher.", '\33[0m')


class Learner:
    """
    Trains a model for a game of TicTacToe of size N.
    ...
    :attribute size: int. Size of board being played on. Default: 3
    :attribute exp_rate: float, optional. Exploration rate of the model builder between 0.0 (no exploration)
                    and 1.0 (only random moves). Default: 0.6
    :attribute learn_rate: float, optional. Rate at which the model builder learns. Default: 0.1
    :attribute discount: float, optional. Discount factor multiplied with the new Q-value obtained from the
                    current epoch. Default: 0.9
    :attribute r_win: int, optional. Reward factor for winning a game. Default: 2
    :attribute r_draw: int, optional. Reward factor for winning a game. Default: -1
    :attribute r_lose: int, optional. Reward factor for winning a game. Default: -3

    :attribute current_states_1 : dict. Stores the hash states of the board for the current epoch for player 1 as keys,
        and their associated Q-values obtained from states_q as values.
    :attribute current_states_2 : dict. Stores the hash states of the board for the current epoch for player 2 as keys,
        and their associated Q-values obtained from states_q as values.
    :attribute states_q : dict. Store the hash states of the board for all epochs so far for both players and their
        associated Q-values.
    :attribute epoch_count_1 : int. Current epoch count for player 1
    :attribute epoch_count_2 : int. Current epoch count for player 2

    :attribute totals : dict. Total wins, draws and losses for the testing phase. Stored as a dict of dicts per move.
        i.e. {'wins': {5:23, 7:11}, 'draws': {...
    ...
    :methods
    counters(winner, player, turns): Updates the counters in the self.total attribute.

    reset_counters(): Resets the counters in the self.total attribute.

    random_move(board, player): Finds a random move on the input board array for the input player.

    best_move(board, player): Finds the best move on the input board array for the input player based on the
        states in self.states_q.

    reward(winner, player): Returns a number for the reward to be given to the input player based on the winner.
        Reward numbers defined by r_win, r_draw and r_lose.

    reinforce(winner, player): Updates the Q-values in self.states_q for the hash values of the current game.

    build_model(game): Builds the self.states_q dictionary by simulating a game between two bot players both
        learning from each other.

    play(game, test=0, learn=False, print_moves=False, print_grid=False, pause=0.5): Tests the built model.
    """

    def __init__(self, N=3, exp=0.9, lrn=0.1, dsc=1.0, r_win=2, r_draw=-1, r_lose=-3):
        """
        Constructs all the necessary attributes for the person object.
        ...
        :param N: int. Size of board being played on. Default: 3
        :param exp: float, optional. Exploration rate of the model builder between 0.0 (no exploration)
            and 1.0 (only random moves). Default: 0.9
        :param lrn: float, optional. Rate at which the model builder learns. Default: 0.1
        :param dsc: float, optional. Discount factor multiplied with the new Q-value obtained from the
            current epoch. Default: 1.0
        :param r_win: int, optional. Reward factor for winning a game. Default: 2
        :param r_draw: int, optional. Reward factor for winning a game. Default: -1
        :param r_lose: int, optional. Reward factor for winning a game. Default: -3
        """

        self.size = N  # board dimensions (n x n)
        self.exp_rate = exp  # exploration rate
        self.learn_rate = lrn  # learning rate of q-learning formula
        self.discount_factor = dsc  # discount factor of q-learning formula

        self.r_win = r_win  # reward value for winning
        self.r_draw = r_draw  # reward value for drawing
        self.r_lose = r_lose  # reward value for losing

        self.current_states_1 = {}  # all moves taken in the current game for player 1 (str): q-values
        self.current_states_2 = {}  # all moves taken in the current game for player 2 (str): q-values
        self.states_q = {}  # all possible states for both players (str): q-values
        self.epoch_count_1 = 0  # total epoch count for player 1 bot
        self.epoch_count_2 = 0  # total epoch count for player 2 bot

        self.totals = {'wins': {}, 'draws': {}, 'losses': {}}  # wdl counts

    def counters(self, winner, player, turns):
        """
        Updates the counters in the self.total attribute.
        ...
        :param winner: int. An integer representing the game winner.
        :param player: int. An integer representing the player to count for.
        :param turns: int. The total number of turns taken this epoch.
        """
        if winner == player:  # get total wins number if won
            update = self.totals.get('wins')
        elif winner == 0:  # get total draws number if draw
            update = self.totals.get('draws')
        else:  # get total losses number if lost
            update = self.totals.get('losses')
        update[turns] = update[turns] + 1 if update.get(turns) is not None else 1  # update wdl for this game's turns

    def reset_counters(self):
        """
        Resets the counters in the self.total attribute.
        """
        self.totals = {'wins': {}, 'draws': {}, 'losses': {}}

    def random_move(self, board, player):
        """
        Finds a random move on the input board array for the input player.
        ...
        :param board: arr. An array of the current board.
        :param player: int. The player to make a move for.
        :return action: tuple. Index of the move to be taken.
        :return boardHash: tuple. A tuple hash of the board after the move has been taken.
        :return value: float. The Q-value obtained from self.states_q for the move to be taken

        """
        positions = np.argwhere(board == 0)  # get available spaces to make a move
        action = positions[np.random.randint(0, len(positions), size=None)]  # randomly select space

        board[action[0]][action[1]] = player  # change board state at position to current player number identifier
        nextBoard = board.flatten()  # cast board state as string
        boardTuple = tuple(nextBoard)  # convert board state to immutable tuple for dict key storage

        # get stored q-value if available, 0 if not
        q = 0 if self.states_q.get(boardTuple) is None else self.states_q.get(boardTuple)

        return action, boardTuple, q

    def best_move(self, board, player):
        """
        Finds the best move on the input board array for the input player based on the states in self.states_q.
        ...
        :param board: arr. An array of the current board.
        :param player: int. The player to find a move for.
        :return action: tuple. Index of the move to be taken.
        :return boardHash: tuple. A tuple hash of the board after the move has been taken.
        :return value: float. The Q-value obtained from self.states_q for the move to be taken
        """
        action = None
        board_tuple = None
        q = None
        q_max = -100  # set low max q-value to improve upon

        positions = np.argwhere(board == 0)  # get available spaces to make a move

        for p in positions:  # iterate through all available positions
            next_board = board.copy()  # copy current board state
            next_board[p[0]][p[1]] = player  # add player move to board copy
            next_board = next_board.flatten()  # cast board state as string
            board_tuple = tuple(next_board)  # convert board state to immutable tuple for dict key storage

            # get stored q-value if available, 0 if not
            q = 0 if self.states_q.get(board_tuple) is None else self.states_q.get(board_tuple)
            if q >= q_max:  # check if stored q-value is better than the current best move
                q_max = q  # update max q-value
                action = p  # store action of highest possible q-value

        return action, board_tuple, q

    def reward(self, winner, player):
        """
        Returns a number for the reward to be given to the input player based on the winner.
        Reward numbers defined by r_win, r_draw and r_lose.
        ...
        :param winner: int. An integer representing the game winner.
        :param player: int. An integer representing the player to be rewarded.
        :return: int. Reward numbers defined by r_win, r_draw and r_lose.
        """
        if winner == player:  # return win reward value
            return self.r_win
        elif winner == 0:  # return draw reward value
            return self.r_draw
        else:  # return lose reward value
            return self.r_lose

    def reinforce(self, winner, player):
        """
        Updates the Q-values in self.states_q for the hash values of the current game.
        ...
        :param winner: int. An integer representing the game winner.
        :param player: int. An integer representing the player to be reinforced.
        """
        if player == 1:  # reinforce bot player 1
            # create two iterators for current and next board state
            iter_a, iter_b = itertools.tee(self.current_states_1.items())
            self.epoch_count_1 += 1
        else:  # reinforce bot player 2
            # create two iterators for current and next board state
            iter_a, iter_b = itertools.tee(self.current_states_2.items())
            self.epoch_count_2 += 1

        reward = self.reward(winner, player)  # get reward value for this epoch
        next(iter_b, (0, 0))  # advance second iterator to next board state

        for board_hash, q in iter_a:
            next_board_hsh, next_q = next(iter_b, (0, 0))  # advance second iterator to next board state

            if self.states_q.get(board_hash) is None:  # add board state to q-states with 0 q-value if not stored
                self.states_q[board_hash] = 0

            if next_board_hsh != 0:
                next_board = np.reshape(next_board_hsh, (-1, self.size))  # reshape into (n x n) numpy array
                action, bestHash, best_q = self.best_move(next_board, player)  # get next best q-value from stored state
                if best_q > next_q:  # use stored states q-value if better move taken in this game
                    next_max_q = best_q
                else:  # otherwise, use this game's q-value
                    next_max_q = next_q
            else:  # use 0 value if not stored
                next_max_q = 0

            # generate updated q-value using reinforcement learning formula
            new_q = (1 - self.learn_rate) * q + self.learn_rate * (reward + self.discount_factor * next_max_q)
            self.states_q[board_hash] = new_q  # update stored q-value for current board state iteration

    def build_model(self, game):
        """
        Builds the self.states_q dictionary by simulating a game between two bot players both learning from each other.
        ...
        :param game: obj. The game as prepared by the TicTacToe() object.
        """
        self.current_states_1 = {}  # reset current states
        self.current_states_2 = {}
        p = 1  # current player

        while game.winner == 0 and not game.draw:  # play game until winner or draw
            # take random move if random number generated is below exploration rate or at a low epoch count
            if np.random.uniform(0, 1) <= self.exp_rate or self.epoch_count_2 < 100:
                action, next_hash, next_q = self.random_move(game.board, p)
            else:  # find best possible move from stored q-values
                action, next_hash, next_q = self.best_move(game.board, p)

            if p == 1:  # add board state: q-value pair to player 1 states
                self.current_states_1[next_hash] = next_q
            else:  # add board state: q-value pair to player 2 states
                self.current_states_2[next_hash] = next_q

            x, y = action[::-1]  # switch x-y coordinate order
            game.move(p, x, y)  # make the move

            p = -p  # switch players

        # reinforce bots for both players
        self.reinforce(game.winner, 1)
        self.reinforce(game.winner, -1)

    def play(self, game, test=0, learn=False, print_moves=False, print_grid=False, pause=0.5):
        """
        Tests the built model.
        ...
        :param game: obj. The game as prepared by the TicTacToe() object.
        :param test: int. The player being tested (either 1 or -1 for player 2)
        :param learn: bool. Define whether the model should learn from the game being played. Default: False
        :param print_moves: bool. Define whether the moves list should be printed. Default: False
        :param print_grid: bool. Define whether a visual representation of the grid should be printed. Default: False
        :param pause: float. The factor by which the printed grid pauses in-between moves if print_grid is True.
            Default: 0.5
        """
        self.current_states_1 = {}  # reset current states
        self.current_states_2 = {}
        p = 1  # current player

        # dictionary of names and colours according to player type dictionary
        names = {'r': ["Random ", '\33[34;1m'], 'b': ["Bot ", '\33[35;1m'], 'h': ["Human ", '\33[36;1m']}

        print_win = True if 'h' in game.players.values() else print_grid  # print winner if human player present
        print_grid = False if 'h' in game.players.values() else print_grid  # don't print grid if human player present

        while game.winner == 0 and not game.draw:  # play game until winner or draw
            if game.players[p] == "h":  # human player move
                action = game.human(p)
            elif game.players[p] == "b":  # bot move
                action, nextHash, next_q = self.best_move(game.board, p)
            else:  # random move
                action, nextHash, next_q = self.random_move(game.board, p)

            x, y = action[::-1]  # switch x-y coordinate order
            game.move(p, x, y)  # make the move

            # string of current player's name for printing
            name = names[game.players[p]][0] + '1' if p == 1 else names[game.players[p]][0] + '2'

            if print_moves and name != 'Human':  # print moves for non-human players or if print_moves is True
                colour = names[game.players[p]][1]  # get text colour
                print(colour, "Player ", 1 if p == 1 else 2, "'s (", name, ") move: (",
                      x + 1, ",", self.size - y, ")", '\33[0m', sep="")

            if print_grid and game.winner == 0 and not game.draw:  # print board if no winner
                game.print_grid()
                sleep(1 * pause)
            elif print_win and (game.winner != 0 or game.draw):  # print coloured winner board if winner or draw
                game.print_win(game.winner, name)
                sleep(1.5 * pause)

            p = -p  # switch players

        if learn:  # update q-values of bot player
            self.reinforce(game.winner, test)

        self.counters(game.winner, test, game.turn)  # update wdl counters for current epoch


class Tester:
    """
    Calls functions in the Learner class to test the capability of the model builder.
    ...
    :attribute size: int. Size of board being played on. Default: 3
    :attribute trainGame obj. The base game built with two bot players.
    :attribute model obj. The object that builds and store the model.
    ...
    :methods
    print_results(title=None, colour=None, amount=None, totals=None, turns=None, average=False):
        Prints the results from each test. Prints wins, draws, losses, and losses per turn if specified.

    build_model(self, epochs=10000, print_bar=True, print_results=True):
        Builds the model in the self.model object.

    test_model(self, player_list, moves_list=None, trials=10, trial_games=1000, print_bar=True,
               print_results=True, print_moves=False, print_grid=False, pause=0.5, learn=False):
        Tests the model over many games and trials and returns the results.

    test_control(self, player_list, moves_list=None, trials=10, trial_games=1000, print_bar=True,
                 print_results=True, print_moves=False, print_grid=False, pause=0.5):
        Tests a randomly player control game over many games and trials and returns the results.
    """

    def __init__(self, N=3, exp=0.9, lrn=0.1, dsc=1.0, r_win=2, r_draw=-1, r_lose=-3):
        """
        Calls functions in the Learner class to test the capability of the model builder.
        ...
        :param N: int. Size of board being played on. Default: 3
        :param exp: float, optional. Exploration rate of the model builder between 0.0 (no exploration)
            and 1.0 (only random moves). Default: 0.9
        :param lrn: float, optional. Rate at which the model builder learns. Default: 0.1
        :param dsc: float, optional. Discount factor multiplied with the new Q-value obtained from the
            current epoch. Default: 1.0
        :param r_win: int, optional. Reward factor for winning a game. Default: 2
        :param r_draw: int, optional. Reward factor for winning a game. Default: -1
        :param r_lose: int, optional. Reward factor for winning a game. Default: -3
        """

        self.size = N  # board dimensions (n x n)
        self.trainGame = Base(self.size, {1: 'b', -1: 'b'})  # instantiate base game object
        self.model = Learner(self.size, exp, lrn, dsc, r_win, r_draw, r_lose)  # instantiate learner object

    def print_results(self, title=None, colour=None, amount=None, totals=None, turns=None, average=False):
        """
        Prints the results from each test. Prints wins, draws, losses, and losses per turn if specified.
        ...
        :param title: str. The heading of the test being conducted. Default: None
        :param colour: str. The colour code to print the string as. e.g. '\33[3m'. Default: None
        :param amount: int. The total number of trials conducted. Default: None
        :param totals: dict. A dict of lists for all wins, draws and losses per trial in the form:
            {'wins': [], 'draws': [], 'losses': []}. Default: None
        :param turns: dict. A dict of dicts for all wins, draws and losses per move in the form:
            {'wins': {}, 'draws': {}, 'losses': {}}. Default: None
        :param average: bool. Determine whether the results printed are an average value or not. Default: False
        """
        print(colour, title, '\33[0m')

        if totals is not None:  # print total wins, draws and losses
            for type, values in totals.items():
                print(round(mean(values)), " average " if average else " ", type, " out of ", amount, sep='', end='')
                print(" -", values) if len(values) > 1 else print()
        if turns is not None:  # print losses on turns specified
            for turn, values in turns.items():
                if turn < self.size ** 2:
                    print('\33[3m', values, " average" if average else " ", " losses on ", turn, " turns ", amount,
                          '\33[0m', sep='')

    def build_model(self, epochs=10000, print_bar=True, print_results=True):
        """
        Builds the model in the self.model object.
        ...
        :param epochs: int. The total number of epochs to conduct tests for. Default: 10000
        :param print_bar: bool. Define whether a progress bar should be printed. Default: False
        :param print_results: bool. Define whether results should be printed. Default: False
        :return: dict. Returns a dict of all the states learned and their associated Q-values.
        """
        bar = trange(epochs, desc='Learning moves', unit='epochs', leave=False, file=sys.stdout) \
            if print_bar else range(epochs)  # progress bar

        for _ in bar:  # train bot over number of epochs
            self.trainGame.reset_state()  # reset game state
            self.model.build_model(self.trainGame)  # run epoch

        if print_results:  # print moves learned
            print('\33[3m', "Learned", len(self.model.states_q), "possible moves", '\33[0m', end='')
            print('\33[3m', "out of 5477", '\33[0m', '\n', sep='') if self.size == 3 else print('\n')

        return self.model.states_q  # return q-state: q-value pairs

    def test_model(self, player_list, moves_list=None, trials=10, trial_games=1000, print_bar=True,
                   print_results=True, print_moves=False, print_grid=False, pause=0.5, learn=False):
        """
        Tests the model over many games and trials and returns the results.
        ...
        :param player_list: list. A list of two strings containing the players to test.
            i.e. ['h','b'] where h, r or b is human, random or trained bot.
        :param moves_list: list. A list of integers of total turns to print losses for. Default: None
        :param trials: int. Total trials to conduct. Default: 10
        :param trial_games: int. Total games per trial. Default: 1000
        :param print_bar: bool. Define whether the progress bar should be shown. Default: True
        :param print_results: bool. Define whether results should be printed. Default: True
        :param print_moves: bool. Define whether the moves list should be printed. Default: False
        :param print_grid: bool. Define whether a visual representation of the grid should be printed. Default: False
        :param pause: float. The factor by which the printed grid pauses in-between moves if print_grid is True.
            Default: 0.5
        :param learn: bool. Define whether the model should learn from the game being played. Default: False
        :return wdl: dict. A dict of lists for all wins, draws and losses per trial in the form:
            {'wins': [], 'draws': [], 'losses': []}
        :return wdl_moves: dict. A dict of dicts for all wins, draws and losses per move in the form:
            {'wins': {}, 'draws': {}, 'losses': {}}
        """
        trial_players = {1: player_list[0], -1: player_list[1]}  # player type dictionary
        try:  # check if bot player present in given player list
            test_player = list(trial_players.keys())[list(trial_players.values()).index('b')]
        except ValueError:
            sys.exit("Bot player not defined in player list")

        moves = {i: 0 for i in moves_list} if moves_list is not None else None  # dictionary to store turn counts
        wdl = {'wins': [], 'draws': [], 'losses': []}  # stores total wins, draws and losses per epoch
        wdl_moves = {'wins': {}, 'draws': {}, 'losses': {}}  # stores total wins, draws and losses per turn count
        trialGame = Base(self.size, trial_players)  # instantiate base game for player dictionary

        number = '1' if test_player == 1 else '2'  # test player number to print progress bar for
        bar = trange(trials, desc='Testing Player ' + number, unit='trials', leave=False, file=sys.stdout) \
            if print_bar else range(trials)  # progress bar

        for _ in bar:  # run tests for given trial number
            self.model.reset_counters()  # reset wdl counters
            for j in range(trial_games):  # run games per trial
                trialGame.reset_state()  # reset game state
                self.model.play(trialGame, test=test_player, learn=learn, print_moves=print_moves,
                                print_grid=print_grid, pause=pause)  # play game

            for result, value in self.model.totals.items():  # update wdl dictionaries for current trial
                wdl_moves[result] = {k: wdl_moves[result].get(k, 0) + value.get(k, 0) for k in set(value)}
                wdl[result].append(sum(value.values()))
                if result == 'losses' and moves is not None:  # update loss values for printing if required
                    for move in moves:
                        if move in value:
                            moves[move] = value[move]

        if print_results:  # print results
            self.print_results(title='Results for Player ' + number, colour='\33[92;1m', amount=trial_games,
                               totals=wdl, turns=moves, average=True)
            print('\33[92;3m', "Total learned Epochs for Player 1", self.model.epoch_count_1, '\33[0m')
            print('\33[92;3m', "Total learned Epochs for Player 2", self.model.epoch_count_2, '\n', '\33[0m')

        return wdl, wdl_moves

    def test_control(self, player_list, moves_list=None, trials=10, trial_games=1000, print_bar=True,
                     print_results=True, print_moves=False, print_grid=False, pause=0.5):
        """
        Tests a randomly player control game over many games and trials and returns the results.
        ...
        :param player_list: list. A list of two strings containing the players to control test.
            i.e. ['h','b'] where h, r or b is human, random or trained bot.
        :param moves_list: list. A list of integers of total turns to print losses for. Default: None
        :param trials: int. Total trials to conduct. Default: 10
        :param trial_games: int. Total games per trial. Default: 1000
        :param print_bar: bool. Define whether the progress bar should be shown. Default: True
        :param print_results: bool. Define whether results should be printed. Default: True
        :param print_moves: bool. Define whether the moves list should be printed. Default: False
        :param print_grid: bool. Define whether a visual representation of the grid should be printed. Default: False
        :param pause: float. The factor by which the printed grid pauses in-between moves if print_grid is True.
            Default: 0.5
        :return wdl: dict. A dict of lists for all wins, draws and losses per trial in the form:
            {'wins': [], 'draws': [], 'losses': []}
        :return wdl_moves: dict. A dict of dicts for all wins, draws and losses per move in the form:
            {'wins': {}, 'draws': {}, 'losses': {}}
        """
        random_players = {1: 'r', -1: 'r'}  # player type dictionary for two random players
        trial_players = {1: player_list[0], -1: player_list[1]}
        try:  # check if bot player present in given player list
            test_player = list(trial_players.keys())[list(trial_players.values()).index('b')]
        except ValueError:
            sys.exit("Bot player not defined in player list")

        moves = {i: 0 for i in moves_list} if moves_list is not None else None  # dictionary to store turn counts
        wdl = {'wins': [], 'draws': [], 'losses': []}  # stores total wins, draws and losses per epoch
        wdl_moves = {'wins': {}, 'draws': {}, 'losses': {}}  # stores total wins, draws and losses per turn count
        randomGame = Base(self.size, random_players)  # instantiate base game for random control

        number = '1' if test_player == 1 else '2'  # test player number to print progress bar for
        bar = trange(trials, desc='Control Testing Player ' + number, unit='trials', file=sys.stdout) \
            if print_bar else range(trials)  # progress bar

        for _ in bar:  # run tests for given trial number
            self.model.reset_counters()  # reset wdl counters
            for j in range(trial_games):  # run games per trial
                randomGame.reset_state()  # reset game state
                self.model.play(randomGame, test=test_player, learn=False, print_moves=print_moves,
                                print_grid=print_grid, pause=pause)  # play game

            for result, value in self.model.totals.items():  # update wdl dictionaries for current trial
                wdl_moves[result] = {k: wdl_moves[result].get(k, 0) + value.get(k, 0) for k in set(value)}
                wdl[result].append(sum(value.values()))
                if result == 'losses' and moves is not None:  # update loss values for printing if required
                    for move in moves:
                        if move in value:
                            moves[move] = value[move]

        if print_results:  # print results
            self.print_results(title='Random control for Player ' + number, colour='\33[93;1m',
                               amount=trial_games, totals=wdl, turns=moves, average=True)

        return wdl, wdl_moves


def parameter_test(test, exp=0.9, lrn=0.1, dsc=1.0, start_epc=500, max_epc=20000, inter_epc=500):
    """
    Tests the exploration rate, learn rate and discount factor parameters within
    ...
    :param test: str. The parameter being tested. 'e' = exploration, 'l' = learn rate, 'd' = discount factor.
    :param exp: float. The exploration rate to be used if not being tested. Default: 0.9
    :param lrn: float. The learn rate rate to be used if not being tested. Default: 0.1
    :param dsc: float. The discount factor to be used if not being tested. Default: 1.0
    :param start_epc: int. The epoch number to start tests on. Default: 500
    :param max_epc: int. The epoch number to end tests on. Default: 20000
    :param inter_epc: int. The epochs to be added per test. Default: 500
    :return moves_learned: list. The moves learned by each interval.
    :return P1_wins: list. The wins player 1 made at each interval.
    :return P2_wins: list. The wins player 2 made at each interval.
    """
    max_epc += 1  # include epoch number in epoch range

    def epoch_loop(rate):
        moves = []  # moves learned per epoch
        P1 = []  # wdl stats per epoch
        P2 = []

        epoch_bar = tqdm(range(start_epc, max_epc, inter_epc), desc='Rate: ' + str(rate), unit='test',
                         leave=False, file=sys.stdout)  # progress bar

        for _ in epoch_bar:  # run parameter test
            # build model and store learned states
            learned_states = parameterTester.build_model(inter_epc, print_bar=False, print_results=False)

            # run parameter test and store wdl and turn counts
            P1_epochs_wdl, P1_epochs_turns = parameterTester.test_model(player_list=['b', 'r'],
                                                                        print_bar=False, print_results=False)
            P2_epochs_wdl, P2_epochs_turns = parameterTester.test_model(player_list=['r', 'b'],
                                                                        print_bar=False, print_results=False)

            moves.append(len(learned_states))  # append number of moves learned for this epoch count
            P1.append(mean(P1_epochs_wdl['wins']))  # append wdl states to dictionary
            P2.append(mean(P2_epochs_wdl['wins']))

        return moves, P1, P2

    moves_learned = {}  # all moves learned per epoch
    P1_wins = {}  # all wdl stats per epoch
    P2_wins = {}

    bar = tqdm(range(0, 11), desc='Testing', unit='test', leave=True, file=sys.stdout)  # progress bar

    for i in bar:
        i = i / 10  # reduce parameter range to increments of 0.1

        # instantiate learner objects for testing
        if test == 'e':  # test exploration rate
            parameterTester = Tester(N=3, exp=i, lrn=lrn, dsc=dsc)
        elif test == 'l':  # test learning rate
            parameterTester = Tester(N=3, exp=exp, lrn=i, dsc=dsc)
        elif test == 'd':  # test discount rate
            parameterTester = Tester(N=3, exp=exp, lrn=lrn, dsc=i)
        else:  # return error
            print('Error: parameter to test required (e, l, or d).')
            return

        moves_list, P1_wins_list, P2_wins_list = epoch_loop(i)  # train and test model for current parameter value
        moves_learned[i] = moves_list  # add moves learned counts
        P1_wins[i] = P1_wins_list  # add wdl stats
        P2_wins[i] = P2_wins_list

    return moves_learned, P1_wins, P2_wins
