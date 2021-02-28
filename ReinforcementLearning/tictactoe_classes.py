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

        self.size = N
        self.players = players

        if players is None:
            self.get_values()

        self.winner = 0
        self.draw = False
        self.turn = 0
        self.board = np.zeros(shape=(self.size, self.size)).astype('int')
        if self.size < 3:
            sys.exit("Error: Grid size too small. Enter a number that is 3 or higher.")

    def reset_state(self):
        """
        Resets the parameters for winner, draw, turn and board state.
        """
        self.winner = 0
        self.draw = False
        self.turn = 0
        self.board = np.zeros(shape=(self.size, self.size)).astype('int')

    def print_grid(self, c=""):
        """
        Prints a visual representation of the current board state including Xs and Os.
        ...
        :param c: str. A string to add to the print function to colour the board. e.g. '\33[3m' Default: ""
        """
        if c == "1":
            c = '\33[92;1m'
        elif c == "-1":
            c = '\33[93;1m'
        elif c == "0":
            c = '\33[94;1m'

        print(c, "  ", "+---" * self.size, "+", '\33[0m', sep="")
        a = self.size + 1
        for i in self.board:
            a -= 1
            print(a, c, " ", '\33[0m', sep="", end="")

            for j in i:
                print(c, "| ", '\33[0m', sep="", end='')
                if j == 0:
                    print("  ", sep="", end='')
                elif j == 1:
                    print('\33[92;1m', "X ", '\33[0m', sep="", end="")
                elif j == -1:
                    print('\33[93;1m', "O ", '\33[0m', sep="", end="")

            print(c, "|", '\33[0m', sep="")
            print(c, "  ", "+---" * self.size, "+", '\33[0m', sep="")
        print(" ", end="", sep="")

        for b in range(self.size):
            print("   ", b + 1, sep="", end="")
        print()

    def print_win(self, winner, name="?"):
        """
        Prints the winner and a visual representation of the current board state
        and colours it depending on the win or draw state.
        ...
        :param winner: int. An integer representing the game winner.
        :param name: str. The name of the game winner Default: "?"
        """
        self.print_grid(str(winner))
        if self.winner == 0:
            print('\33[94m', "Draw!", '\33[0m')
        else:
            print('\33[92;1m', name, "wins!", '\33[0m')
        print('\33[91;3m', "Turns taken:", self.turn, '\33[0m', '\n')

    def check_win(self):
        """
        Checks if there is a winner in the current board state.
        """
        diag_LR = sum(self.board.diagonal())
        diag_RL = sum(np.fliplr(self.board).diagonal())

        if self.size in (diag_LR, diag_RL):
            self.winner = 1
        elif -self.size in (diag_LR, diag_RL):
            self.winner = -1
        else:
            for (i, j) in zip(self.board, np.transpose(self.board)):
                row = sum(i)
                col = sum(j)
                if self.size in (row, col):
                    self.winner = 1
                elif -self.size in (row, col):
                    self.winner = -1

    def move(self, player, x, y):
        """
        Updates the current board state with a move based on x and y.
        ...
        :param player: int. The player to make a move for.
        :param x: int. Column index of the move to be taken.
        :param y: int. Row index of the move to be taken.
        """
        self.turn += 1
        self.board[y][x] = player

        if self.turn >= self.size * 2 - 1:
            self.check_win()
            self.draw = False if np.count_nonzero(self.board) != self.size ** 2 else True

        return

    def human(self, player):
        """
        Gets a move from the user.
        ...
        :param player: int. The player to make a move for.
        :return: Row and column indices of the move to be taken.
        """
        self.print_grid()
        x = -1
        y = -1

        if player == 1:
            print('\33[92;1m', "Player 1's move - X", '\33[0m', sep="", end="")
        else:
            print('\33[93;1m', "Player 2's move - O", '\33[0m', sep="", end="")
        print('\33[94m', "   Enter X and Y coordinates of move from 1 to ", self.size, " in the form:", '\33[0m',
              '\33[91;1m', " x,y", '\33[0m', sep="")

        while x < 0 or y < 0:
            try:
                user_input = input("Your move: ")
                if user_input == 'q':
                    sys.exit("User quit the program")
                else:
                    x, y = user_input.split(',')
            except ValueError:
                print('\33[91;3;1m', "Error: Enter two numbers in the form: x,y", '\33[0m')
                print('\33[91;3m', "e.g: 2,1", '\33[0m')
                print('\33[91m', "(Enter 'q' to end program)", '\33[0m')
                continue

            try:
                x = int(x) - 1
                y = self.size - int(y)
            except ValueError:
                print('\33[91;3;1m', "Error: Enter integers between 1 and", self.size, "only", '\33[0m')
                x = -1
                y = -1
                continue

            if x >= self.size or y >= self.size or x < 0 or y < 0:
                print('\33[91;3;1m', "Error: Only enter numbers between 1 and", self.size, '\33[0m')
                x = -1
                y = -1
            elif self.board[y][x] != 0:
                print('\33[91;3;1m', "Space taken. Try another space!", '\33[0m')
                x = -1
                y = -1

        return y, x

    def basic_play(self):
        """
        Plays 1 game based on the input players for testing.
        """
        self.reset_state()
        p = 1

        while self.winner == 0 and not self.draw:
            if self.players[p] == "h":
                name = 'Human Player ' + str(p) if p == 1 else 'Human Player 2'
                y, x = self.human(p)
            else:
                name = 'Random Player ' + str(p) if p == 1 else 'Random Player 2'
                positions = np.argwhere(self.board == 0)
                y, x = positions[np.random.randint(0, len(positions), size=None)]
                print('\33[34;1m', "Random Player ", p if p == 1 else 2, "'s move: (",
                      x + 1, ",", self.size - y, ")", '\33[0m', sep="")

            self.move(p, x, y)

            if 'h' not in self.players.values() and self.winner == 0 and not self.draw:
                self.print_grid()
                sleep(1)
            elif self.winner != 0 or self.draw:
                self.print_win(self.winner, name)

            p = -p

    def get_values(self):
        """
        Gets input from the user on grid size and players if grid size is below 3 or the players dict is not present.
        """
        self.players = {1: input("Is player 1 human (h), bot (b) or random (r)?: "),
                        -1: input("Is player 2 human (h), bot (b) or random (r)?: ")}
        while self.players[1] not in ('h', 'b', 'r') and self.players[-1] not in ('h', 'b', 'r'):
            print('\33[91;3;1m', "Error: Enter only 'h' or 'b' or 'r'", '\33[0m')
            self.players = {1: input("Is player 1 human (h), bot (b) or random (r)?: "),
                            -1: input("Is player 2 human (h), bot (b) or random (r)?: ")}

        while self.size < 3:
            try:
                self.size = int(input("Grid size (enter a number that is 3 or higher): "))
            except ValueError:
                print('\33[91;3;1m', "Error: Enter only integers", '\33[0m')
                continue

            if self.size < 3:
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

        self.size = N
        self.exp_rate = exp
        self.learn_rate = lrn
        self.discount_factor = dsc

        self.r_win = r_win
        self.r_draw = r_draw
        self.r_lose = r_lose

        self.current_states_1 = {}  # all moves taken -> q-values
        self.current_states_2 = {}  # all moves taken -> q-values
        self.states_q = {}  # state -> q-value
        self.epoch_count_1 = 0
        self.epoch_count_2 = 0

        self.totals = {'wins': {}, 'draws': {}, 'losses': {}}

    def counters(self, winner, player, turns):
        """
        Updates the counters in the self.total attribute.
        ...
        :param winner: int. An integer representing the game winner.
        :param player: int. An integer representing the player to count for.
        :param turns: int. The total number of turns taken this epoch.
        """
        if winner == player:
            update = self.totals.get('wins')
        elif winner == 0:
            update = self.totals.get('draws')
        else:
            update = self.totals.get('losses')
        update[turns] = update[turns] + 1 if update.get(turns) is not None else 1

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
        positions = np.argwhere(board == 0)
        action = positions[np.random.randint(0, len(positions), size=None)]

        board[action[0]][action[1]] = player
        nextBoard = board.flatten()
        boardHash = tuple(nextBoard)

        value = 0 if self.states_q.get(boardHash) is None else self.states_q.get(boardHash)

        return action, boardHash, value

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
        action = []
        boardHash = ()
        value = -999
        value_max = -999

        positions = np.argwhere(board == 0)

        for p in positions:
            nextBoard = board.copy()
            nextBoard[p[0]][p[1]] = player
            nextBoard = nextBoard.flatten()
            boardHash = tuple(nextBoard)

            value = 0 if self.states_q.get(boardHash) is None else self.states_q.get(boardHash)
            if value >= value_max:
                value_max = value
                action = p

        return action, boardHash, value

    def reward(self, winner, player):
        """
        Returns a number for the reward to be given to the input player based on the winner.
        Reward numbers defined by r_win, r_draw and r_lose.
        ...
        :param winner: int. An integer representing the game winner.
        :param player: int. An integer representing the player to be rewarded.
        :return: int. Reward numbers defined by r_win, r_draw and r_lose.
        """
        if winner == player:
            return self.r_win
        elif winner == 0:
            return self.r_draw
        else:
            return self.r_lose

    def reinforce(self, winner, player):
        """
        Updates the Q-values in self.states_q for the hash values of the current game.
        ...
        :param winner: int. An integer representing the game winner.
        :param player: int. An integer representing the player to be reinforced.
        """
        if player == 1:
            iter_a, iter_b = itertools.tee(self.current_states_1.items())
            self.epoch_count_1 += 1
        else:
            iter_a, iter_b = itertools.tee(self.current_states_2.items())
            self.epoch_count_2 += 1

        reward = self.reward(winner, player)
        next(iter_b, (0, 0))

        for key, val in iter_a:
            hsh = key
            prev_q = val
            next_hsh, next_q = next(iter_b, (0, 0))

            if self.states_q.get(hsh) is None:
                self.states_q[hsh] = 0

            if next_hsh != 0:
                nextBoard = np.reshape(next_hsh, (-1, self.size))
                action, bestHash, best_q = self.best_move(nextBoard, player)
                if best_q > next_q:
                    next_max_q = best_q
                else:
                    next_max_q = next_q
            else:
                next_max_q = 0

            new_q = (1 - self.learn_rate) * prev_q + self.learn_rate * (reward + self.discount_factor * next_max_q)
            self.states_q[hsh] = new_q

    def build_model(self, game):
        """
        Builds the self.states_q dictionary by simulating a game between two bot players both learning from each other.
        ...
        :param game: obj. The game as prepared by the TicTacToe() object.
        """
        self.current_states_1 = {}
        self.current_states_2 = {}
        p = 1

        while game.winner == 0 and not game.draw:
            if np.random.uniform(0, 1) <= self.exp_rate or self.epoch_count_2 < 100:
                action, nextHash, next_q = self.random_move(game.board, p)
            else:
                action, nextHash, next_q = self.best_move(game.board, p)

            if p == 1:
                self.current_states_1[nextHash] = next_q
            else:
                self.current_states_2[nextHash] = next_q

            x, y = action[::-1]
            game.move(p, x, y)

            p = -p

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
        self.current_states_1 = {}
        self.current_states_2 = {}
        p = 1

        names = {'r': ["Random ", '\33[34;1m'], 'b': ["Bot ", '\33[35;1m'], 'h': ["Human ", '\33[36;1m']}

        print_win = True if 'h' in game.players.values() else print_grid
        print_grid = False if 'h' in game.players.values() else print_grid

        while game.winner == 0 and not game.draw:
            if game.players[p] == "h":
                action = game.human(p)
            elif game.players[p] == "b":
                action, nextHash, next_q = self.best_move(game.board, p)
            else:
                action, nextHash, next_q = self.random_move(game.board, p)

            x, y = action[::-1]
            game.move(p, x, y)

            name = names[game.players[p]][0] + '1' if p == 1 else names[game.players[p]][0] + '2'

            if print_moves and name != 'Human':
                colour = names[game.players[p]][1]
                print(colour, "Player ", 1 if p == 1 else 2, "'s (", name, ") move: (",
                      x + 1, ",", self.size - y, ")", '\33[0m', sep="")

            if print_grid and game.winner == 0 and not game.draw:
                game.print_grid()
                sleep(1 * pause)
            elif print_win and (game.winner != 0 or game.draw):
                game.print_win(game.winner, name)
                sleep(1.5 * pause)

            p = -p

        if learn:
            self.reinforce(game.winner, test)

        self.counters(game.winner, test, game.turn)


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

        self.size = N
        self.trainGame = Base(self.size, {1: 'b', -1: 'b'})
        self.model = Learner(self.size, exp, lrn, dsc, r_win, r_draw, r_lose)

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

        if totals is not None:
            for type, values in totals.items():
                print(round(mean(values)), " average " if average else " ", type, " out of ", amount, sep='', end='')
                print(" -", values) if len(values) > 1 else print()
        if turns is not None:
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
            if print_bar else range(epochs)

        for _ in bar:
            self.trainGame.reset_state()
            self.model.build_model(self.trainGame)

        if print_results:
            print('\33[3m', "Learned", len(self.model.states_q), "possible moves", '\33[0m', end='')
            print('\33[3m', "out of 5477", '\33[0m', '\n', sep='') if self.size == 3 else print('\n')

        return self.model.states_q

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
        trial_players = {1: player_list[0], -1: player_list[1]}
        try:
            test_player = list(trial_players.keys())[list(trial_players.values()).index('b')]
        except ValueError:
            sys.exit("Bot player not defined in player list")

        moves = {i: 0 for i in moves_list} if moves_list is not None else None
        wdl = {'wins': [], 'draws': [], 'losses': []}
        wdl_moves = {'wins': {}, 'draws': {}, 'losses': {}}
        trialGame = Base(self.size, trial_players)

        number = '1' if test_player == 1 else '2'
        bar = trange(trials, desc='Testing Player ' + number, unit='trials', leave=False, file=sys.stdout) \
            if print_bar else range(trials)

        for _ in bar:
            self.model.reset_counters()
            for j in range(trial_games):
                trialGame.reset_state()
                self.model.play(trialGame, test=test_player, learn=learn, print_moves=print_moves,
                                print_grid=print_grid, pause=pause)

            for result, value in self.model.totals.items():
                wdl_moves[result] = {k: wdl_moves[result].get(k, 0) + value.get(k, 0) for k in set(value)}
                wdl[result].append(sum(value.values()))
                if result == 'losses' and moves is not None:
                    for move in moves:
                        if move in value:
                            moves[move] = value[move]

        if print_results:
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
        random_players = {1: 'r', -1: 'r'}
        trial_players = {1: player_list[0], -1: player_list[1]}
        try:
            test_player = list(trial_players.keys())[list(trial_players.values()).index('b')]
        except ValueError:
            sys.exit("Bot player not defined in player list")

        moves = {i: 0 for i in moves_list} if moves_list is not None else None
        wdl = {'wins': [], 'draws': [], 'losses': []}
        wdl_moves = {'wins': {}, 'draws': {}, 'losses': {}}
        randomGame = Base(self.size, random_players)

        number = '1' if test_player == 1 else '2'
        bar = trange(trials, desc='Control Testing Player ' + number, unit='trials', file=sys.stdout) \
            if print_bar else range(trials)

        for _ in bar:
            self.model.reset_counters()
            for j in range(trial_games):
                randomGame.reset_state()
                self.model.play(randomGame, test=test_player, learn=False, print_moves=print_moves,
                                print_grid=print_grid, pause=pause)

            for result, value in self.model.totals.items():
                wdl_moves[result] = {k: wdl_moves[result].get(k, 0) + value.get(k, 0) for k in set(value)}
                wdl[result].append(sum(value.values()))
                if result == 'losses' and moves is not None:
                    for move in moves:
                        if move in value:
                            moves[move] = value[move]

        if print_results:
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
    max_epc += 1

    def epoch_loop(rate):
        moves = []
        P1 = []
        P2 = []

        epoch_bar = tqdm(range(start_epc, max_epc, inter_epc), desc='Rate: ' + str(rate), unit='test',
                         leave=False, file=sys.stdout)

        for _ in epoch_bar:
            learned_states = parameterTester.build_model(inter_epc, print_bar=False, print_results=False)

            P1_epochs_wdl, P1_epochs_turns = parameterTester.test_model(player_list=['b', 'r'],
                                                                        print_bar=False, print_results=False)
            P2_epochs_wdl, P2_epochs_turns = parameterTester.test_model(player_list=['r', 'b'],
                                                                        print_bar=False, print_results=False)

            moves.append(len(learned_states))
            P1.append(mean(P1_epochs_wdl['wins']))
            P2.append(mean(P2_epochs_wdl['wins']))

        return moves, P1, P2

    moves_learned = {}
    P1_wins = {}
    P2_wins = {}

    bar = tqdm(range(0, 11), desc='Testing', unit='test', leave=True, file=sys.stdout)

    for i in bar:
        i = i / 10
        if test == 'e':
            parameterTester = Tester(N=3, exp=i, lrn=lrn, dsc=dsc)
        elif test == 'l':
            parameterTester = Tester(N=3, exp=exp, lrn=i, dsc=dsc)
        elif test == 'd':
            parameterTester = Tester(N=3, exp=exp, lrn=lrn, dsc=i)
        else:
            print('Error: parameter to test required (e, l, or d).')
            return

        moves_list, P1_wins_list, P2_wins_list = epoch_loop(i)
        moves_learned[i] = moves_list
        P1_wins[i] = P1_wins_list
        P2_wins[i] = P2_wins_list

    return moves_learned, P1_wins, P2_wins
