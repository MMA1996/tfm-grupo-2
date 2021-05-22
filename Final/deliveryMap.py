import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np

class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
    CURRENT = 2  # indicates current cell of the agent

class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3

class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2

class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2

class DeliveryMap:
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.
        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at storageCell. The agent chooses actions (move left/right/up/down) in order to reach the deliveryCell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.05), returning to a cell the agent visited earlier a bigger penalty (-0.25) and running into
        a wall a large penalty (-0.75). The reward (+10.0) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around clueless and looses.
        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlib's plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]  # all possible actions
    
    '''Constructor: método que se ejecuta al instanciar la clase'''
    def __init__(self, arrayMap, storageCell=None):
        """ Se crea un nuevo mapa para el sistema de reparto.
            :param numpy.array arrayMap: matriz de dos dimensiones formada por celdas vacías (valor nulo),
                                         celdas con obstáculos a evitar en la medida de lo posible (valor en el intervalo (0, 1))
                                         y celdas con obstáculos inquebrantables (valor igual a la unidad).
            :param tuple storageCell: celda en la que se sitúa el almacén de paquetes. Opcional (por defecto, esquina inferior derecha)
            :param tuple deliveryCell: celda en la que se sitúa el origen del pedido. Opcional (por defecto, esquina superior izquierda)
        """
        self.arrayMap = arrayMap
        
        '''Asignación de recompensas. Todas ellas vienen referidas al tamaño del mapa'''
        self.rewardDeliveryAccomplished = 0.3 * self.arrayMap.size  # recompensa por llegar al almacén
        self.penaltyFlyingCell = -8e-4 * self.arrayMap.size  # sanción por desplazamiento
        self.penaltyReturningCell = -8e-3 * self.arrayMap.size  # sanción por volver a una celda ya sobrevolada
        self.maximumPenalty = -1.2e-2 * self.arrayMap.size  # máxima sanción
        self.__rewardThreshold = -0.45 * self.arrayMap.size  # límite negativo que puede tener la recompensa acumulada antes de dar por perdida la iteración

        self.nrows, self.ncols = self.arrayMap.shape
        self.cells = [(col, row) for col in range(self.ncols) for row in range(self.nrows)]
        self.allowableCells = [(col, row) for col in range(self.ncols) for row in range(self.nrows) if self.arrayMap[row, col] != Cell.OCCUPIED]
        
        self.storageCell = (self.ncols-1, self.nrows-1) if storageCell == None else storageCell
        self.allowableCells.remove(self.storageCell)

        # Errores por introducir una celda de almacén errónea.
        if self.storageCell not in self.cells:
            raise Exception("Error: alamcén de posición {} no está dentro del mapa".format(self.storageCell))
        if self.arrayMap[self.storageCell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: almacén de posición {} debe ser una celda libre (sin obstáculos inquebrantables)".format(self.storageCell))

        # Variables for rendering
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell
        
        deliveryCell = self.allowableCells[0]
        self.reset(deliveryCell)

    def reset(self, deliveryCell):
        """ Comienzo de un nuevo viaje. Se establece la nueva casilla del pedido y se coloca al agente en ella.
            La recompensa total pasa a ser cero.
            :param tuple deliveryCell: nuevo origen del viaje.
            :return: posición del dron en el mapa.
        """
        if deliveryCell not in self.cells:
            raise Exception("Error: el origen del pedido de posicion {} no está dentro del mapa".format(deliveryCell))
        if self.arrayMap[deliveryCell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: el origen del pedido de posición {} debe ser una celda libre (sin obstáculos inquebrantables)".format(deliveryCell))
        if deliveryCell == self.storageCell:
            raise Exception("Error: el alamcén no puede ser el origen del pedido de posición {}".format(deliveryCell))
        
        self.__previousCell = self.__currentCell = deliveryCell
        self.__totalReward = 0.0  # recompensa acumulada (nula, pues se trata del comienzo de una nueva iteración de entrenamiento)
        self.__visitedCells = set()  # conjunto de celdas ya visitadas (vacío, pues se trata del comienzo de una nueva iteración)

        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.arrayMap.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__currentCell, "rs", markersize=30)  # el origen del pedido aparece como un cuadrado rojo con la palabra "Deliv"
            self.__ax1.text(*self.__currentCell, "Deliv", ha="center", va="center", color="white")
            self.__ax1.plot(*self.storageCell, "gs", markersize=30)  # el alamcén aparece como un cuadrado verde con la palabra "Stor"
            self.__ax1.text(*self.storageCell, "Stor", ha="center", va="center", color="white")
            self.__ax1.imshow(self.arrayMap, cmap="Reds") # Los obstáculos se visualizan con un degradado de colores rojos.
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe() #posición del agente (deliveryCell)

    def __draw(self):
        """ Se dibuja una línea desde la celda anterior a la actual. """
        self.__ax1.plot(*zip(*[self.__previousCell, self.__currentCell]), "bo-")  # las celdas por las que ha pasado el dron aparecen con un punto azul en el centro.
        self.__ax1.plot(*self.__currentCell, "ro")  # la celda actual aparece indicada con un círculo rojo en el centro.
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING):
        """ Record what will be rendered during play and/or training.
            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.
            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__totalReward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __execute(self, action):
        """ Se ejecuta la acción correspondiente (cambio de celda)
            :param Action action: direction in which the agent will move
            :return float: reward or penalty which results from the action
        """
        possibleActions = self.__possibleActions(self.__currentCell)

        if not possibleActions:
            reward = self.__rewardThreshold - 1  # cannot move anywhere, force end of game
        elif action in possibleActions:
            col, row = self.__currentCell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1

            self.__previousCell = self.__currentCell
            self.__currentCell = (col, row)

            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__currentCell == self.storageCell:
                reward = self.rewardDeliveryAccomplished  # maximum reward when reaching the storage cell
            elif self.__currentCell in self.__visitedCells:
                reward = self.penaltyReturningCell + self.maximumPenalty*self.arrayMap[self.__currentCell[::-1]]  # penalty when returning to a cell which was visited earlier
            else:
                reward = self.penaltyFlyingCell + self.maximumPenalty*self.arrayMap[self.__currentCell[::-1]]  # penalty for a move which did not result in finding the exit cell

            self.__visitedCells.add(self.__currentCell)
        else:
            reward = self.maximumPenalty  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def __possibleActions(self, cell=None):
        """ Create a list with all possible actions from 'cell', avoiding the maze's edges and walls.
            :param tuple cell: location of the agent (optional, else use current cell)
            :return list: all possible actions
        """
        if cell is None:
            col, row = self.__currentCell
        else:
            col, row = cell

        possibleActions = DeliveryMap.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        if row == 0 or (row > 0 and self.arrayMap[row - 1, col] == Cell.OCCUPIED):
            possibleActions.remove(Action.MOVE_UP)
        if row == self.nrows - 1 or (row < self.nrows - 1 and self.arrayMap[row + 1, col] == Cell.OCCUPIED):
            possibleActions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.arrayMap[row, col - 1] == Cell.OCCUPIED):
            possibleActions.remove(Action.MOVE_LEFT)
        if col == self.ncols - 1 or (col < self.ncols - 1 and self.arrayMap[row, col + 1] == Cell.OCCUPIED):
            possibleActions.remove(Action.MOVE_RIGHT)

        return possibleActions

    def __status(self):
        """ Return the game status.
            :return Status: current game status (WIN, LOSE, PLAYING)
        """
        if self.__currentCell == self.storageCell:
            return Status.WIN

        if self.__totalReward < self.__rewardThreshold:  # force end of game after to much loss
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):
        """ Devuelve el estado del mapa, es decir, la posición del agente
            :return numpy.array [1][2]: posición actual del agente
        """
        return np.array([[*self.__currentCell]])

    def play(self, model, deliveryCell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.
            :param class AbstractModel model: the prediction model to use
            :param tuple storageCell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        self.reset(deliveryCell)

        state = self.__observe()

        no_steps = 0
        while True:
            no_steps += 1
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return status, self.__totalReward, no_steps

    def checkWinAll(self, model, previousWin):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = Render.NOTHING  # avoid rendering anything during execution of the check games

        win = 0
        lose = 0
        
        cellsRanking = {}
        sumForProb = 0
        for cell in self.allowableCells:
            if self.play(model, cell)[0] == Status.WIN:
                win += 1
            else:
                lose += 1

        self.__render = previous  # restore previous rendering setting

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        result = True if lose == 0 else False
        
        if lose == 0:
            previousWin += 1
        else:
            previousWin = 0          

        return result, win / (win + lose), previousWin

    def render_q(self, model):
        """ Render the recommended action(s) for each cell as provided by 'model'.
        :param class AbstractModel model: the prediction model to use
        """
        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.TRAINING:
            nrows, ncols = self.arrayMap.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.storageCell, "gs", markersize=30)  # exit is a big green square
            self.__ax2.text(*self.storageCell, "Stor", ha="center", va="center", color="white")

            for cell in self.allowableCells:
                q = model.q(cell) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.2
                    if action == Action.MOVE_RIGHT:
                        dx = +0.2
                    if action == Action.MOVE_UP:
                        dy = -0.2
                    if action == Action.MOVE_DOWN:
                        dy = 0.2

                    # color (red to green) represents the certainty
                    color = clip((q[action] - -1)/(1 - -1))

                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__ax2.imshow(self.arrayMap, cmap="Reds")
            self.__ax2.get_figure().canvas.draw()
    
    def saveBestMove(self, carpeta_caso, experiment):
        
        self.__ax2.get_figure().savefig(carpeta_caso + '/Best moves/Best move ' + experiment + '.png')