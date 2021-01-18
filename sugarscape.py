import numpy as np
import random
import pandas as pd

class Agent(object):

    def __init__(self, board):
        """
        Upon initialization will be placed on the supplied the board class.
        The agent is also assigned random properties.
        """
        self.board = board
        if board.allow_overlap:
            self.position = random.choice([pos for (pos,elem) in np.ndenumerate(board.array)]) # Random place on the board. Can lead to doubles
        else:
            self.position = random.choice([pos for (pos,elem) in np.ndenumerate(board.array) if not board.array.mask[pos]])
            board.array.mask[self.position] = True # Make sure it becomes occupied
        
        self.vision = np.random.randint(low = 1, high = 7) # Nr of squares it can see (not diagonal)
        self.sugarneed = np.random.randint(low = 1, high = 5) # Units of sugar needed each round
        self.sugar = 4 # Starting amount of sugar 
        self.age = 0 
        self.maxage = np.random.randint(low = 40, high = 70) # age in nr of rounds
        self.alive = True        
        self.history = pd.DataFrame(index = pd.RangeIndex(self.age,self.maxage, name = 'age'))
        # TODO: add columns for the history based on the properties (class?)
    
    def execute_turn(self):
        order = ['move', 'eat', 'account']
        arguments = {}
        returns = {}
        if (self.alive):
            for methodname in order:
                f = getattr(self, methodname)
                returns[methodname] = f(**arguments) # board is an redundant argument for account. Only eat will give a return value, namely the board with sugar removed.
        #return(returns)
        return(returns['eat']) # Returns the altered board to the game class
    
    def move(self, **kwargs):
        """
        Look in four directions to find maximum tile within the vision. Then move to it
        """
        colinds, rowinds = np.meshgrid(np.arange(self.board.size), np.arange(self.board.size))
        # Start masking outside vision. Rownr (colnr) right and then the amount of columns (rows) within vision.
        hori_vis = np.logical_and(rowinds == self.position[0], np.logical_and(colinds >= self.position[1] - self.vision, colinds <= self.position[1] + self.vision))
        verti_vis = np.logical_and(colinds == self.position[1], np.logical_and(rowinds >= self.position[0] - self.vision, rowinds <= self.position[0] + self.vision))
        # If overlap is allowed we want to mask only the invisible, otherwise also the occupied squares, in that case the board mask will contribute True's (to be masked) too
        mask =  np.logical_or(self.board.array.mask, np.logical_not(np.logical_or(hori_vis, verti_vis)))
        available_squares = np.ma.masked_array(self.board.array.base, mask = mask)
        # Register position of maximum tile (first occurrence) and move
        ind_max = np.unravel_index(np.argmax(available_squares, axis = None), available_squares.shape)
        
        if not self.board.allow_overlap:
            self.board.array.mask[self.position] = False # Unoccupy old square
            self.board.array.mask[ind_max] = True # occupy old square
        self.position = ind_max    
    
    def eat(self, **kwargs):
        """
        Eat and remove the sugar unit from the board
        """
        self.sugar = self.sugar + self.board.array[self.position]
        self.board.clear_tile(position = self.position)
    
    def account(self, **kwargs):
        """
        At the end of each round, add age and remove sugarneed, kill if below zero or exceeding maxage
        """
        self.age = self.age + 1
        self.sugar = self.sugar - self.sugarneed
        if (self.age > self.maxage or self.sugar < 0):
            self.alive = False        
        #TODO add history

class Board(object):
    """
    TODO: make this a masked array with a mask at every position where an agent is? Such that no agents on top of each other are allowed.
    """

    def __init__(self,size = 20, allow_overlap = True):
        """
        Initializes a square empty board of size*size
        allow_overlap determines whether multiple agents are allowed at a single location
        """
        self.size = size
        self.allow_overlap = allow_overlap
        values = np.zeros((size,size))
        self.array = np.ma.array(values, mask = np.full_like(values,False), dtype = int)

    def __repr__(self):
        return f'{self.array}'
    
    def create_sugar_mountains(self, maxheight = 4):
        """
        Resets the board to the state of two mountains, one south-west, the other north-east
        By using the euclidian distance to the mountain centers, which is normalized with some measure of the size of the board
        """
        quarter_increment = self.size // 4
        northeast_center = (quarter_increment, self.size - 1 - quarter_increment)
        southwest_center = (self.size - 1 - quarter_increment, quarter_increment)
        
        j_ind, i_ind = np.meshgrid(np.arange(self.size), np.arange(self.size))
        
        dist_northeast = ((i_ind - northeast_center[0])**2 + (j_ind - northeast_center[1])**2)**0.5 / self.size * 2
        dist_southwest = ((i_ind - southwest_center[0])**2 + (j_ind - southwest_center[1])**2)**0.5 / self.size * 2
        
        dist_northeast[ dist_northeast > 1 ] = 1 # Put a cap on the normalized distance.
        dist_southwest[ dist_southwest > 1 ] = 1
        
        sugar_northeast = (1 - dist_northeast) * maxheight
        sugar_southwest = (1 - dist_southwest) * maxheight
        
        self.array = np.ma.array((np.maximum(sugar_northeast, sugar_southwest)), mask = self.array.mask, dtype = int)
    
    def grow_sugar(self, growthrate = 1, position = None, non_zero_only = False):
        """
        Adds the supplied units of sugar to a position (tuple)
        If this is not supplied then to non-zero tiles only, or to all.
        NOTE: Not sure if non-zero only is correct, because tile would become inactive after eating.
        """
        if isinstance(position, tuple):
            self.array[position] = self.array[position] + growthrate
        elif non_zero_only:
            self.array = np.where(self.array == 0, self.array, self.array + growthrate)
        else:
            self.array += growthrate
    
    def clear_tile(self, position):
        """
        Clears the specified tile. Position is a 2D tuple: row, col
        """
        self.array[position] = 0
        #TODO Remove this method. Board can be directly acted upon by agent

class Game(object):
    
    def __init__(self, board):
        self.board = board
        self.growthrate = 1
    
    def initialize_agents(self, number):
        if not self.board.allow_overlap:
            assert number < self.board.array.size, f'when demanding non-overlap the number of agents {number} should be lower than squares on the board {self.board.array.size}'
        self.agents = [Agent(self.board) for i in range(number)] # Random assignment of agent properties in their definition
        
    def play_round(self):
        for i in range(len(self.agents)):
            self.agents[i].execute_turn() # Updates the board
        
        #self.board.grow_sugar() # grow sugar
   
    def display_agent_locations(self):
        if self.board.allow_overlap:
            positions = self.get_agent_attr('position')
            occupied = np.zeros_like(self.board.array.base, dtype = int)
            for position in positions:
                occupied[position] += 1
        else:
            occupied = self.board.array.mask.astype(int) # Mask has a meaning in this case
        print(occupied)

    def get_agent_attr(self, attribute):
        results = []
        for i in range(len(self.agents)):
            results.append(getattr(self.agents[i], attribute))
        return(results)
    
    def run(self):
        nodead = True
        while nodead:
            self.play_round()
            print(self.board)
            nodead = all(self.get_agent_attr('alive'))
        print('game finished')
            
            

self = Board(20, allow_overlap = False)
self.create_sugar_mountains()

game = Game(self)
game.initialize_agents(300)
#game.play_round()

# TODO: write tests. Not sure if masked movement covers all corner cases


# To-do
# let each agent record a history of sugar and position throughout its age.
# what needs to be the visual output during play?
    
#test = Game(10)
#test.initialize_agents(20)
#print(test.board)
#test.run()
#print test.get_agent_attr('alive')
#print test.get_agent_attr('maxage')
#print test.get_agent_attr('sugar')
