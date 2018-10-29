import numpy as np
import random

class Agent(object):

    def __init__(self, board):
        """
        Board should be 2D
        """
        self.vision = np.random.randint(low = 1, high = 7) # Nr of squares it can see (not diagonal)
        self.sugarneed = np.random.randint(low = 1, high = 5) # Units of sugar needed each round
        self.sugar = 4 # Starting amount of sugar 
        self.age = 1 
        self.maxage = np.random.randint(low = 40, high = 70) # age in nr of rounds
        self.alive = True
        self.position = random.choice([pos for (pos,elem) in np.ndenumerate(board)])# Random place on the board. Can lead to doubles. Problem?
    
    def execute_turn(self, board):
        order = ['move', 'eat', 'account']
        arguments = {'board':board}
        returns = {}
        #f = getattr(self, order[0])
        #f(**arguments)
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
        board = kwargs['board']
        rowinds = np.array(range(board.shape[0])*board.shape[0]).reshape(board.shape).T
        colinds = np.array(range(board.shape[0])*board.shape[0]).reshape(board.shape)
        # Start masking outside vision. Rownr (colnr) right and then the amount of columns (rows) within vision.
        hori_vis = np.logical_and(rowinds == self.position[0], np.logical_and(colinds >= self.position[1] - self.vision, colinds <= self.position[1] + self.vision))
        verti_vis = np.logical_and(colinds == self.position[1], np.logical_and(rowinds >= self.position[0] - self.vision, rowinds <= self.position[0] + self.vision))
        b_masked = np.ma.masked_array(board, mask = np.logical_not(np.logical_or(hori_vis, verti_vis)))
        #print b_masked
        # Register position of maximum tile (first occurrence) and move
        ind_max = np.unravel_index(np.argmax(b_masked, axis = None), b_masked.shape)
        #print ind_max
        self.position = ind_max
    
    def eat(self, **kwargs):
        """
        Eat and remove the sugar unit from the board
        """
        board = kwargs['board']
        self.sugar = self.sugar + board[self.position]
        board[self.position] = 0
        #print board
        return(board) # returns board
    
    def account(self, **kwargs):
        """
        At the end of each round, add age and remove sugarneed, kill if below zero or exceeding maxage
        """
        self.age = self.age + 1
        self.sugar = self.sugar - self.sugarneed
        if (self.age > self.maxage or self.sugar < 0):
            self.alive = False        
        

class Game(object):
    
    def __init__(self, size):
        self.board = np.array(np.random.randint(low = 0, high = 10, size = size**2))
        self.board = np.reshape(self.board, (size,size))
        #self.board = np.zeros((size, size)) # Square board
        self.growthrate = 1
    
    def initialize_agents(self, number):
        self.agents = []
        for i in xrange(number):
            self.agents.append(Agent(self.board)) # Random assignment of agent properties in their definition
        
    def play_round(self):
        for i in xrange(len(self.agents)):
            self.board = self.agents[i].execute_turn(self.board) # Update the board
        
        self.board += self.growthrate # grow sugar
    
    def get_agent_attr(self, attribute):
        results = []
        for i in xrange(len(self.agents)):
            results.append(getattr(self.agents[i], attribute))
        return(results)
    
    def run(self):
        nodead = True
        while nodead:
            self.play_round()
            print self.board
            nodead = all(self.get_agent_attr('alive'))
        print 'game finished'
            
            

# To-do
# let each agent record a history of sugar and position throughout its age.
# what needs to be the visual output during play?
    
test = Game(10)
test.initialize_agents(20)
print test.board
test.run()
print test.get_agent_attr('alive')
print test.get_agent_attr('maxage')
print test.get_agent_attr('sugar')

#agent = Agent(board = test.board)
#print agent.position
#round1 = agent.execute_turn(board = test.board)
#print round1
#test.initialize_agents(30)
#print test.get_agent_attributes('vision')

# Agents should 
