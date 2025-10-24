#CS 421 HW3
#Authors: Chengen Li, James Nguyen


import random
import sys
import math
import os
import json
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
# from Constants import PLAYER_ONE, PLAYER_TWO
import Constants
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
# type alias for a coordinate position
Position = tuple[int, int]


##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
    
    # Genetic Algorithm constants
    POPULATION_SIZE = 50
    GAMES_PER_GENE = 5  # Faster evaluation for more generations
    MUTATION_RATE = 0.15  # Increased for more exploration
    GENE_LENGTH = 12

    ##
    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Genetic Algorithm")
        # Internal index for this game (0 or 1). Do not overwrite self.playerId
        self.myIndex = None
        
        # Required genetic algorithm instance variables
        self.population = []
        self.currentGeneIndex = 0
        self.fitnessScores = []
        self.gamesPlayedWithCurrentGene = 0
        self.winsWithCurrentGene = 0
        self.generation = 0  # Track current generation
        self.totalGamesPlayed = 0  # Track total games for population updates
        
        # Initialize population
        self._initPopulation()
    
    # Simple genetic algorithm methods
    def _initPopulation(self):
        """Initialize population from file or create random"""
        filename = "./lic27_nguyenj25_population.txt"
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.population = []
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line.startswith('#') or not line:
                            continue
                        # Parse gene line (format: Gene_XX: weight1 weight2 ...)
                        if line.startswith('Gene_'):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                weights = [float(x) for x in parts[1].strip().split()]
                                self.population.append(weights)
            except:
                # Fallback to JSON format for backward compatibility
                try:
                    with open(filename, 'r') as f:
                        self.population = json.load(f)
                except:
                    # Try Reiss format (one gene per line)
                    try:
                        with open(filename, 'r') as f:
                            self.population = []
                            for line in f:
                                if line.strip() and not line.startswith('#'):
                                    gene = [float(x) for x in line.strip().split(',')]
                                    self.population.append(gene)
                    except:
                        self.population = [[random.uniform(-10, 10) for _ in range(self.GENE_LENGTH)] for _ in range(self.POPULATION_SIZE)]
        else:
            self.population = [[random.uniform(-10, 10) for _ in range(self.GENE_LENGTH)] for _ in range(self.POPULATION_SIZE)]
        self.fitnessScores = [0.0] * len(self.population)
    
    def mateGenes(self, parent1, parent2):
        """Simple crossover and mutation"""
        child1, child2 = parent1[:], parent2[:]
        for i in range(self.GENE_LENGTH):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return self._mutate(child1), self._mutate(child2)
    
    def _mutate(self, gene):
        """Simple mutation"""
        for i in range(len(gene)):
            if random.random() < self.MUTATION_RATE:
                gene[i] += random.gauss(0, 0.5)
                gene[i] = max(-10, min(10, gene[i]))
        return gene
    
    def _updatePopulationFile(self):
        """Update population file with current state"""
        print(f"\n--- UPDATING POPULATION FILE (Games played: {self.totalGamesPlayed}) ---")
        
        with open("./lic27_nguyenj25_population.txt", 'w') as f:
            f.write("# Genetic Algorithm Population\n")
            f.write("# Format: Each line represents one gene with 12 feature weights\n")
            f.write("# Features: Food_diff, Queen_health_diff, Drone_diff, Soldier_diff, Worker_diff, Ranged_diff, Offensive_cap, Dist_enemy_queen, Dist_my_queen, Dist_enemy_anthill, Worker_queen_dist, Queen_queen_dist\n")
            f.write(f"# Population size: {len(self.population)}, Gene length: {len(self.population[0])}, Generation: {self.generation}\n")
            f.write(f"# Total games played: {self.totalGamesPlayed}, Current gene: {self.currentGeneIndex}\n\n")
            
            for i, gene in enumerate(self.population):
                # Format each gene on one line with clean spacing
                gene_str = " ".join([f"{weight:8.4f}" for weight in gene])
                f.write(f"Gene_{i:2d}: {gene_str}\n")
    
    def createNextGeneration(self):
        """Create next generation using top 50%"""
        sorted_pop = sorted(zip(self.fitnessScores, self.population), reverse=True)
        elite = [gene for _, gene in sorted_pop[:self.POPULATION_SIZE//2]]  # Top 25 genes
        new_pop = elite[:]
        while len(new_pop) < self.POPULATION_SIZE:
            p1, p2 = random.choice(elite), random.choice(elite)
            c1, c2 = self.mateGenes(p1, p2)
            new_pop.extend([c1, c2])
        # Print generation statistics BEFORE resetting fitness scores
        print(f"\n=== GENERATION {self.generation + 1} COMPLETED ===")
        print(f"Best fitness: {max(self.fitnessScores):.3f}")
        print(f"Average fitness: {sum(self.fitnessScores)/len(self.fitnessScores):.3f}")
        print(f"Worst fitness: {min(self.fitnessScores):.3f}")
        
        self.population = new_pop[:self.POPULATION_SIZE]
        self.fitnessScores = [0.0] * len(self.population)
        self.currentGeneIndex = 0
        self.gamesPlayedWithCurrentGene = 0
        self.winsWithCurrentGene = 0
        self.generation += 1  # Increment generation counter
        
        # Update population file
        self._updatePopulationFile()
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        # Capture our in-game index once
        if self.myIndex is None:
            self.myIndex = currentState.whoseTurn
        
        numToPlace = 0
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(6, 9)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # registerWin
    #
    # This agent does'nt learn
    ##
    def registerWin(self, hasWon):
        """Override registerWin for genetic algorithm"""
        if hasWon:
            self.winsWithCurrentGene += 1
        self.gamesPlayedWithCurrentGene += 1
        self.totalGamesPlayed += 1
        
        if self.gamesPlayedWithCurrentGene >= self.GAMES_PER_GENE:
            fitness = self.winsWithCurrentGene / self.gamesPlayedWithCurrentGene
            self.fitnessScores[self.currentGeneIndex] = fitness
            
            # Print gene completion status
            print(f"Gene {self.currentGeneIndex:2d} completed: {self.winsWithCurrentGene}/{self.gamesPlayedWithCurrentGene} wins (fitness: {fitness:.3f})")
            
            self.currentGeneIndex += 1
            self.gamesPlayedWithCurrentGene = 0
            self.winsWithCurrentGene = 0
            
            # Update population file every 250 games
            if self.totalGamesPlayed % 250 == 0:
                self._updatePopulationFile()
            
            if self.currentGeneIndex >= len(self.population):
                self.createNextGeneration()

    ##
    # calculateStateUtility
    # Description: Calculates the utility value for a given game state by converting heuristic to utility
    #
    # Parameters:
    #   parentState - The previous game state (GameState)
    #   currentState - The current game state to evaluate (GameState)
    #
    # Return: A utility score representing state favorability
    ##
    def calculateStateUtility(self, currentState):
        """Use genetic algorithm utility instead of heuristic"""
        if not self.population or self.currentGeneIndex >= len(self.population):
            return 0.0
        
        # Simple feature extraction (14 features)
        myId = self.myIndex if self.myIndex is not None else currentState.whoseTurn
        enemyId = 1 - myId
        myInv = currentState.inventories[myId]
        enemyInv = currentState.inventories[enemyId]
        
        features = [
            myInv.foodCount - enemyInv.foodCount,  # 0: Food difference
            (myInv.getQueen().health if myInv.getQueen() else 0) - (enemyInv.getQueen().health if enemyInv.getQueen() else 0),  # 1: Queen health diff
            len(getAntList(currentState, myId, (DRONE,))) - len(getAntList(currentState, enemyId, (DRONE,))),  # 2: Drone diff
            len(getAntList(currentState, myId, (SOLDIER,))) - len(getAntList(currentState, enemyId, (SOLDIER,))),  # 3: Soldier diff
            len(getAntList(currentState, myId, (WORKER,))) - len(getAntList(currentState, enemyId, (WORKER,))),  # 4: Worker diff
            len(getAntList(currentState, myId, (R_SOLDIER,))) - len(getAntList(currentState, enemyId, (R_SOLDIER,))),  # 5: Ranged diff
            1 if len(getAntList(currentState, myId, (DRONE, SOLDIER, R_SOLDIER))) > len(getAntList(currentState, enemyId, (DRONE, SOLDIER, R_SOLDIER))) else 0,  # 6: Offensive capability
            self._calculateAvgDistanceToEnemyQueen(currentState, myId, enemyId),  # 7: Distance to enemy queen
            self._calculateAvgDistanceToMyQueen(currentState, myId, enemyId),  # 8: Distance to my queen
            self._calculateAvgDistanceToEnemyAnthill(currentState, myId, enemyId),  # 9: Distance to enemy anthill
            self._calculateAvgWorkerToQueenDistance(currentState, myId),  # 10: Irrelevant feature
            self._calculateQueenToQueenDistance(currentState, myId, enemyId)  # 11: Irrelevant feature
        ]
        
        # Calculate utility using current gene weights
        current_gene = self.population[self.currentGeneIndex]
        utility = sum(current_gene[i] * features[i] for i in range(len(features)))
        return utility
    
    # Helper methods for distance calculations
    def _calculateAvgDistanceToEnemyQueen(self, currentState, myId, enemyId):
        """Calculate average distance from my offensive ants to enemy queen"""
        myOffensiveAnts = getAntList(currentState, myId, (DRONE, SOLDIER, R_SOLDIER))
        enemyQueens = getAntList(currentState, enemyId, (QUEEN,))
        
        if not myOffensiveAnts or not enemyQueens:
            return 0.0
        
        total_distance = sum(approxDist(ant.coords, enemyQueens[0].coords) for ant in myOffensiveAnts)
        return total_distance / len(myOffensiveAnts)
    
    def _calculateAvgDistanceToMyQueen(self, currentState, myId, enemyId):
        """Calculate average distance from enemy offensive ants to my queen"""
        enemyOffensiveAnts = getAntList(currentState, enemyId, (DRONE, SOLDIER, R_SOLDIER))
        myQueens = getAntList(currentState, myId, (QUEEN,))
        
        if not enemyOffensiveAnts or not myQueens:
            return 0.0
        
        total_distance = sum(approxDist(ant.coords, myQueens[0].coords) for ant in enemyOffensiveAnts)
        return total_distance / len(enemyOffensiveAnts)
    
    def _calculateAvgDistanceToEnemyAnthill(self, currentState, myId, enemyId):
        """Calculate average distance from my offensive ants to enemy anthill"""
        myOffensiveAnts = getAntList(currentState, myId, (DRONE, SOLDIER, R_SOLDIER))
        enemyAnthills = getConstrList(currentState, enemyId, (ANTHILL,))
        
        if not myOffensiveAnts or not enemyAnthills:
            return 0.0
        
        total_distance = sum(approxDist(ant.coords, enemyAnthills[0].coords) for ant in myOffensiveAnts)
        return total_distance / len(myOffensiveAnts)
    
    def _calculateAvgWorkerToQueenDistance(self, currentState, myId):
        """Calculate average distance from my workers to my queen (irrelevant feature)"""
        myWorkers = getAntList(currentState, myId, (WORKER,))
        myQueens = getAntList(currentState, myId, (QUEEN,))
        
        if not myWorkers or not myQueens:
            return 0.0
        
        total_distance = sum(approxDist(worker.coords, myQueens[0].coords) for worker in myWorkers)
        return total_distance / len(myWorkers)
    
    def _calculateQueenToQueenDistance(self, currentState, myId, enemyId):
        """Calculate distance between my queen and enemy queen (irrelevant feature)"""
        myQueens = getAntList(currentState, myId, (QUEEN,))
        enemyQueens = getAntList(currentState, enemyId, (QUEEN,))
        
        if not myQueens or not enemyQueens:
            return 0.0
        
        return approxDist(myQueens[0].coords, enemyQueens[0].coords)

    

    

    ##
    # getMove
    # Description: Gets the next move by picking the node with the highest evaluation
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):
        moves = listAllLegalMoves(currentState)
        nodes = []
        
        for move in moves:
            nextState = getNextState(currentState, move)
            utilityScore = self.calculateStateUtility(nextState)
            node = self.node(move, nextState, utilityScore, None)
            nodes.append(node)
        
        best = self.bestMove(nodes)
        return best["move"]

    ## Node representation
    #
    def node(self, move, state, utility, parent, depth=1):
        return {
            "move": move,
            "state": state,
            "evaluation": (utility + depth),
            "parent": parent
        }
    
    def bestMove(self, nodes):
        return max(nodes, key=lambda x: x["evaluation"])


