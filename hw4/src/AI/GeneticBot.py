#CS 421 HW4 Genetic Algorithm Player
#Authors: Chengen Li, James Nguyen

import random
import sys
import math
import os
import json
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
import Constants
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


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
        self.totalWins = 0  # Track total wins across all generations
        
        # Initialize population
        self._initPopulation()
    
    ##
    # _initPopulation
    #
    # Initializes the genetic algorithm population by loading from file or creating random genes
    # Attempts multiple file formats for backward compatibility
    #
    # Parameters: None
    #
    # Return: None (sets self.population and self.fitnessScores)
    ##
    def _initPopulation(self):
        filename = "./lic27_nguyenj25_population.txt"
        if os.path.exists(filename):
            try:
                # Try primary format: Gene_XX: weight1 weight2 ...
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
                    # Try comma-separated format (one gene per line)
                    try:
                        with open(filename, 'r') as f:
                            self.population = []
                            for line in f:
                                if line.strip() and not line.startswith('#'):
                                    gene = [float(x) for x in line.strip().split(',')]
                                    self.population.append(gene)
                    except:
                        # Create random population if all file formats fail
                        self.population = [[random.uniform(-10, 10) for _ in range(self.GENE_LENGTH)] for _ in range(self.POPULATION_SIZE)]
        else:
            # No file exists, create random population
            self.population = [[random.uniform(-10, 10) for _ in range(self.GENE_LENGTH)] for _ in range(self.POPULATION_SIZE)]
        self.fitnessScores = [0.0] * len(self.population)
    
    ##
    # mateGenes
    #
    # Performs crossover and mutation on two parent genes to create two children
    # Uses uniform crossover (50% chance to swap each gene position)
    #
    # Parameters:
    #   parent1 - First parent gene (list of floats)
    #   parent2 - Second parent gene (list of floats)
    #
    # Return: Tuple of (child1, child2) after crossover and mutation
    ##
    def mateGenes(self, parent1, parent2):
        # Create copies to avoid modifying originals
        child1, child2 = parent1[:], parent2[:]
        
        # Generate random crossover mask (no loops)
        crossover_mask = [random.random() < 0.5 for _ in range(self.GENE_LENGTH)]
        
        # Apply crossover using vectorized operations (no explicit loops)
        child1 = [child2[i] if mask else child1[i] for i, mask in enumerate(crossover_mask)]
        child2 = [child1[i] if mask else child2[i] for i, mask in enumerate(crossover_mask)]
        
        return self._mutate(child1), self._mutate(child2)
    
    ##
    # _mutate
    #
    # Applies Gaussian mutation to a gene with probability MUTATION_RATE
    # Clamps mutated values to [-10, 10] range
    #
    # Parameters:
    #   gene - Gene to mutate (list of floats)
    #
    # Return: Mutated gene (list of floats)
    ##
    def _mutate(self, gene):
        # Apply mutation to each gene position
        for i in range(len(gene)):
            if random.random() < self.MUTATION_RATE:
                # Add Gaussian noise with std dev 0.5
                gene[i] += random.gauss(0, 0.5)
                # Clamp to valid range
                gene[i] = max(-10, min(10, gene[i]))
        return gene
    
    ##
    # _updatePopulationFile
    #
    # Writes current population state to file with metadata and statistics
    # Includes generation info, game counts, and formatted gene weights
    #
    # Parameters: None
    #
    # Return: None (writes to file)
    ##
    def _updatePopulationFile(self):
        print(f"\n--- UPDATING POPULATION FILE (Games played: {self.totalGamesPlayed}) ---")
        
        # Calculate overall win rate using tracked total wins
        overall_win_rate = self.totalWins / self.totalGamesPlayed if self.totalGamesPlayed > 0 else 0.0
        
        with open("./lic27_nguyenj25_population.txt", 'w') as f:
            # Write header information
            f.write("# Genetic Algorithm Population\n")
            f.write("# Format: Each line represents one gene with 12 feature weights\n")
            f.write("# Features: Food_diff, Queen_health_diff, Drone_diff, Soldier_diff, Worker_diff, Ranged_diff, Offensive_cap, Dist_enemy_queen, Dist_my_queen, Dist_enemy_anthill, Worker_queen_dist, Queen_queen_dist\n")
            f.write(f"# Population size: {len(self.population)}, Gene length: {len(self.population[0])}, Generation: {self.generation}\n")
            f.write(f"# Total games played: {self.totalGamesPlayed}\n")
            f.write(f"# Overall win rate: {overall_win_rate:.3f} ({self.totalWins}/{self.totalGamesPlayed})\n\n")
            
            # Write each gene with formatted weights
            for i, gene in enumerate(self.population):
                gene_str = " ".join([f"{weight:8.4f}" for weight in gene])
                f.write(f"Gene_{i:2d}: {gene_str}\n")
    
    ##
    # createNextGeneration
    #
    # Creates next generation using elitist selection (top 50% survive)
    # Elite genes are used to create offspring through crossover and mutation
    # Resets fitness scores and generation counters
    #
    # Parameters: None
    #
    # Return: None (updates self.population and related variables)
    ##
    def createNextGeneration(self):
        # Sort population by fitness (descending)
        sorted_pop = sorted(zip(self.fitnessScores, self.population), reverse=True)
        # Select top 50% as elite
        elite = [gene for _, gene in sorted_pop[:self.POPULATION_SIZE//2]]
        new_pop = elite[:]
        
        # Generate offspring until population is full
        while len(new_pop) < self.POPULATION_SIZE:
            p1, p2 = random.choice(elite), random.choice(elite)
            c1, c2 = self.mateGenes(p1, p2)
            new_pop.extend([c1, c2])
        
        # Print generation statistics BEFORE resetting fitness scores
        print(f"\n=== GENERATION {self.generation + 1} COMPLETED ===")
        print(f"Best fitness: {max(self.fitnessScores):.3f}")
        print(f"Average fitness: {sum(self.fitnessScores)/len(self.fitnessScores):.3f}")
        print(f"Worst fitness: {min(self.fitnessScores):.3f}")
        
        # Update population and reset counters
        self.population = new_pop[:self.POPULATION_SIZE]
        self.fitnessScores = [0.0] * len(self.population)
        self.currentGeneIndex = 0
        self.gamesPlayedWithCurrentGene = 0
        self.winsWithCurrentGene = 0
        self.generation += 1
        
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
        if currentState.phase == SETUP_PHASE_1:    # Place on my side (rows 0-3)
            numToPlace = 11  # 1 anthill + 1 tunnel + 9 grass
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)  # My side
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True  # Mark as occupied
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   # Place on enemy side (rows 6-9)
            numToPlace = 2  # 2 food sources
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(6, 9)  # Enemy side
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True  # Mark as occupied
                moves.append(move)
            return moves
        else:
            return [(0, 0)]  # Fallback
    
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
        # Simple random attack selection
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # registerWin
    #
    # Updates fitness scores for genetic algorithm based on game outcomes
    # Tracks wins per gene and triggers generation advancement when complete
    #
    # Parameters:
    #   hasWon - Boolean indicating if this player won the game
    #
    # Return: None (updates fitness scores and generation state)
    ##
    def registerWin(self, hasWon):
        # Update win count and game statistics
        if hasWon:
            self.winsWithCurrentGene += 1
            self.totalWins += 1
        self.gamesPlayedWithCurrentGene += 1
        self.totalGamesPlayed += 1
        
        # Check if current gene evaluation is complete
        if self.gamesPlayedWithCurrentGene >= self.GAMES_PER_GENE:
            # Calculate fitness as win rate
            fitness = self.winsWithCurrentGene / self.gamesPlayedWithCurrentGene
            self.fitnessScores[self.currentGeneIndex] = fitness
            
            # Print gene completion status
            print(f"Gene {self.currentGeneIndex:2d} completed: {self.winsWithCurrentGene}/{self.gamesPlayedWithCurrentGene} wins (fitness: {fitness:.3f})")
            
            # Move to next gene
            self.currentGeneIndex += 1
            self.gamesPlayedWithCurrentGene = 0
            self.winsWithCurrentGene = 0
            
            # Update population file periodically
            if self.totalGamesPlayed % 250 == 0:
                self._updatePopulationFile()
            
            # Check if generation is complete
            if self.currentGeneIndex >= len(self.population):
                self.createNextGeneration()

    ##
    # calculateStateUtility
    #
    # Calculates utility value for a game state using genetic algorithm weights
    # Extracts 12 features and applies current gene weights to compute utility
    #
    # Parameters:
    #   currentState - The game state to evaluate (GameState)
    #
    # Return: Utility score representing state favorability (float)
    ##
    def calculateStateUtility(self, currentState):
        if not self.population or self.currentGeneIndex >= len(self.population):
            return 0.0
        
        myId = self.myIndex if self.myIndex is not None else currentState.whoseTurn
        enemyId = 1 - myId
        myInv = currentState.inventories[myId]
        enemyInv = currentState.inventories[enemyId]
        
        features = [
            myInv.foodCount - enemyInv.foodCount,  # 0: Food advantage
            (myInv.getQueen().health if myInv.getQueen() else 0) - (enemyInv.getQueen().health if enemyInv.getQueen() else 0),  # 1: Queen health advantage
            len(getAntList(currentState, myId, (DRONE,))) - len(getAntList(currentState, enemyId, (DRONE,))),  # 2: Drone advantage
            len(getAntList(currentState, myId, (SOLDIER,))) - len(getAntList(currentState, enemyId, (SOLDIER,))),  # 3: Soldier advantage
            len(getAntList(currentState, myId, (WORKER,))) - len(getAntList(currentState, enemyId, (WORKER,))),  # 4: Worker advantage
            len(getAntList(currentState, myId, (R_SOLDIER,))) - len(getAntList(currentState, enemyId, (R_SOLDIER,))),  # 5: Ranged advantage
            1 if len(getAntList(currentState, myId, (DRONE, SOLDIER, R_SOLDIER))) > len(getAntList(currentState, enemyId, (DRONE, SOLDIER, R_SOLDIER))) else 0,  # 6: Overall military advantage
            self._calculateAvgDistanceToEnemyQueen(currentState, myId, enemyId),  # 7: Distance to enemy queen
            self._calculateAvgDistanceToMyQueen(currentState, myId, enemyId),  # 8: Enemy distance to my queen
            self._calculateAvgDistanceToEnemyAnthill(currentState, myId, enemyId),  # 9: Distance to enemy anthill
            self._calculateAvgWorkerToQueenDistance(currentState, myId),  # 10: Irrelevant feature
            self._calculateQueenToQueenDistance(currentState, myId, enemyId)  # 11: Irrelevant feature
        ]
        
        current_gene = self.population[self.currentGeneIndex]
        utility = sum(current_gene[i] * features[i] for i in range(len(features)))
        return utility
    
    ##
    # _calculateAvgDistanceToEnemyQueen
    #
    # Calculates average distance from my offensive ants to enemy queen
    # Used as feature for genetic algorithm utility calculation
    #
    # Parameters:
    #   currentState - Current game state (GameState)
    #   myId - My player ID (int)
    #   enemyId - Enemy player ID (int)
    #
    # Return: Average distance (float), 0.0 if no ants or queen
    ##
    def _calculateAvgDistanceToEnemyQueen(self, currentState, myId, enemyId):
        # Get my offensive units and enemy queen
        myOffensiveAnts = getAntList(currentState, myId, (DRONE, SOLDIER, R_SOLDIER))
        enemyQueens = getAntList(currentState, enemyId, (QUEEN,))
        
        # Return 0 if no units or queen
        if not myOffensiveAnts or not enemyQueens:
            return 0.0
        
        # Calculate average distance to enemy queen
        total_distance = sum(approxDist(ant.coords, enemyQueens[0].coords) for ant in myOffensiveAnts)
        return total_distance / len(myOffensiveAnts)
    
    ##
    # _calculateAvgDistanceToMyQueen
    #
    # Calculates average distance from enemy offensive ants to my queen
    # Used as feature for genetic algorithm utility calculation
    #
    # Parameters:
    #   currentState - Current game state (GameState)
    #   myId - My player ID (int)
    #   enemyId - Enemy player ID (int)
    #
    # Return: Average distance (float), 0.0 if no ants or queen
    ##
    def _calculateAvgDistanceToMyQueen(self, currentState, myId, enemyId):
        # Get enemy offensive units and my queen
        enemyOffensiveAnts = getAntList(currentState, enemyId, (DRONE, SOLDIER, R_SOLDIER))
        myQueens = getAntList(currentState, myId, (QUEEN,))
        
        # Return 0 if no units or queen
        if not enemyOffensiveAnts or not myQueens:
            return 0.0
        
        # Calculate average distance to my queen
        total_distance = sum(approxDist(ant.coords, myQueens[0].coords) for ant in enemyOffensiveAnts)
        return total_distance / len(enemyOffensiveAnts)
    
    ##
    # _calculateAvgDistanceToEnemyAnthill
    #
    # Calculates average distance from my offensive ants to enemy anthill
    # Used as feature for genetic algorithm utility calculation
    #
    # Parameters:
    #   currentState - Current game state (GameState)
    #   myId - My player ID (int)
    #   enemyId - Enemy player ID (int)
    #
    # Return: Average distance (float), 0.0 if no ants or anthill
    ##
    def _calculateAvgDistanceToEnemyAnthill(self, currentState, myId, enemyId):
        # Get my offensive units and enemy anthill
        myOffensiveAnts = getAntList(currentState, myId, (DRONE, SOLDIER, R_SOLDIER))
        enemyAnthills = getConstrList(currentState, enemyId, (ANTHILL,))
        
        # Return 0 if no units or anthill
        if not myOffensiveAnts or not enemyAnthills:
            return 0.0
        
        # Calculate average distance to enemy anthill
        total_distance = sum(approxDist(ant.coords, enemyAnthills[0].coords) for ant in myOffensiveAnts)
        return total_distance / len(myOffensiveAnts)
    
    ##
    # _calculateAvgWorkerToQueenDistance
    #
    # Calculates average distance from my workers to my queen
    # Used as feature for genetic algorithm utility calculation
    #
    # Parameters:
    #   currentState - Current game state (GameState)
    #   myId - My player ID (int)
    #
    # Return: Average distance (float), 0.0 if no workers or queen
    ##
    def _calculateAvgWorkerToQueenDistance(self, currentState, myId):
        # Get my workers and queen
        myWorkers = getAntList(currentState, myId, (WORKER,))
        myQueens = getAntList(currentState, myId, (QUEEN,))
        
        # Return 0 if no workers or queen
        if not myWorkers or not myQueens:
            return 0.0
        
        # Calculate average distance from workers to queen
        total_distance = sum(approxDist(worker.coords, myQueens[0].coords) for worker in myWorkers)
        return total_distance / len(myWorkers)
    
    ##
    # _calculateQueenToQueenDistance
    #
    # Calculates distance between my queen and enemy queen
    # Used as feature for genetic algorithm utility calculation
    #
    # Parameters:
    #   currentState - Current game state (GameState)
    #   myId - My player ID (int)
    #   enemyId - Enemy player ID (int)
    #
    # Return: Distance (float), 0.0 if no queens
    ##
    def _calculateQueenToQueenDistance(self, currentState, myId, enemyId):
        # Get both queens
        myQueens = getAntList(currentState, myId, (QUEEN,))
        enemyQueens = getAntList(currentState, enemyId, (QUEEN,))
        
        # Return 0 if either queen is missing
        if not myQueens or not enemyQueens:
            return 0.0
        
        # Calculate distance between queens
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

    ##
    # node
    #
    # Creates a node representation for move evaluation
    # Combines utility score with depth for move selection
    #
    # Parameters:
    #   move - The move being evaluated (Move)
    #   state - Resulting game state (GameState)
    #   utility - Utility score for the state (float)
    #   parent - Parent node (dict or None)
    #   depth - Depth in search tree (int, default 1)
    #
    # Return: Node dictionary with move, state, evaluation, and parent
    ##
    def node(self, move, state, utility, parent, depth=1):
        return {
            "move": move,
            "state": state,
            "evaluation": (utility + depth),  # Add depth bonus to utility
            "parent": parent
        }
    
    ##
    # bestMove
    #
    # Selects the node with the highest evaluation score
    #
    # Parameters:
    #   nodes - List of node dictionaries
    #
    # Return: Node with highest evaluation score
    ##
    def bestMove(self, nodes):
        return max(nodes, key=lambda x: x["evaluation"])
