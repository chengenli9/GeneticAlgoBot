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
    
    # Strategic evaluation constants
    BASE_SCORE = 50000
    PENALTY_MULTIPLIER = 2.5
    
    # Food collection constants
    FOOD_BASE_COST = 1500
    FOOD_WORKER_BONUS = 800
    FOOD_WORKER_PENALTY = 120000
    FOOD_DEFICIT_MULTIPLIER = 3
    
    # Attack strategy constants
    ATTACK_BASE_EVALUATION = 1200
    DRONE_BONUS = 500
    RANGED_SOLDIER_BONUS = 600
    WORKER_ELIMINATION_BONUS = 300
    MILITARY_UNIT_PENALTY = 100000
    
    # Queen positioning constants
    QUEEN_MISSING_PENALTY = 15000
    QUEEN_BLOCKING_PENALTY = 12000
    QUEEN_IDEAL_POSITION = (2, 1)
    
    # Minimax search constants
    SEARCH_DEPTH = 3
    PRUNING_RATIO = 0.15
    MIN_NODE_LIMIT = 15
    DISTANCE_PENALTY_THRESHOLD = 3
    DISTANCE_PENALTY_BONUS = 1
    
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



    ### HOMEWORK 3 CODE ###

    ##
    # convertDistanceToMoves
    # Description: Converts a distance to the number of moves required based on ant type movement rate
    #
    # Parameters:
    #   distance - The distance to travel (float)
    #   antType - The type of ant (int)
    #
    # Return: Number of moves required (int)
    ##
    def convertDistanceToMoves(self, distance, antType):
        # Get movement capabilities
        speed = UNIT_STATS[antType][MOVEMENT]
        
        # Calculate base moves needed
        baseMoves = distance / speed
        
        # Apply strategic rounding (always round up for conservative estimates)
        tacticalMoves = math.ceil(baseMoves)
        
        # Add small penalty for longer distances (realistic movement cost)
        if tacticalMoves > self.DISTANCE_PENALTY_THRESHOLD:
            tacticalMoves += self.DISTANCE_PENALTY_BONUS
        
        return tacticalMoves

    ##
    # calculateFoodHeuristic
    # Description: Calculates the heuristic value for food collection strategy
    #
    # Parameters:
    #   parentState - The previous game state (GameState)
    #   currentState - The current game state to evaluate (GameState)
    #
    # Return: Heuristic cost for food collection strategy
    ##
    def calculateFoodHeuristic(self, parentState, currentState):
        # Early exit if goal achieved
        pid = self.myIndex if self.myIndex is not None else currentState.whoseTurn
        inv = currentState.inventories[pid]
        deficit = FOOD_GOAL - inv.foodCount
        if deficit == 0:
            return 0
        
        # Initialize cost components
        cost = self.FOOD_BASE_COST
        bonus = 0
        extra = 0
        
        # Get workers and tunnel
        workers = getAntList(currentState, pid, (WORKER,))
        tunnel = getConstrList(currentState, pid, (TUNNEL,))[0]
        
        # Check worker count
        if len(workers) == 0 or len(workers) > 2:
            return self.FOOD_WORKER_PENALTY
        
        bonus = self.FOOD_WORKER_BONUS
        
        # Process workers
        for worker in workers:
            if worker.carrying:
                dist = approxDist(worker.coords, tunnel.coords)
                cost += self.convertDistanceToMoves(dist, WORKER)
                deficit -= 1
            else:
                foods = self._getFoods(currentState)
                if len(foods) >= 2:
                    path = self._getPath(worker, foods, tunnel)
                    cost += self.convertDistanceToMoves(path, WORKER)
                    deficit -= 1
        
        # Calculate remaining food cost
        if deficit > 0:
            foods = self._getFoods(currentState)
            if len(foods) >= 2:
                extra = self._getExtraCost(foods, tunnel, deficit)
        
        cost -= bonus
        cost += extra
        
        return cost
    
    def _getFoods(self, currentState):
        """Get food sources in player territory"""
        foods = getConstrList(currentState, None, (FOOD,))
        myFoods = []
        for food in foods:
            if food.coords[1] <= 3:
                myFoods.append(food)
        return myFoods
    
    def _getPath(self, worker, foods, tunnel):
        """Calculate optimal foraging path"""
        d1 = approxDist(worker.coords, foods[0].coords)
        d2 = approxDist(worker.coords, foods[1].coords)
        
        if d1 < d2:
            return d1 + approxDist(foods[0].coords, tunnel.coords)
        else:
            return d2 + approxDist(foods[1].coords, tunnel.coords)
    
    def _getExtraCost(self, foods, tunnel, deficit):
        """Calculate remaining food cost"""
        dist = min(
            approxDist(foods[0].coords, tunnel.coords),
            approxDist(foods[1].coords, tunnel.coords)
        )
        return self.FOOD_DEFICIT_MULTIPLIER * deficit * self.convertDistanceToMoves(dist, WORKER)

    ##
    # calculateAttackHeuristic
    # Description: Calculates the heuristic value for  strategy focusing on queen elimination
    #
    # Parameters:
    #   parentState - The previous game state (GameState)
    #   currentState - The current game state to evaluate (GameState)
    #
    # Return: Heuristic cost for attack strategy
    ##
    def calculateAttackHeuristic(self, parentState, currentState):
        # Initialize attack cost
        cost = self.ATTACK_BASE_EVALUATION
        bonus = 0
        moveCost = 0
        
        # Get player info
        myId = self.myIndex if self.myIndex is not None else currentState.whoseTurn
        enemyId = 1 - myId
        
        # Check if enemy queen exists
        enemyQueens = getAntList(currentState, enemyId, (QUEEN,))
        if not enemyQueens:
            return 0
        
        # Analyze military units
        units = getAntList(currentState, myId)
        comp = self._analyzeUnits(units)
        
        # Validate strategy
        if not self._validateStrategy(comp):
            return self.MILITARY_UNIT_PENALTY
        
        # Calculate bonuses
        bonus = self._getBonuses(comp, currentState, enemyId)
        
        # Calculate movement costs with anti-looping logic
        moveCost = self._getMoveCosts(units, currentState, enemyId)
        
        # Add anti-looping penalty for units that might be stuck
        loopPenalty = self._calculateLoopPenalty(parentState, currentState, units)
        
        cost -= bonus
        cost += moveCost
        cost += loopPenalty
        
        return cost
    
    def _calculateLoopPenalty(self, parentState, currentState, units):
        """Calculate penalty for units that might be stuck in loops"""
        penalty = 0
        
        # Check if units are moving back and forth
        for unit in units:
            if unit.type in [DRONE, SOLDIER, R_SOLDIER]:
                # Check if unit moved to a position it was at before
                if parentState:
                    parentUnits = getAntList(parentState, self.myIndex if self.myIndex is not None else currentState.whoseTurn)
                    for parentUnit in parentUnits:
                        if parentUnit.type == unit.type:
                            # If unit is back where it was 2 turns ago, add penalty
                            if unit.coords == parentUnit.coords:
                                penalty += 50  # Small penalty for staying in same spot
                                break
        
        # Add penalty for units that are far from any target
        enemyQueens = getAntList(currentState, 1 - (self.myIndex if self.myIndex is not None else currentState.whoseTurn), (QUEEN,))
        enemyAnthill = getConstrList(currentState, 1 - (self.myIndex if self.myIndex is not None else currentState.whoseTurn), (ANTHILL,))[0]
        
        for unit in units:
            if unit.type in [DRONE, SOLDIER, R_SOLDIER]:
                # Calculate distance to nearest target
                if enemyQueens:
                    distToQueen = approxDist(unit.coords, enemyQueens[0].coords)
                    distToAnthill = approxDist(unit.coords, enemyAnthill.coords)
                    minDist = min(distToQueen, distToAnthill)
                    
                    # If unit is very far from targets, add penalty
                    if minDist > 8:  # Arbitrary threshold
                        penalty += 30
        
        return penalty
    
    def _analyzeUnits(self, units):
        """Analyze military unit composition"""
        comp = {'soldiers': 0, 'ranged': 0, 'drones': 0}
        
        for unit in units:
            if unit.type == SOLDIER:
                comp['soldiers'] += 1
            elif unit.type == R_SOLDIER:
                comp['ranged'] += 1
            elif unit.type == DRONE:
                comp['drones'] += 1
        
        return comp
    
    def _validateStrategy(self, comp):
        """Validate military strategy"""
        if comp['soldiers'] > 0:
            return False
        if comp['drones'] > 1:
            return False
        if comp['ranged'] > 1:
            return False
        return True
    
    def _getBonuses(self, comp, currentState, enemyId):
        """Calculate tactical bonuses"""
        bonus = 0
        
        if comp['drones'] > 0:
            bonus += self.DRONE_BONUS
        if comp['ranged'] > 0:
            bonus += self.RANGED_SOLDIER_BONUS
        
        # Bonus for eliminating enemy workers
        enemyWorkers = getAntList(currentState, enemyId, (WORKER,))
        if not enemyWorkers:
            bonus += self.WORKER_ELIMINATION_BONUS
        
        return bonus
    
    def _getMoveCosts(self, units, currentState, enemyId):
        """Calculate movement costs with improved targeting"""
        cost = 0
        
        # Get enemy targets
        enemyQueens = getAntList(currentState, enemyId, (QUEEN,))
        enemyAnthill = getConstrList(currentState, enemyId, (ANTHILL,))[0]
        enemyTunnel = getConstrList(currentState, enemyId, (TUNNEL,))[0]
        enemyWorkers = getAntList(currentState, enemyId, (WORKER,))
        
        # Calculate costs for each unit
        for unit in units:
            if unit.type == DRONE:
                # Drone always targets the queen
                if enemyQueens:
                    dist = approxDist(unit.coords, enemyQueens[0].coords)
                    cost += self.convertDistanceToMoves(dist, DRONE)
                else:
                    # If no queen, target anthill
                    dist = approxDist(unit.coords, enemyAnthill.coords)
                    cost += self.convertDistanceToMoves(dist, DRONE)
            elif unit.type == SOLDIER:
                # Regular soldier targets anthill
                dist = approxDist(unit.coords, enemyAnthill.coords)
                cost += self.convertDistanceToMoves(dist, SOLDIER)
            elif unit.type == R_SOLDIER:
                # Ranged soldier targets tunnel if workers exist, otherwise queen
                if enemyWorkers:
                    dist = approxDist(unit.coords, enemyTunnel.coords)
                elif enemyQueens:
                    dist = approxDist(unit.coords, enemyQueens[0].coords)
                else:
                    dist = approxDist(unit.coords, enemyAnthill.coords)
                cost += self.convertDistanceToMoves(dist, R_SOLDIER)
        
        return cost
    
    ##
    # calculateQueenHeuristic
    # Description: Calculates the heuristic value for queen positioning strategy
    #
    # Parameters:
    #   parentState - The previous game state (GameState)
    #   currentState - The current game state to evaluate (GameState)
    #
    # Return: Heuristic cost for queen positioning strategy
    ##
    def calculateQueenHeuristic(self, parentState, currentState):
        # Get player info
        myId = self.myIndex if self.myIndex is not None else currentState.whoseTurn
        infra = self._getInfra(currentState, myId)
        
        # Check if queen exists
        if not infra['queen']:
            return self.QUEEN_MISSING_PENALTY
        
        queen = infra['queen']
        
        # Check for blocking
        penalty = self._checkBlocking(queen, infra)
        if penalty > 0:
            return penalty
        
        # Calculate positioning cost
        dist = approxDist(queen.coords, self.QUEEN_IDEAL_POSITION)
        return self.convertDistanceToMoves(dist, QUEEN)
    
    def _getInfra(self, currentState, playerId):
        """Get player infrastructure"""
        infra = {
            'anthill': getConstrList(currentState, playerId, (ANTHILL,))[0],
            'tunnel': getConstrList(currentState, playerId, (TUNNEL,))[0],
            'queen': None,
            'foods': []
        }
        
        # Get queen
        queens = getAntList(currentState, playerId, (QUEEN,))
        if queens:
            infra['queen'] = queens[0]
        
        # Get food sources
        foods = getConstrList(currentState, None, (FOOD,))
        for food in foods:
            if food.coords[1] <= 3:
                infra['foods'].append(food)
        
        return infra
    
    def _checkBlocking(self, queen, infra):
        """Check for blocking situations"""
        pos = queen.coords
        
        # Check infrastructure blocking
        if pos == infra['anthill'].coords:
            return self.QUEEN_BLOCKING_PENALTY
        if pos == infra['tunnel'].coords:
            return self.QUEEN_BLOCKING_PENALTY
        
        # Check food blocking
        for food in infra['foods']:
            if pos == food.coords:
                return self.QUEEN_BLOCKING_PENALTY
        
        return 0

    ##
    # computeHeuristicValue
    # Description: Computes the total heuristic value by evaluating different strategic components
    #
    # Parameters:
    #   parentState - The previous game state (GameState)
    #   currentState - The current game state to evaluate (GameState)
    #
    # Return: Combined heuristic value from all strategic components
    ##
    def computeHeuristicValue(self, parentState, currentState):
        # Calculate all components
        food = self.calculateFoodHeuristic(parentState, currentState)
        attack = self.calculateAttackHeuristic(parentState, currentState)
        queen = self.calculateQueenHeuristic(parentState, currentState)
        
        return food + attack + queen

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
    def calculateStateUtility(self, parentState, currentState):
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
    # buildSearchNode
    # Description: Creates a search node with move, state, and utility information
    #
    # Parameters:
    #   move - The move that led to this state (Move)
    #   parentNode - The parent search node (dict)
    #   currentState - The current game state (GameState)
    #
    # Return: Dictionary representing a search node
    ##
    def buildSearchNode(self, move, parentNode, currentState):
        # Initialize node data structure
        nodeData = {}
        
        # Set the move that led to this state
        nodeData['move'] = move
        
        # Determine depth based on parent relationship
        if parentNode is None:
            nodeData['depth'] = 0
        else:
            nodeData['depth'] = parentNode['depth'] + 1
        
        # Store the current game state
        nodeData['currentState'] = currentState
        
        # Store reference to parent for backtracking
        nodeData['parentNode'] = parentNode
        
        # Calculate utility value for this node
        if parentNode is not None:
            # Get the previous state for comparison
            previousState = parentNode['currentState']
            # Compute utility based on state transition
            nodeData['utility'] = self.calculateStateUtility(previousState, currentState)
        else:
            # Root node has no utility calculation
            nodeData['utility'] = 0
        
        return nodeData

    ##
    # generateChildNodes
    # Description: Generates all possible child nodes from the current node by exploring legal moves
    #
    # Parameters:
    #   currentNode - The current search node (dict)
    #
    # Return: List of child nodes representing possible moves
    ##
    def generateChildNodes(self, currentNode):
        # Extract the current game state
        currentGameState = currentNode['currentState']
        
        # Obtain all possible legal moves
        legalMoves = listAllLegalMoves(currentGameState)
        
        # Initialize collection for child nodes
        childNodeCollection = []
        
        # Process each legal move to create child nodes
        for move in legalMoves:
            # Simulate the move to get next state
            nextGameState = getNextStateAdversarial(currentGameState, move)
            
            # Create a new search node for this move
            newChildNode = self.buildSearchNode(move, currentNode, nextGameState)
            
            # Add the new node to our collection
            childNodeCollection.append(newChildNode)

        # Return the complete set of child nodes
        return childNodeCollection

    ##
    # performMinimaxSearch
    # Description: Performs minimax search with alpha-beta pruning to find optimal move
    #
    # Parameters:
    #   currentNode - The current search node (dict)
    #   searchDepth - Maximum depth to search (int)
    #   pruningRatio - Ratio of nodes to consider for pruning (float)
    #   alpha - Alpha value for pruning (float)
    #   beta - Beta value for pruning (float)
    #
    # Return: Tuple of (best value, best child node)
    ##
    def performMinimaxSearch(self, currentNode, searchDepth, pruningRatio, alpha, beta):
        state = currentNode['currentState']

        # Terminal condition
        if currentNode['depth'] >= searchDepth:
            return currentNode['utility'], None
        
        # Generate children
        children = self.generateChildNodes(currentNode)
        if not children:
            return currentNode['utility'], None
            
        # Sort and prune
        children.sort(key=lambda node: node['utility'])
        total = len(children)
        limit = max(self.MIN_NODE_LIMIT, math.ceil(total * pruningRatio))
        
        # Determine player type
        # Decide maximizing/minimizing relative to our in-game index
        isMax = (state.whoseTurn == (self.myIndex if self.myIndex is not None else state.whoseTurn))
        
        # Select candidates
        if isMax:
            candidates = children[-limit:]
            bestVal = -float('inf')
        else:
            candidates = children[:limit]
            bestVal = float('inf')
            
        bestNode = None

        # Evaluate candidates
        for child in candidates:
            val, _ = self.performMinimaxSearch(child, searchDepth, pruningRatio, alpha, beta)

            if isMax:
                if val > bestVal:
                    bestVal = val
                    bestNode = child
                    alpha = max(alpha, bestVal)
            else:
                if val < bestVal:
                    bestVal = val
                    bestNode = child
                    beta = min(beta, bestVal)

            if beta <= alpha:
                break
        
        return bestVal, bestNode

    ##
    # getMove
    # Description: Gets the next move from the Player using minimax search with alpha-beta pruning
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):
        # Capture our in-game index once
        if self.myIndex is None:
            self.myIndex = currentState.whoseTurn

        # Create root node
        root = self.buildSearchNode(move=None, parentNode=None, currentState=currentState)
        
        # Perform minimax search
        val, bestNode = self.performMinimaxSearch(
            currentNode=root,
            searchDepth=self.SEARCH_DEPTH,
            pruningRatio=self.PRUNING_RATIO,
            alpha=-float('inf'),
            beta=float('inf')
        )

        return bestNode['move']


def unitTest() -> None:
    test_state: GameState = GameState.getBasicState()
    test_player: AIPlayer = AIPlayer(0)
    
    move: Move = test_player.getMove(test_state)
    next_state: GameState = getNextState(test_state, move)
    # utility_value: float = test_player.utility(test_state, next_state)

    if type(move) is not Move:
        print("getMove() test failed; invalid move returned")

    # if type(utility_value) is not int:
    #     print(f"utility() failed; value should be an integer: {utility_value}")

if __name__ == "__main__":
    unitTest()

