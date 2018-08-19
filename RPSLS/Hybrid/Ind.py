""""""
from random import randint as rnd
import copy as cp
import pickle as pkl
import os
import numpy as np
#%%
class Game(object):
    """"""
    @staticmethod
    def getHands():
        """"""
        hands = {'Rock': 0, 'Paper': 1, 'Sissors': 2, 'Lizard': 3, 'Spock': 4}
        return hands
    @staticmethod
    def getrules(a=0, b=0):
        rules = [[2, 0, 10, 10, 0],
                 [11, 2, 0, 0, 11],
                 [0, 12, 2, 12, 0],
                 [0, 13, 0, 2, 13],
                 [14, 0, 14, 0, 2]]
        return (rules[a][b])/100.0
#%%
class History(object):
    """"""
    hands = Game().getHands()
    def __init__(self):
        self.history = self.hands['Rock']
    def getHistory(self):
        return self.history
    def setHistory(self, value='Rock'):
        self.history = value
    def resetHistory(self):
        self.history = self.hands[0]
#%%
class Gene(object):
    """"""
    def setValue(self, value=0):
        self.geneValue = value
    def getValue(self):
        return self.geneValue
    def __init__(self, value=0):
        self.geneValue = value
#%%
class Chromosome(object):
    """"""
    def __init__(self, chromosomeValue=[Gene()]*5):
        self.content = [Gene()]*5
        self.content = chromosomeValue
    def isValid(self):
        totalSum = 0
        for i in range(len(self.content)):
            totalSum = totalSum + self.content[i].getValue()
        if totalSum == 0:
            return -1
        else:
            return 0
    def getValue(self):
        return self.content
    def getGene(self, index=0):
        return self.content[index]
    def setValue(self, index=0, value=Gene() ):
        if self.isValid() == -1:
            return -1
        else:
            self.content[index] = value
            return 0
#%%
class Individual(object):
    """"""
    def __init__(self, individualValue=[Chromosome()]*25):
        self.genome = individualValue
        self.history = History()
        self.fitness = 0.0
    def getGenome(self):
        return self.genome
    def getChromosome(self, index=0):
        return self.genome[index]
    def setChromosome(self, index=0, value=0):
        self.genome[index] = value
    def setFitness(self, fitness=0):
        self.fitness = fitness
    def getFitness(self):
        return self.fitness
    def setHistory(self, hand=History().getHistory()):
        self.history.setHistory(hand)
    def getHistory(self):
        return self.history
    def playHand(self, oppHistory=History()):
        active_chromosome = 5 * self.history.getHistory() + oppHistory.getHistory()
        totalSum = 0
        for i in range(5):
            totalSum += self.genome[active_chromosome].getGene(i).getValue()
        rockrange = range(0, self.genome[active_chromosome].getGene(1).getValue())
        paperRange = range(self.genome[active_chromosome].getGene(2).getValue()+1, self.genome[active_chromosome].getGene(1).getValue())
        sissorsRange = range(self.genome[active_chromosome].getGene(3).getValue() +1, self.genome[active_chromosome].getGene(2).getValue())
        lizardRange = range(self.genome[active_chromosome].getGene(4).getValue()+1, self.genome[active_chromosome].getGene(3).getValue())
        decision = rnd(0, totalSum)
        if  decision in rockrange:
            return  Game.getHands()['Rock']
        elif decision in paperRange:
            return  Game.getHands()['Paper']
        elif  decision in sissorsRange:
            return Game.getHands()['Sissors']
        elif decision in lizardRange:
            return Game.getHands()['Lizard']
        else:
            return Game.getHands()['Spock']
    def mutate(self):
        valid = -1
        chromosomePosistion = 0
        genePosistion = 0
        gene = Gene()
        chrm = Chromosome()
        while valid == -1:
            chromosomePosistion = rnd(0, 24)
            chrm = self.genome[chromosomePosistion]
            genePosistion = rnd(0, 4)
            gene = chrm.getGene(index=genePosistion)
            gene = Population.genRandGenes()
            valid = chrm.setValue(index=genePosistion, value=gene)
            
        self.genome[chromosomePosistion].setValue(index=genePosistion,value=gene)
#%%
class Population(object):
    """"""
    temprature = 100
    generationNumber = 0
    @staticmethod
    def genRandGenes():
        return Gene(value=rnd(0, 10))
    @staticmethod
    def genRandChromosome():
        dummy_chromosome = [Gene()]*5
        for i in range(5):
            dummy_chromosome[i] = Population.genRandGenes()
        return Chromosome(dummy_chromosome)
    @staticmethod
    def genRandIndividual():
        dummy_gene = [Chromosome()] * 25
        for i in range(25):
            dummy_gene[i] = Population.genRandChromosome()
        return Individual(dummy_gene)
    def __init__(self, populationSize=1000, PC=600, PM=5, survivalRate=0.2):
        self.members = []
        self.newMembers = []
        self.PC = PC
        self.PM = PM
        self.survivalRate = survivalRate
        self.populationSize = populationSize
        self.variance = 0
        for i in range(populationSize):
            dummy_indivedual = Population.genRandIndividual()
            self.members.append(dummy_indivedual)
            i = i
    def getTemprature(self):
        return self.temprature
    def fitnessEval(self):
        for i in range(len(self.members)):
            for j in range(i + 1, len(self.members)):
                for k in range(250):
                    handI = self.members[i].playHand()
                    handJ = self.members[j].playHand()
                    scoreI = Game.getrules(handI, handJ)
                    scoreJ = Game.getrules(handJ, handI)
                    self.members[i].setFitness(self.members[i].getFitness() + scoreI)
                    self.members[j].setFitness(self.members[j].getFitness() + scoreJ)
                    self.members[i].setHistory(handI)
                    self.members[j].setHistory(handJ)
                    k = k
        fit = []
        for i in range(self.populationSize):
            fit.append(self.members[i].getFitness())
        self.variance = np.var(fit)
    def selection(self):
        self.members = sorted(self.members, key=Individual.getFitness, reverse=False)
        self.newMembers = cp.deepcopy(self.members[0:10])
        self.temprature = 100 * np.tanh(self.temprature/100)
        totalSum = 0.0
        for i in range(self.populationSize):
            totalSum += self.members[i].getFitness()
        avgFitness = totalSum/len(self.members)
        fit = 0.0
        for i in range(len(self.members)):
            fit += np.exp(self.members[i].getFitness()/self.temprature)/np.exp(avgFitness/self.temprature)
            if i == 0:
                self.members[i].setFitness(fit)
            else:
                self.members[i].setFitness(fit + self.members[i-1].getFitness())
        for i in range(int(self.populationSize * self.survivalRate)):
            value = rnd(0, int(fit))
            for j in range(self.populationSize):
                if j == 0:
                    if value < self.members[j+1].getFitness():
                        self.newMembers.append(cp.deepcopy(self.members[j]))
                        break
                elif j == len(self.members)-1:
                    if value >= self.members[j-1].getFitness():
                        self.newMembers.append(cp.deepcopy(self.members[j]))
                        break
                else:
                    if (value >= self.members[j-1].getFitness()) and (value < self.members[j+1].getFitness()):
                        self.newMembers.append(cp.deepcopy(self.members[j]))
                        break
        self.members = []
        self.members = cp.deepcopy(self.newMembers[0:len(self.newMembers)])
        self.newMembers = []
        self.generationNumber += 1
    def mutation(self):
        for i in range(len(self.members)):
            if rnd(0, 1000) < self.PM:
                self.members[i].mutate()
    def crossOver(self):
        bound = len(self.members)
        chrm = [Chromosome]*25
        gene = [Gene()]*5
        while len(self.members) < self.populationSize:
            if rnd(0, 1000) < self.PC:
                parentA = self.members[rnd(0, bound)]
                parentB = self.members[rnd(0, bound)]
                for i in range(25):
                    valid = -1
                    while valid == -1:
                        for j in range(5):
                            if rnd(0, 1) == 1:
                                gene[j] = cp.deepcopy(parentA.getChromosome(index=i).getGene(index=j))
                            else:
                                gene[j] = cp.deepcopy(parentB.getChromosome(index=i).getGene(index=j))
                        chrm[i] = Chromosome(gene)
                        valid = chrm[i].isValid()
                offspring = Individual(chrm)
                self.members.append(offspring)
    def report(self):
        print "Temprature = " + str(self.temprature)
        print "Generation = " + str(self.generationNumber)
        fit = []
        for i in range(len(self.members)):
            fit.append(self.members[i].getFitness())
        fitVar = np.var(fit)
        print "Fitness Variance = "+ str(self.variance)
        print "Expected Values Variance = " + str(fitVar)
    def store(self):
        bestFitFile = os.getcwd() + "best fit.pkl"
        output = open(bestFitFile, 'wb')
        pkl.Pickler(output, pkl.HIGHEST_PROTOCOL).dump(self.members[0])
        output.close()
#%%
x = Population(populationSize=100)
while x.temprature > 1:
    x.fitnessEval()
    x.selection()
    x.report()
    x.store()
    x.crossOver()
    x.mutation()
#%%