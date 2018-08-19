import Rnd as nd
import pickle as pkl

x = nd.Individual()



bestFitFile = os.getcwd() + "best fit.pkl"
output = open(bestFitFile, 'wb')
x = pkl.pickle.load(output)
output.close()
