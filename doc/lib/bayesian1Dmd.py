from matplotlib import pyplot as plt
import numpy as np


class ProbabilityDistribution:
    def __init__(self,minX,maxX,stepSize):
        self.minX=minX
        self.maxX=maxX
        self.stepSize=stepSize
        self.domain=np.arange(minX,maxX,stepSize)
        self._name="Probability Distribution"
    def getDensities(self,domain):
        pass
    def getDist(self):
        return self.getDensities(self.domain)
    def getExpectation(self):
        return np.sum(np.multiply(self.domain,self.getDist()))*self.stepSize
    def getVariance(self):
        M=self.getExpectation()
        return np.sum(np.multiply(np.power(self.domain-M,2),self.getDist()))*self.stepSize
    def getMode(self):
        """Mode (statistics) The mode of a set of data values is the value that appears most often. 
        It is the value x at which its probability mass function takes its maximum value. 
        In other words, it is the value that is most likely to be sampled.          """
        return self.domain[self.getDist().argmax()]
    def getReport(self):
        report="""Probability Distribution: {}\n
        -------------------\n
        Expectation: {}\n
        Variance   : {}\n
        Mode       : {}\n
        -------------------\n
        """.format(self._name,self.getExpectation(),self.getVariance(),self.getMode())
        return report
    def plot(self):
        plt.plot(self.domain,self.getDist())


from scipy.special import gamma
class Gamma(ProbabilityDistribution):
    def __init__(self,minX,maxX,stepSize,a,b):
        ProbabilityDistribution.__init__(self,minX,maxX,stepSize)
        self.a=a
        self.b=b
        self._name="Gamma"
    def getDensities(self,domain):
        return np.power(self.b,self.a)*np.multiply(np.power(domain,self.a-1),
                                                   np.exp(-self.b*domain))/gamma(self.a)



class Normal(ProbabilityDistribution):
    def __init__(self,minX,maxX,stepSize,M,S):
        ProbabilityDistribution.__init__(self,minX,maxX,stepSize)
        self.M=M
        self.S=S
        self._name="Normal/Gaussian"
    def getDensities(self,domain):
        return (1/np.sqrt(2*np.pi*self.S**2))*np.exp(-np.power((domain-self.M),2)/(2*self.S**2))



class Beta(ProbabilityDistribution):
    def __init__(self,stepSize,a,b):
        ProbabilityDistribution.__init__(self,stepSize,1,stepSize)
        self.a=a
        self.b=b
        self._name="Beta"
    def getExpectation(self):
        return self.a/(self.a+self.b+0.0)
    def getVariance(self):
        return (self.a*self.b)/(np.power(self.a+self.b,2)*(self.a+self.b-1))
    def __beta(self,a,b):
        return gamma(a+b)/(gamma(a)*gamma(b)*1.0)
    def getDensities(self,domain):
        pdf=np.multiply(np.power(domain,self.a-1)
                           ,np.power(1-domain,self.b-1))/self.__beta(self.a,self.b)
        return pdf/pdf.sum()
    
    
class GMixtureM(ProbabilityDistribution):
    def __init__(self,minX,maxX,stepSize):
        ProbabilityDistribution.__init__(self,minX,maxX,stepSize)
        self.gausians=[]
        self.weights=[]
        self._name="GMM"
    def addGausian(self,M,S,weight):
        self.gausians.append(Normal(self.minX,self.maxX,self.stepSize,M,S))
        self.weights.append(weight)
    def getDensities(self,domain):
        cumDist=np.zeros_like(domain)
        for i,dist in enumerate(self.gausians):
            cumDist+=self.weights[i]*dist.getDensities(domain)
        return cumDist
    def getReport(self):
        report=ProbabilityDistribution.getReport(self)+"\n\n"
        for i,dist in enumerate(self.gausians):
            report+="Model {}\n{}\n".format(i,dist.getReport())
        return report
    def plot(self):
        cumDist=self.getDist()
        legend=[]
        for i,dist in enumerate(self.gausians):
            dist.plot()
            legend.append("model {}".format(i))
        legend.append("GMM")
        plt.plot(self.domain,cumDist)
        plt.grid()
        plt.legend(legend)