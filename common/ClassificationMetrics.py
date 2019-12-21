from sklearn.metrics import *

class ClassificationMetrics(object):
    """description of class"""
    def __init__(self, yPredict,yActual):
        self.yPredict = yPredict
        self.yActual= yActual
        if(self.yPredict is None or self.yActual is None):
            raise(Exception('yPredict and yActual cannot be null'))

    def getAccuracyScore(self):
        return accuracy_score(self.yActual,self.yPredict)

    def getPrecisionScore(self):
        return precision_score(self.yActual,self.yPredict,average='micro')

    def getRecallScore(self):
        return recall_score(self.yActual,self.yPredict,average='micro')

    def getF1Score(self):
        return f1_score(self.yActual,self.yPredict,average='micro')

    def getROCAUCScore(self):
        return roc_auc_score(self.yActual,self.yPredict,average='micro')

    def getLogLossScore(self):
        return log_loss(self.yActual,self.yPredict)

    def getClassificationReport(self):
        return  classification_report(self.yActual,self.yPredict)