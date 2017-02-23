package weka.classifiers.meta.timeseriesensembles.voting;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Each class' probability is defined as the average of each classifier that predicts this class' weighted 
 * confidence that the instance is of this class
 * 
 * @author James Large
 */
public class AverageVoteByConfidence extends ModuleVotingScheme {
    
    public AverageVoteByConfidence() {
        this.requiresDistsForInstances = true;
    }
    
    public AverageVoteByConfidence(int numClasses) {
        this.numClasses = numClasses;
        this.requiresDistsForInstances = true;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        int[] numPredsForClass = new int[numClasses];
        
        int pred; 
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].trainResults.predClassVals[trainInstanceIndex]; 
            ++numPredsForClass[pred];
            
            preds[pred] += modules[m].priorWeight * 
                    modules[m].posteriorWeights[pred] * 
                    modules[m].trainResults.distsForInsts[trainInstanceIndex][pred];
        }
        
        for (int c = 0; c < numClasses; c++) 
            if (numPredsForClass[c] != 0)
                preds[c]/=numPredsForClass[c];
    
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        int[] numPredsForClass = new int[numClasses];
        
        int pred; 
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].testResults.predClassVals[testInstanceIndex]; 
            ++numPredsForClass[pred];
            
            preds[pred] += modules[m].priorWeight * 
                    modules[m].posteriorWeights[pred] * 
                    modules[m].testResults.distsForInsts[testInstanceIndex][pred];
        }
        
        for (int c = 0; c < numClasses; c++) 
            if (numPredsForClass[c] != 0)
                preds[c]/=numPredsForClass[c];
    
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        int[] numPredsForClass = new int[numClasses];
        
        double[] dist;
        int pred; 
        for(int m = 0; m < modules.length; m++){
            dist = modules[m].getClassifier().distributionForInstance(testInstance);
            storeModuleTestResult(modules[m], dist);
            
            pred = (int)indexOfMax(dist);
            ++numPredsForClass[pred];
            
            preds[pred] += modules[m].priorWeight * 
                        modules[m].posteriorWeights[pred] * 
                        dist[pred];
        }
        
        for (int c = 0; c < numClasses; c++) 
            if (numPredsForClass[c] != 0)
                preds[c]/=numPredsForClass[c];
    
        return normalise(preds);
    }
}
