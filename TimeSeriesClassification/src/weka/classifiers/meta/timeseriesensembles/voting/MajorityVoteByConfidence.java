package weka.classifiers.meta.timeseriesensembles.voting;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Majority vote, however classifiers' vote is weighted by the confidence in their prediction,
 * i.e distForInst[pred]
 * 
 * @author James Large
 */
public class MajorityVoteByConfidence extends ModuleVotingScheme {
    
    public MajorityVoteByConfidence() {
        this.requiresDistsForInstances = true;
    }
    
    public MajorityVoteByConfidence(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        
        int pred;
        for(int c = 0; c < modules.length; c++){
            pred = (int) modules[c].trainResults.predClassVals[trainInstanceIndex]; 
            
            preds[pred] += modules[c].priorWeight * 
                            modules[c].posteriorWeights[pred] * 
                            modules[c].trainResults.distsForInsts[trainInstanceIndex][pred];
        }
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        
        int pred;
        for(int c = 0; c < modules.length; c++){
            pred = (int) modules[c].testResults.predClassVals[testInstanceIndex]; 
            
            preds[pred] += modules[c].priorWeight * 
                            modules[c].posteriorWeights[pred] * 
                            modules[c].testResults.distsForInsts[testInstanceIndex][pred];
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        int pred;
        double[] dist;
        for(int c = 0; c < modules.length; c++){
            dist = modules[c].classifier.distributionForInstance(testInstance);
            pred = (int)indexOfMax(dist);
            
            preds[pred] += modules[c].priorWeight * 
                            modules[c].posteriorWeights[pred] * 
                            dist[pred];
        }
        
        return normalise(preds);
    }
    
}
