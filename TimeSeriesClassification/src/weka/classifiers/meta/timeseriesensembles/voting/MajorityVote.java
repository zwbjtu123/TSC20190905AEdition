
package weka.classifiers.meta.timeseriesensembles.voting;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Simple majority vote, gets the prediction of each module and adds it's weight
 * to that class' overall weight
 * 
 * @author James Large
 */
public class MajorityVote extends ModuleVotingScheme {

    public MajorityVote() {
        this.requiresDistsForInstances = false;
    }
    
    public MajorityVote(int numClasses) {
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
        
        int pred;
        for(int c = 0; c < modules.length; c++){
            pred = (int) modules[c].trainResults.predClassVals[trainInstanceIndex]; 
            
            preds[pred] += modules[c].priorWeight * 
                           modules[c].posteriorWeights[pred];
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
                           modules[c].posteriorWeights[pred];
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        int pred;
        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = modules[m].getClassifier().distributionForInstance(testInstance); 
            storeModuleTestResult(modules[m], dist);
            
            pred = (int)indexOfMax(dist);
            preds[pred] += modules[m].priorWeight * 
                           modules[m].posteriorWeights[pred];
        }
        
        return normalise(preds);
    }
}
