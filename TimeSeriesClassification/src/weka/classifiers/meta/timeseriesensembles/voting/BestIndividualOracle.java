package weka.classifiers.meta.timeseriesensembles.voting;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;
import weka.core.Instance;

/**
 * TODO what if there's tie for best? UNTESTED
 * 
 * The ensemble's distribution for an instance is equal to the single 'best' individual,
 * as defined by THEIR TEST ACCURACY. Results must have been read from file (i.e test preds
 * already exist at train time) Weighting scheme is irrelevant, only considers accuracy.
 * 
 * Mostly just written so that I can do the best individual within the existing framework for 
 * later testing
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class BestIndividualOracle extends ModuleVotingScheme {

    int bestModule;
    
    public BestIndividualOracle() {
        this.requiresDistsForInstances = true;
    }
    
    public BestIndividualOracle(int numClasses) {
        this.numClasses = numClasses;
        this.requiresDistsForInstances = true;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        super.trainVotingScheme(modules, numClasses);
        
        double bestAcc = -1;
        for (int m = 0; m < modules.length; ++m) {         
            if (modules[m].testResults.acc > bestAcc) {
                bestAcc = modules[m].testResults.acc;
                bestModule = m;
            }
        }
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        return modules[bestModule].trainResults.distsForInsts[trainInstanceIndex];
    }

    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        return modules[bestModule].testResults.distsForInsts[testInstanceIndex];
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        return modules[bestModule].getClassifier().distributionForInstance(testInstance);
    }
}
