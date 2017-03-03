package weka.classifiers.meta.timeseriesensembles.voting;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;
import weka.core.Instance;

/**
 *
 * TODO what if there's tie for best?  UNTESTED
 * 
 * The ensemble's distribution for an instance is equal to the single 'best' individual,
 * as defined by whatever (uniform) weighting scheme is being used. 
 * 
 * Mostly just written so that I can do the best individual within the existing framework for 
 * later testing
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class BestIndividualTrain extends ModuleVotingScheme {

    int bestModule;
    
    public BestIndividualTrain() {
    }
    
    public BestIndividualTrain(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        super.trainVotingScheme(modules, numClasses);
        
        double bestWeight = -1;
        for (int m = 0; m < modules.length; ++m) {
            
            //checking that the weights are uniform
            double prevWeight = modules[m].posteriorWeights[0];
            for (int c = 1; c < numClasses; ++c)  {
                if (prevWeight == modules[m].posteriorWeights[c])
                    prevWeight = modules[m].posteriorWeights[c];
                else 
                    throw new Exception("BestIndividualTrain cannot be use with non-uniform weighitn schemes");
            }
            
            if (modules[m].posteriorWeights[0] > bestWeight) {
                bestWeight = modules[m].posteriorWeights[0];
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