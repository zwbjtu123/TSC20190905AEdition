package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;

/**
 *
 * Uses Confusion Entropy (CEN) to weight modules, which is a measure related to the 
 * entropy of a confusion matrix
 * 
 * Reportedly unreliable for 2 class matrices in some cases, implemented for completeness 
 * 
 * @author James Large
 */
public class CENWeighting extends ModuleWeightingScheme {

    @Override
    public double defineWeighting(ModulePredictions trainPredictions) {
        return computeCEN(trainPredictions.confusionMatrix);
    }
    
    private double computeCEN(double[][] confusionMatrix) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
