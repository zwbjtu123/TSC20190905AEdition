
package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;

/**
 * Simply uses the modules train acc as it's weighting
 * 
 * @author James Large
 */
public class TrainAccWeighting extends ModuleWeightingScheme {

    @Override
    public double defineWeighting(ModulePredictions trainPredictions) {
        return trainPredictions.acc;
    }
    
}
