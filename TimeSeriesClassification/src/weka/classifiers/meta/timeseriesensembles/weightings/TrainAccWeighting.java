
package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;

/**
 * Simply uses the modules train acc as it's weighting
 * 
 * @author James Large
 */
public class TrainAccWeighting extends ModuleWeightingScheme {

    public TrainAccWeighting() {
        uniformWeighting = true;
    }
    
    @Override
    public double[] defineWeighting(ModulePredictions trainPredictions, int numClasses) {
        return makeUniformWeighting(trainPredictions.acc, numClasses);
    }
    
}
