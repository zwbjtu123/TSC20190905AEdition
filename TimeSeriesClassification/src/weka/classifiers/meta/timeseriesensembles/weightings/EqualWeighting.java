package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;


/**
 *
 * Gives equal weights to all modules, i.e simple majority vote
 * 
 * @author James Large
 */
public class EqualWeighting extends ModuleWeightingScheme {

    public EqualWeighting() {
        uniformWeighting = true;
    }
    
    @Override
    public double[] defineWeighting(ModulePredictions trainPredictions, int numClasses) {
        return makeUniformWeighting(1.0, numClasses);
    }
    
}
