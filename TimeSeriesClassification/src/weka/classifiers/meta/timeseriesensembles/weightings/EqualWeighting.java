package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;


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
    public double[] defineWeighting(EnsembleModule trainPredictions, int numClasses) {
        return makeUniformWeighting(1.0, numClasses);
    }
    
}
