
package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;

/**
 * Simply uses the modules train acc as it's weighting
 * 
 * @author James Large
 */
public class TrainAcc extends ModuleWeightingScheme {

    public TrainAcc() {
        uniformWeighting = true;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(module.trainResults.acc, numClasses);
    }
    
}
