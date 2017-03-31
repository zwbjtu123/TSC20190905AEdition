
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Simply uses the modules train acc as it's weighting
 * 
 * @author James Large
 */
public class TrainAcc extends ModuleWeightingScheme {

    public TrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(module.trainResults.acc, numClasses);
    }
    
}
