package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;


/**
 *
 * Gives equal weights to all modules, i.e simple majority vote
 * 
 * @author James Large
 */
public class EqualWeighting extends ModuleWeightingScheme {

    @Override
    public double defineWeighting(ModulePredictions trainPredictions) {
        return 1;
    }
    
}
