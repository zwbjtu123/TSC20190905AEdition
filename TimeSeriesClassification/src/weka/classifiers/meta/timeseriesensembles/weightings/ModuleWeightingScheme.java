package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;

/**
 * Base class for defining the weighting of a classifiers votes in ensemble classifiers
 * 
 * @author James Large
 */
public abstract class ModuleWeightingScheme {
    
    public abstract double defineWeighting(ModulePredictions trainPredictions);
    
}
