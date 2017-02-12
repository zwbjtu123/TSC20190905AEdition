package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;

/**
 * Base class for defining the weighting of a classifiers votes in ensemble classifiers
 * 
 * @author James Large
 */
public abstract class ModuleWeightingScheme {
    
    boolean uniformWeighting;
    
    public abstract double[] defineWeighting(ModulePredictions trainPredictions, int numClasses);
    
    protected double[] makeUniformWeighting(double weight, int numClasses) {
        double[] weights = new double[numClasses];
        for (int i = 0; i < weights.length; ++i)
            weights[i] = weight;
        return weights;
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
    
}
