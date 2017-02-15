
package weka.classifiers.meta.timeseriesensembles.voting;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;
import weka.core.Instance;

/**
 * Base class for methods on combining ensemble members' ouputs into a single classification/distribution
 * 
 * @author James Large
 */
public abstract class ModuleVotingScheme {
    
    protected int numClasses;
    protected boolean requiresDistsForInstances;
    
    public boolean getRequiresDistsForInstances() { return requiresDistsForInstances; }
    
    public abstract void trainVotingScheme(EnsembleModule[] modules, int numClasses);
    
    public abstract double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex);
    
    public double classifyTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] dist = distributionForTrainInstance(modules, trainInstanceIndex);
        return indexOfMax(dist);
    }
    
    public abstract double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex);
    
    public double classifyTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] dist = distributionForTestInstance(modules, testInstanceIndex);
        return indexOfMax(dist);
    }
    
    public abstract double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception;
    
    public double classifyInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] dist = distributionForInstance(modules, testInstance);
        return indexOfMax(dist);
    }
    
    protected double indexOfMax(double[] dist) {
        double max = dist[0];
        double maxInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        return maxInd;
    }
    
    /**
     * makes array sum to 1
     */
    protected double[] normalise(double[] dist) {
        //normalise so all sum to one 
        double sum=dist[0];
        for(int i=1;i<dist.length;i++)
            sum+=dist[i];
        for(int i=0;i<dist.length;i++)
            dist[i]/=sum;
        return dist;
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
