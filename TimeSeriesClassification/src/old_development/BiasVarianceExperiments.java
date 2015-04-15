/*
 Run experiments to perform a bias variance decomposition 
 */

package old_development;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
* The standard test/train splits are maintained. 
* 
* 
* 200 separate training bootstrap samples
of size 200 or half the training set size (whichever is smaller) 
* were taken by uniformly sampling with replacement from the training
set. We then compute the main prediction, bias and both the unbiased and biased
variance, and net-variance (as defined in Section 2.3) over the 200 test sets.
* 
 * @author ajb
 */
public class BiasVarianceExperiments {
    Classifier c;   //The classifier we are measuring the bias/variance for
    public ExperimentStats singleExperiment(Instances all, int trainSize){
//Perform random split        
        ExperimentStats results=new ExperimentStats(trainSize);
        return results;
    } 
    
    
    public static class ExperimentStats{
        int[] predictions;
        int[] actuals;
        int[] indexes;
        public ExperimentStats(int size){
            predictions = new int[size];
            actuals = new int[size];
            indexes = new int[size];
            
        }
    }
}
