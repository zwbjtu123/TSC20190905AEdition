package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.EnsembleModule;

/**
 * Non uniform weighting scheme, uses F-measure to give a weighting composed of 
 * the classifier's precision and recall *for each class*
 * 
 * @author James Large
 */
public class FScore extends ModuleWeightingScheme {

    double beta;
    
    public FScore() {
        this.beta = 1;
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    public FScore(double beta) {
        this.beta = beta;
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        double[] weights = new double[numClasses];
        for (int c = 0; c < numClasses; c++) 
            weights[c] = computeFScore(module.trainResults.confusionMatrix, c);

        return weights;
    }
    
    protected double computeFScore(double[][] confMat, int c) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        if (tp == .0)
            return .0000001; 
        //some very small non-zero value, in the extreme case that no classifiers
        //in the entire ensemble classified cases of this class correctly
        //happened once on adiac (37 classes)
        
        double fp = 0.0, fn = 0.0;
        
        for (int i = 0; i < confMat.length; i++) {
            if (i!=c) {
                fp += confMat[i][c];
                fn += confMat[c][i];
            }
        }
        
        double precision = tp / (tp+fp);
        double recall = tp / (tp+fn);
        
        return (1+beta*beta) * (precision*recall) / ((beta*beta)*precision + recall);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(" + beta + ")";
    }
    
}
