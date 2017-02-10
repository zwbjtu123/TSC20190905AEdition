/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta.timeseriesensembles;

/**
 * A little class to store information about a (unspecified) classifier's results on a (unspecified) dataset
 * Used in the ensemble classes HESCA and EnsembleFromFile to store loaded results
 * 
 * Will be expanded in the future
 * 
 * @author James Large
 */
public class ModulePredictions {
    public double[][] distsForInsts;
    public double[] preds;
    public double acc; 
    
//    public double[][] predsConfusionMatrix;
//    public double[][] distsConfusionMatrix;

    public ModulePredictions(double acc, double[] preds, double[][] distsForInsts) {
        this.preds = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;
    }
    
    
//    /**
//    * @return [actual class][predicted class]
//    * @throws Exception 
//    */
//    public double[][] buildConfusionMatrix(double[] classVals, double[] preds, int numClasses) throws Exception {
//        double[][] matrix = new double[numClasses][numClasses];
//        for (int i = 0; i < preds.length; ++i)
//            ++matrix[(int)classVals[i]][(int)preds[i]];
//        
//        return matrix;
//    }
}
