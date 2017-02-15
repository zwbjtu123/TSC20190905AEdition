/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta.timeseriesensembles;

/**
 * Simple container class for the results of a classifier (module) on a dataset
 * 
 * @author James Large
 */
public class ModuleResults {
    public final double[] predClassVals;
    public final double acc; 
    public final double[][] distsForInsts; //may be null
    public final double[][] confusionMatrix; //[actual class][predicted class]
    public final double[] trueClassVals;
    public final double variance;

    private int numClasses;

    public ModuleResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, int numClasses) {        
        this.predClassVals = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;

        this.numClasses = numClasses;

        this.trueClassVals = classVals;
        this.confusionMatrix = buildConfusionMatrix();
        
        this.variance = -1; //not defined 
    }
    
    public ModuleResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, double variance, int numClasses) {        
        this.predClassVals = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;

        this.numClasses = numClasses;

        this.trueClassVals = classVals;
        this.confusionMatrix = buildConfusionMatrix();
        
        this.variance = variance; 
    }

    /**
    * @return [actual class][predicted class]
    */
    private double[][] buildConfusionMatrix() {
        double[][] matrix = new double[numClasses][numClasses];
        for (int i = 0; i < predClassVals.length; ++i)
            ++matrix[(int)trueClassVals[i]][(int)predClassVals[i]];

        return matrix;
    }
}
