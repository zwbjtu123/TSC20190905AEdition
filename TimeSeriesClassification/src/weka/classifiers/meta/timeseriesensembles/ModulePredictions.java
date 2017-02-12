/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta.timeseriesensembles;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import weka.core.Instances;

/**
 * A little class to store information about a (unspecified) classifier's results on a (unspecified) dataset
 * Used in the ensemble classes HESCA and EnsembleFromFile to store loaded results
 * 
 * Will be expanded in the future
 * 
 * @author James Large
 */
public class ModulePredictions {
    //directly from file
    public final double[] preds;
    public final double acc; 
    public final double[][] distsForInsts; //may be null
    public final double[][] confusionMatrix; //[actual class][predicted class]
    private final double[] classVals;
    
    private int numClasses;
    
    public ModulePredictions(double acc, double[] preds, double[][] distsForInsts) {
        this.preds = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;
        
        this.numClasses = -1;
        
        this.confusionMatrix = null;
        this.classVals = null;
    }

    private ModulePredictions(double acc, double[] classVals, double[] preds, double[][] distsForInsts, int numClasses) {        
        this.preds = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;
        
        this.numClasses = numClasses;
        
        this.classVals = classVals;
        this.confusionMatrix = buildConfusionMatrix();
    }
    
    /**
    * @return [actual class][predicted class]
    */
    private double[][] buildConfusionMatrix() {
        double[][] matrix = new double[numClasses][numClasses];
        for (int i = 0; i < preds.length; ++i)
            ++matrix[(int)classVals[i]][(int)preds[i]];
        
        return matrix;
    }
     
    private static void validateResultsFile(File file, double acc, ArrayList<Double> alclassvals, ArrayList<Double> alpreds, ArrayList<ArrayList<Double>> aldists, int numClasses) throws Exception {
        if (!aldists.isEmpty()) {
            if (alpreds.size() != aldists.size())
                throw new Exception("validateResultsFile Exception: "
                        + "somehow have different number of predictions and distForInstances: " + file.getAbsolutePath());

            for (ArrayList<Double> dist : aldists)
                if (dist.size() != numClasses)
                    throw new Exception("validateResultsFile Exception: "
                            + "instance reports different numbers of classes: " + file.getAbsolutePath());
        }
        
        double count = 0.0;
        for (int i = 0; i < alpreds.size(); ++i)
            if (alpreds.get(i).equals(alclassvals.get(i)))
                count++;
        
        double a = count/alpreds.size();
        if (a != acc)
            throw new Exception("validateResultsFile Exception: "
                    + "incorrect accuracy (" + acc + "reported vs" +a +"actual) reported in: " + file.getAbsolutePath());
    }
    
    public static ModulePredictions loadResultsFile(File file, int numClasses) throws Exception {    
                
        ArrayList<Double> alpreds = new ArrayList<>();
        ArrayList<Double> alclassvals = new ArrayList<>();
        ArrayList<ArrayList<Double>> aldists = new ArrayList<>();
        
        Scanner scan = new Scanner(file);
        scan.useDelimiter("\n");
        scan.next();
        scan.next();
        double acc = Double.parseDouble(scan.next().trim());
        
        String [] lineParts = null;
        while(scan.hasNext()){
            lineParts = scan.next().split(",");

            if (lineParts == null || lineParts.length < 2)
                continue;
            
            alclassvals.add(Double.parseDouble(lineParts[0].trim()));
            alpreds.add(Double.parseDouble(lineParts[1].trim()));
            
            if (lineParts.length > 3) {//dist for inst is present
                ArrayList<Double> dist = new ArrayList<>();
                for (int i = 3; i < lineParts.length; ++i)  //act,pred,[empty],firstclassprob.... therefore 3 start
                    if (lineParts[i] != null && !lineParts[i].equals("")) //may have an extra comma on the end...
                        dist.add(Double.parseDouble(lineParts[i].trim()));
                aldists.add(dist);
            }
        }
        
        scan.close();
        
        validateResultsFile(file, acc, alclassvals, alpreds, aldists, numClasses);
        
        double [] classVals = new double[alclassvals.size()];
        for (int i = 0; i < alclassvals.size(); ++i)
            classVals[i]= alclassvals.get(i);
        
        double [] preds = new double[alpreds.size()];
        for (int i = 0; i < alpreds.size(); ++i)
            preds[i]= alpreds.get(i);
        
        double[][] distsForInsts = null;
        if (!aldists.isEmpty()) {
            distsForInsts = new double[aldists.size()][aldists.get(0).size()];
            for (int i = 0; i < aldists.size(); ++i)
                for (int j = 0; j < aldists.get(i).size(); ++j) 
                    distsForInsts[i][j] = aldists.get(i).get(j);
        }
        
        return new ModulePredictions(acc, classVals, preds, distsForInsts, numClasses);
    }
}
