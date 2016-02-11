/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development.Aaron;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.BalancedClassShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory.calculateNumberOfShapelets;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory.calculateOperations;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.classValue.NormalClassValue;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.SubSeqDistance;

/**
 *
 * @author Aaron
 */
public class SpeedAndAccuracyExperiments {
    
    //this is 10^13 this should do all datasets smallest than excluding PhalangesOutlinesCorrect.
    //should be 41.
    public static final long opCountThreshold = 10000000000000l; 
    
    protected static enum Parameters{
        standard, prune, pruneRobin, pruneRobinOnline, pruneRobinImpOnline, pruneRobinImpOnlineBinary, all
    }
    
    
    public static void main(String[] args) throws IOException{

        String dataset = development.DataSets.fileNames[Integer.parseInt(args[0])-1];
        int value = Integer.parseInt(args[1]);
        
        String windowsDir = "";//Dropbox/";
        
        String dirPath = "../../" + windowsDir + "TSC Problems (1)/";
        File dir  = new File(dirPath);
        
        String path = dir.getPath() + File.separator + dataset + File.separator + dataset;
        Instances train = ClassifierTools.loadData(path+ "_TRAIN.arff");
        Instances test = ClassifierTools.loadData(path + "_TEST.arff");

        long ops = calculateOperations(train, 3, train.numAttributes()-1);

        System.out.printf("%s,%d\n",dataset, ops);

        if(ops > opCountThreshold)  return; //too big 
        
        File logFile = new File("Experiments" + File.separator + dataset + File.separator + Parameters.values()[value] + ".csv");
        if(!logFile.exists()){
            logFile.getParentFile().mkdirs();
            logFile.createNewFile();
        }
        
        experiments(logFile, train, test, value);
    }
    
    
    public static void experiments(File logFile, Instances train, Instances test, int value){
        try {
            FullShapeletTransform fst = constructTransform(Parameters.values()[value], train);
            
            Instances processedTrain    = fst.process(train);
            
            String write = Parameters.values()[value] + "," + fst.getCount();
            Instances processedTest     =  fst.process(test);
            
            WeightedEnsemble we = new WeightedEnsemble();
            we.setWeightType("prop");
            we.buildClassifier(processedTrain);
            
            double accuracy = utilities.ClassifierTools.accuracy(processedTest, we);
            write += "," + accuracy;
            System.out.print(write);
            
            try (PrintWriter pw = new PrintWriter(logFile)) {
                pw.print(write);
            }
                
        } catch (Exception ex) {
            System.out.println("Exception " + ex);
        }
    }
    
    public static FullShapeletTransform constructTransform(Parameters value, Instances train){
        switch(value)  {    
            case standard:
                return createTransform(train, new FullShapeletTransform(), false, false, new NormalClassValue(), new SubSeqDistance());
            case prune: 
                return createTransform(train, new FullShapeletTransform(), true, false, new NormalClassValue(), new SubSeqDistance());
            case pruneRobin:
                return createTransform(train, new FullShapeletTransform(), true, true, new NormalClassValue(), new SubSeqDistance());
            case pruneRobinOnline:
                return createTransform(train, new FullShapeletTransform(), true, true, new NormalClassValue(), new OnlineSubSeqDistance());
            case pruneRobinImpOnline:
                return createTransform(train, new FullShapeletTransform(), true, true, new NormalClassValue(), new ImprovedOnlineSubSeqDistance());
            case pruneRobinImpOnlineBinary:
                return createTransform(train, new FullShapeletTransform(), true, true, new BinarisedClassValue(), new ImprovedOnlineSubSeqDistance());
            default: case all:
                return createTransform(train, new BalancedClassShapeletTransform(), true, true, new BinarisedClassValue(), new ImprovedOnlineSubSeqDistance());
        }
    }
    
    
    public static FullShapeletTransform createTransform(Instances train, FullShapeletTransform fst, boolean prune, boolean robin, NormalClassValue ncv, SubSeqDistance ssq){
        fst.setNumberOfShapelets(train.numInstances() * 10);
        fst.setShapeletMinAndMax(3, train.numAttributes()-1);
        fst.setCandidatePruning(prune);
        fst.setRoundRobin(robin);
        fst.setClassValue(ncv);
        fst.setSubSeqDistance(ssq);
        fst.supressOutput();
        return fst;
    }
    
}
