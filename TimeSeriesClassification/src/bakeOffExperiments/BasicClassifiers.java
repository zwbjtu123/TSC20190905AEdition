/*
Default weka classifiers in the time domain.
 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.OutFile;
import java.util.*;
import utilities.ClassifierTools;
import utilities.ThreadedClassifierExperiment;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class BasicClassifiers {
    public static OutFile out;
    
    public static void singleClassifier(Classifier c, OutFile results) throws Exception{
         ThreadedClassifierExperiment[] thr=new ThreadedClassifierExperiment[DataSets.fileNames.length]; 
         out=results;
         String str="C:/Users/ajb/Dropbox/";
         List<String> notNormed=Arrays.asList(DataSets.notNormalised);
         NormalizeCase nc = new NormalizeCase();
         for(int i=0;i<DataSets.fileNames.length;i++){
//Load train test
            String s=DataSets.fileNames[i];
            Classifier cls=AbstractClassifier.makeCopy(c);
            Instances train=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TEST");
            if(notNormed.contains(s)){   //Need to normalise
                train=nc.process(train);
                test=nc.process(test);
            }
            
            thr[i]=new ThreadedClassifierExperiment(train,test,cls,DataSets.fileNames[i]);
            thr[i].start();
            System.out.println(" started ="+s);
         }   
         for(int i=0;i<DataSets.fileNames.length;i++)
             thr[i].join();
         
    }
    public static void main(String[] args) throws Exception{
        RandomForest randF= new RandomForest();
        randF.setNumTrees(500);
        OutFile res = new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Working docs\\RandF500.csv");
        singleClassifier(randF,res);
        
        
    }





}
