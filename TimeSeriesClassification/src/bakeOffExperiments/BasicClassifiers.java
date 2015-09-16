/*
Default weka classifiers in the time domain.
 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.OutFile;
import java.io.File;
import java.util.*;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
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
    
    public static void threadedSingleClassifier(Classifier c, OutFile results) throws Exception{
         ThreadedClassifierExperiment[] thr=new ThreadedClassifierExperiment[DataSets.fileNames.length]; 
         out=results;
         String str=DataSets.dropboxPath;
         
         List<String> notNormed=Arrays.asList(DataSets.notNormalised);
         NormalizeCase nc = new NormalizeCase();
//         String[] files=DataSets.fileNames;
         String[] files=DataSets.notNormalised;
         for(int i=0;i<files.length;i++){
//Load train test
            String s=files[i];
            Classifier cls=AbstractClassifier.makeCopy(c);
            Instances train=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TEST");
            if(notNormed.contains(s)){   //Need to normalise
                train=nc.process(train);
                test=nc.process(test);
            }
           
            thr[i]=new ThreadedClassifierExperiment(train,test,cls,files[i]);
            thr[i].start();
            System.out.println(" started ="+s);
         }   
         for(int i=0;i<files.length;i++)
             thr[i].join();
         
    }
    public static void clusterSingleClassifier(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String s=DataSets.fileNames[Integer.parseInt(args[1])];
        Classifier c=null;
        switch(classifier){
            case "NaiveBayes":
                c=new NaiveBayes();
                break;
            case "MultilayerPerceptron":
                c=new MultilayerPerceptron();
                break;
            case "RotationForest50":
                c= new RotationForest();
                ((RotationForest)c).setNumIterations(50);
                break;
            default:
                c= new kNN(1);
        }
        String str=DataSets.clusterPath;
        System.out.println("Classifier ="+str+" problem ="+s);
        File f=new File(str+"Results/"+classifier);
        Instances train=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TEST");
        if(!f.exists())
            f.mkdir();
        OutFile of=new OutFile(str+"Results/"+classifier+"/"+s+".csv");
        of.writeString(s+",");
        double[] folds=ClusterClassifierExperiment.resampleExperiment(train,test,c,100,of);
        of.writeString("\n");
    }
    public static void main(String[] args) throws Exception{
        System.out.println("******** J48 **********");
        Classifier c;
        c= new J48();
        OutFile res = new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Working docs\\C45Norm.csv");
        threadedSingleClassifier(c,res);
        System.out.println("******** Naive Bayes *******************");
        c= new NaiveBayes();
        res = new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Working docs\\NBNorm.csv");
        threadedSingleClassifier(c,res);
        System.out.println("**************** SVML *************8**********");
        c = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(1);
        ((SMO)c).setKernel(p);
        res = new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Working docs\\SVMLNorm.csv");
        threadedSingleClassifier(c,res);
        System.out.println("******** Random Forest ***************************");
        c= new RandomForest();
        ((RandomForest) c).setNumTrees(500);
        res = new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Working docs\\RandFNorm.csv");
        threadedSingleClassifier(c,res);
        c= new BayesNet();
        res = new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Working docs\\BayesNetNorm.csv");
        threadedSingleClassifier(c,res);
//        RotationForest c= new RotationForest();
//        c.setNumIterations(50);
//        SMO c = new SMO();
 //       PolyKernel p=new PolyKernel();
 //       p.setExponent(1);
 //       c.setKernel(p);


//        clusterSingleClassifier(args);
//        System.exit(0);
        
        
    }





}
