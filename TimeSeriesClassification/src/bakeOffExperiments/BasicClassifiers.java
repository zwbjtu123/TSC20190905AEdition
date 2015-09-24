/*
Default weka classifiers in the time domain.
 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.*;
import other_peoples_algorithms.NN_CID;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
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
    static String[] unfinished={"Adiac","FiftyWords","Fish","Mallat","SonyAIBORobotSurface1","StarlightCurves","UWaveGestureLibraryZ","Wafer","Yoga"};
    
    public static void threadedSingleClassifier(Classifier c, OutFile results) throws Exception{
         ThreadedClassifierExperiment[] thr=new ThreadedClassifierExperiment[DataSets.fileNames.length]; 
         out=results;
         
         String str=DataSets.dropboxPath;
         
         String[] files=DataSets.fileNames;
         for(int i=0;i<files.length;i++){
//Load train test
            String s=files[i];
            Classifier cls=AbstractClassifier.makeCopy(c);
            Instances train=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TEST");
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
        String s=DataSets.fileNames[Integer.parseInt(args[1])-1];        
//        String s=unfinished[Integer.parseInt(args[1])-1];
        Classifier c=null;
        switch(classifier){
            case "SVMQ":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(2);
                ((SMO)c).setKernel(p);

                break;
            case "SVML":
                c=new SMO();
                PolyKernel p2=new PolyKernel();
                p2.setExponent(1);
                ((SMO)c).setKernel(p2);
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "RotF":
                c= new RotationForest();
                ((RotationForest)c).setNumIterations(50);
                break;
            case "Logistic":
                c= new Logistic();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "DTW":
                c=new DTW_1NN();
                ((DTW_1NN)c).setR(1.0);
                ((DTW_1NN)c).optimiseWindow(false);
                break;
            case "DTWCV":
                c=new DTW_1NN();
                ((DTW_1NN)c).optimiseWindow(true);
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
    public static void formatResults(String path, String algo){
        InFile inF=new InFile(path+algo+".csv");
        OutFile out= new OutFile(path+algo+"Formatted.csv");
        
    }


    
    public static void main(String[] args) throws Exception{
        clusterSingleClassifier(args);
        System.exit(0);
        Classifier c;
        String s="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\standard\\";
        OutFile res;
       
        System.out.println("**************** SVML *************8**********");
        c = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(1);
        ((SMO)c).setKernel(p);
        res = new OutFile(s+"SVML.csv");
        threadedSingleClassifier(c,res);

        /*        
        System.out.println("******** Random Forest ***************************");
        c= new RandomForest();
        ((RandomForest) c).setNumTrees(500);
        res = new OutFile(s+"RandF.csv");
        threadedSingleClassifier(c,res);

        res = new OutFile(s+"C45.csv");
        c= new J48();
        System.out.println("******** J48 **********");
        threadedSingleClassifier(c,res);
        res = new OutFile(s+"BayesNet.csv");
        c= new BayesNet();
        System.out.println("******** BayesNet **********");
        threadedSingleClassifier(c,res);

        /*        
        c = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(2);
        ((SMO)c).setKernel(p);
        res = new OutFile(s+"SVMQ.csv");
        threadedSingleClassifier(c,res);
 /*

        /*        
        res = new OutFile(s+"C45.csv");
        c= new J48();
        System.out.println("******** J48 **********");
        threadedSingleClassifier(c,res);
        c= new RotationForest();
        ((RotationForest)c).setNumIterations(50);
        res = new OutFile(s+"RotF.csv");
        threadedSingleClassifier(c,res);
        c = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(2);
        ((SMO)c).setKernel(p);
        res = new OutFile(s+"SVMQ.csv");
        threadedSingleClassifier(c,res);
        c= new MultilayerPerceptron();
        res = new OutFile(s+"MLP.csv");
        threadedSingleClassifier(c,res);
        c= new Logistic();
        res = new OutFile(s+"Logistic.csv");
        threadedSingleClassifier(c,res);
        */
//        clusterSingleClassifier(args);
//        System.exit(0);
        
        
    }





}
