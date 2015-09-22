/*
Default weka classifiers in the time domain.
 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.*;
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
    public static void formatResults(String path, String algo){
        InFile inF=new InFile(path+algo+".csv");
        OutFile out= new OutFile(path+algo+"Formatted.csv");
        
    }

/* Takes a possibly partial list of results and format into the outfile */
    public static class Results{
        String name;
        double mean;
        double stdDev;
        double[] accs;
        public boolean equals(Object o){
            if( ((Results)o).name.equals(this.name))
                return true;
            return false;
        }
    }
    public static void resultsParser(InFile f, OutFile of) throws Exception{
        int lines= f.countLines();
        f.reopen();
        ArrayList<Results> res=new ArrayList<>();
  //      System.out.println("Lines = "+lines);

        
        for(int i=0;i<lines;i++){
            String line=f.readLine();
            String[] split=line.split(",");
            Results r=new Results();
            r.name=split[0];
            r.accs=new double[split.length-1];
//            System.out.println("length="+r.accs.length+"::::"+line);
            for(int j=0;j<r.accs.length;j++){
                try{
                r.accs[j]=Double.parseDouble(split[j+1]);
                r.mean+=r.accs[j];
                }catch(Exception e){
                    System.out.println("ERROR: "+split[j]+" giving error "+e);
                    System.exit(0);
                }
            }
            r.mean/=r.accs.length;
            r.stdDev=0;
            for(int j=0;j<r.accs.length;j++){
                r.stdDev+=(r.accs[j]-r.mean)*(r.accs[j]-r.mean);
            }
            r.stdDev/=(r.accs.length-1);
            res.add(r);
        }
        for(int i=0;i<DataSets.fileNames.length;i++){
            of.writeString(DataSets.fileNames[i]+",");
            int j=0; //Wasteful linear scan
            while(j<res.size() && !DataSets.fileNames[i].equals(res.get(j).name))
                j++;
            System.out.println("J =: "+j+" "+res.size());
            if(j<res.size()){
                Results r=res.get(j);
                of.writeLine(r.mean+","+r.stdDev+","+r.accs.length);
            }
            else
                of.writeLine(",,");
        }
    }
    
    public static void main(String[] args) throws Exception{
        Classifier c;
        String s="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Standard Classifiers\\";
        OutFile res;

        res = new OutFile(s+"C45.csv");
        c= new J48();
        System.out.println("******** J48 **********");
        threadedSingleClassifier(c,res);
/*        
        c = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(2);
        ((SMO)c).setKernel(p);
        res = new OutFile(s+"SVMQ.csv");
        threadedSingleClassifier(c,res);
 /*
        res = new OutFile(s+"BayesNet.csv");
        c= new BayesNet();
        System.out.println("******** BayesNet **********");
        threadedSingleClassifier(c,res);

        /*        
        res = new OutFile(s+"C45.csv");
        c= new J48();
        System.out.println("******** J48 **********");
        threadedSingleClassifier(c,res);
        System.out.println("******** Naive Bayes *******************");
        c= new NaiveBayes();
        res = new OutFile(s+"NB.csv");
        threadedSingleClassifier(c,res);
        
        System.out.println("**************** SVML *************8**********");
        c = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(1);
        ((SMO)c).setKernel(p);
        res = new OutFile(s+"SVML.csv");
        threadedSingleClassifier(c,res);
        System.out.println("******** Random Forest ***************************");
        c= new RandomForest();
        ((RandomForest) c).setNumTrees(500);
        res = new OutFile(s+"RandF.csv");
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
