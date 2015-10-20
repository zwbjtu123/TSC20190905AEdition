/*

 */
package bakeOffExperiments;

import tsc_algorithms.*;
import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.*;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class Experiments extends Thread{
   Instances train;
   Instances test;
//For threaded version only
   String problem;
   String classifier;
    double testAccuracy;
    String name;
    int resamples=100;
    int fold=0;
    String path;
    String preds;
    double acc;
    public static boolean removeUseless=false;
    public Experiments(Instances tr, Instances te, String cls, String prob, String predictions,int res, int f){
        train=tr;
        test=te;
        classifier=cls;
        problem=prob;
        preds=predictions;
        resamples=res;
        fold=f;
    }
    public double getAcc(){return acc;}
   @Override
    public void run(){
        Classifier c=setClassifier(classifier);
        if(resamples==1){
            //Open up fold file to write to
            File f=new File(preds+"/fold"+fold+".csv");
            if(!f.exists() || f.length()==0){
                acc=singleSampleExperiment(train,test,c,fold,preds);
                System.out.println("Fold "+fold+" acc ="+acc);
            }
            else
                System.out.println("Fold "+fold+" already complete");
        }
        else{
            OutFile of=new OutFile(DataSets.resultsPath+classifier+"/"+problem+".csv");
            of.writeString(problem+",");
            double[] folds=resampleExperiment(train,test,c,100,of,preds);
            of.writeString("\n");
        }
    }
   
//All classifier names  
    //<editor-fold defaultstate="collapsed" desc="Directory names for all classifiers">   
    static String[] standard={"NB","C45","SVML","SVMQ","Logistic","BayesNet","RandF","RotF","MLP"};
    static String[] elastic = {"Euclidean_1NN","DTW_R1_1NN","DTW_Rn_1NN","DDTW_R1_1NN","DDTW_Rn_1NN","ERP_1NN","LCSS_1NN","MSM_1NN","TWE_1NN","WDDTW_1NN","WDTW_1NN","DD_DTW","DTD_C","DTW_Features"};
    static String[] shapelet={"ST","LS","FS"};
    static String[] dictionary={"BoP","SAXVSM","BOSS"};
    static String[] interval={"TSF","TSBF","LPS"};
    static String[] ensemble={"ACF","PS","EE","COTE"};
    static String[] complexity={"CID_ED","CID_DTW","RPCD"};
    static String[][] classifiers={standard,elastic,shapelet,dictionary,interval,ensemble,complexity};
    static final String[] directoryNames={"standard","elastic","shapelet","dictionary","interval","ensemble","complexity"};
      //</editor-fold> 
    
  //  public static OutFile out;
    
    //<editor-fold defaultstate="collapsed" desc="unfinished problems">   
    static boolean doUnfinished=false;
    static String[] unfinishedSVMQ={"FordA","FordB"};
    static String[] unfinishedRotF={"StarlightCurves"};
    static String[] unfinishedLogistic={"NonInvasiveFatalECGThorax1","NonInvasiveFatalECGThorax2","Phoneme","ShapesAll","StarlightCurves","UWaveGestureLibraryZ"};
    static String[] unfinishedMLP={"HandOutlines","StarlightCurves"};

    static String[] unfinishedCID_DTW={"FordA","FordB","HandOutlines","NonInvasiveFatalECGThorax1","NonInvasiveFatalECGThorax2","Phoneme","StarlightCurves","UWaveGestureLibraryX","UWaveGestureLibraryY","UWaveGestureLibraryZ","UWaveGestureLibraryAll"};
    static int[] doneCIDCount={11,18,5,10,10,20,0,58,68,6,5};
static String[] unfinishedTSF={"CinCECGtorso","ElectricDevices","FordA","FordB","HandOutlines","Mallat","NonInvasiveFatalECGThorax1","NonInvasiveFatalECGThorax2","Phoneme","ShapesAll","StarlightCurves","UWaveGestureLibraryX","UWaveGestureLibraryY","UWaveGestureLibraryZ","UWaveGestureLibraryAll"};
    static int[] doneTSFCount={88,17,15,16,15,85,13,13,63,53,13,74,75,75,22};
    
    static String[] unfinishedDD_DTW={"StarlightCurves", "UWaveGestureLibraryZ"};
    static String[] unfinishedDTD_C={"ElectricDevices","FordA","FordB","HandOutlines","InlineSkate","NonInvasiveFatalECGThorax1","NonInvasiveFatalECGThorax2","Phoneme","StarlightCurves", "UWaveGestureLibraryY","UWaveGestureLibraryZ","UWaveGestureLibraryAll"};
      //</editor-fold> 
    
    //Global file to write to 
    static OutFile out;
    
    public static Classifier setClassifier(String classifier){
        Classifier c=null;
        switch(classifier){
            case "SVMQ":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(2);
                ((SMO)c).setKernel(p);

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
            case "CID_ED":
                c=new NN_CID();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "LearnShapelets": case "LS":
                c=new LearnShapelets();
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
            case "DD_DTW":
                c=new NNDerivativeWeighting();
                break;
            case "DTD_C":
                c=new NNTransformWeighting();
                break;
            case "TSF":
                c=new TimeSeriesForest();
                break;
            case "ACF":
                c=new ACF_Ensemble();
                ((ACF_Ensemble)c).setClassifierType("WE");
                break;
            case "PS":
                c=new PS_Ensemble();
                ((PS_Ensemble)c).setClassifierType("WE");
                break;
            case "TSBF":
                c=new TSBF();
                break;
//            case "FS":
//                c=new FastShapelet();
            default:
                System.out.println("UNKNOWN CLASSIFIER");
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
    
    
 //Do all the reps for one problem   
    public static void threadedSingleClassifierSingleProblem(String classifier, String problem,int reps, int start) throws Exception{
        
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        f=new File(DataSets.resultsPath+classifier+"/"+problem+".csv");
        if(!f.exists()){
            out=new OutFile(DataSets.resultsPath+classifier+"/"+problem+".csv");
            out.writeString(problem+",");
        }
        else{
            out=new OutFile(DataSets.resultsPath+classifier+"/"+problem+"2.csv");
            out.writeString(problem+",");            
        }
        Experiments[] thr=new Experiments[reps-start];
        for (int i = start; i < reps; i++) {
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train,test, i);
            thr[i-start]=new Experiments(data[0],data[1],classifier,problem,predictions,1,i);
        }
//Do in batches        
        int processors= Runtime.getRuntime().availableProcessors();
        int count=0;
        Thread[] current=new Thread[processors];
        while(count<thr.length){
            if(thr.length-count<processors){
                processors=thr.length-count;
            }
            for(int i=0;i<processors;i++){
                System.out.println("\t starting repetition "+(start+count));
                current[i]=thr[count++];
                current[i].start();
            }
            for(int i=0;i<processors;i++)
                current[i].join();        
        System.out.println(" Finished the first "+(start+count)+" batches");
        }
        double[] accs=new double[reps-start];
        for (int i = 0; i < accs.length; i++) {
            accs[i]=thr[i].getAcc();
            out.writeString(accs[i]+",");
        }
    }
    
  
    public static void threadedSingleClassifierSingleProblem(Classifier c, String results,String problem, int reps) throws Exception{
         ThreadedClassifierExperiment[] thr=new ThreadedClassifierExperiment[reps]; 
//Load train test
         int count=0;
         while(count<reps){
            for(int j=0;j<8;j++){
//                OutFile out=new OutFile(results+"fold"+count+".csv");
                Classifier cls=AbstractClassifier.makeCopy(c);
                Instances train=ClassifierTools.loadData(problem+"_TRAIN");
                Instances test=ClassifierTools.loadData(problem+"_TEST");
//Check results directory exists                
                thr[count]=new ThreadedClassifierExperiment(train,test,cls,problem,results);
                thr[count].resamples=1;
                thr[count].start();
                System.out.println(" started rep="+count);
                count++;
            }
            for(int j=0;j<8;j++){
                 thr[count-j-1].join();
            }
            System.out.println(" finished batch="+count);

       }
         
    }
   
    
 
    
    public static void singleClassifier(String classifier,String problemName) throws Exception{
//
        int position=1;
        while(position<=DataSets.fileNames.length && !DataSets.fileNames[position-1].equals(problemName))
            position++;
        if(position<DataSets.fileNames.length){
            String[] args={classifier,position+""};
            singleClassifier(args);
        }
        else{
            System.out.println("Invalid problem name ="+problemName);
            System.exit(0);
        }
        
    } 
    public static void singleClassifier(String[] args) throws Exception{
//first gives the problem file  
        String classifier=args[0];
        String s=DataSets.fileNames[Integer.parseInt(args[1])-1];        
//        String s=unfinished[Integer.parseInt(args[1])-1];
        Classifier c=setClassifier(classifier);
        System.out.println("Classifier ="+classifier+" problem ="+s);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+s;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        
        OutFile of=new OutFile(DataSets.resultsPath+classifier+"/"+s+".csv");
        of.writeString(s+",");
        double[] folds;
        if(s.equals("ACF")){
            train=ACF.formChangeCombo(train);
            test=ACF.formChangeCombo(test);
            ((ACF_Ensemble) c).doACFTransform(false);
        }else if(s.equals("PS")){
            PowerSpectrum ps=((PS_Ensemble) c).getTransform();
            train=ps.process(train);
            test=ps.process(test);
            ((PS_Ensemble) c).doTransform(false);
        }
        folds=resampleExperiment(train,test,c,100,of,predictions);
        of.writeString("\n");
    }
    
    
    public static void singleClassifierAndFold(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String s=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+s;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/fold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(s+","); );
            double acc =singleSampleExperiment(train,test,c,fold,predictions);
 //       of.writeString("\n");
        }
    }
    public static void singleClassifierAndFold(String classifier,String problemName, int fold) throws Exception{
        int position=1;
        while(position<=DataSets.fileNames.length && !DataSets.fileNames[position-1].equals(problemName))
            position++;
        if(position<DataSets.fileNames.length){
            String[] args={classifier,position+"",(fold+1)+""};
            singleClassifier(args);
        }
        else{
            System.out.println("Invalid problem name ="+problemName);
            System.exit(0);
        }
        
    }    

    
//Static methds for experiments    
    public static double[] resampleExperiment(Instances train, Instances test, Classifier c, int resamples,OutFile of,String preds){

       double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
            File f=new File(preds+"/fold"+i+".csv");
            if(!f.exists() || f.length()==0){
            foldAcc[i]=singleSampleExperiment(train,test,c,i,preds);
                of.writeString(foldAcc[i]+",");
            }
            else
                of.writeString(",");
        }            
         return foldAcc;
    }


    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, sample);
        double acc=0;
        double act,pred;
        OutFile p=new OutFile(preds+"/fold"+sample+".csv");
// hack here to save internal CV for furhter ensembling         
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(data[0]);
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=data[1].instance(j).classValue();
                pred=c.classifyInstance(data[1].instance(j));
                if(act==pred)
                    acc++;
                p.writeLine(act+","+pred);
            }
            acc/=data[1].numInstances();
//            of.writeString(foldAcc[i]+",");

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }

        
    public static void clusterRun(String[] args) throws Exception{
        if(args.length>2)   //
            singleClassifierAndFold(args);
        else
            singleClassifier(args);
            
    }
    public static void main(String[] args){
        try{
            if(args.length>0){ //Cluster run

            DataSets.resultsPath=DataSets.clusterPath+"Results/";
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            clusterRun(args);

         }
         else{       
//Local threaded run    
             DataSets.resultsPath="C:/Users/ajb/Dropbox/Big TSC Bake Off/New Results/";
             DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            String classifier="DTD_C";
             String problem;
// TSF    FordA 15, FordB 16, HandOutlines 15, Mallat 85, NonInvasiveFatalECGThorax1 13, NonInvasiveFatalECGThorax2 13,
//     Phoneme 63, ShapesAll 53, StarlightCurves13,  UWaveGestureLibraryX 74, UWaveGestureLibraryY 75
//     UWaveGestureLibraryZ 75, UWaveGestureLibraryAll22
//             String problem="UWaveGestureLibraryY"; 68
             problem="InsectWingbeatSound";

             DataSets.resultsPath+=getFolder(classifier)+"/";
             System.out.println("Running "+classifier+" on "+problem);
  
 //            singleClassifier(classifier,problem);
             threadedSingleClassifierSingleProblem(classifier,problem,100,12);
             System.out.println("Finished");
         }
        }catch(Exception e){
            System.out.println("Exception thrown ="+e);
            System.exit(0);
        }
        
       
        
    }
    public static String getFolder(String classifier){
        for(int i=0;i<classifiers.length;i++)
            for(int j=0;j<classifiers[i].length;j++)
                if(classifiers[i][j].equals(classifier))
                    return directoryNames[i];
        return null;
    }




}
