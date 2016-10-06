/*
Class to run one of various simulations.  
*/
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import statistics.simulators.DataSimulator;
import statistics.simulators.Model;
import statistics.simulators.SimulateAR;
import statistics.simulators.SimulateDictionaryData;
import statistics.simulators.SimulateShapeletData;
import statistics.simulators.SimulateWholeSeriesData;
import tsc_algorithms.*;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import weka.classifiers.trees.EnhancedRandomForest;
import weka.core.Instances;
import tsc_algorithms.*;
import utilities.ClassifierTools;
/**
 *
 * @author ajb
 */
public class SimulationExperiment {
    static boolean local=false;
    static int []casesPerClass={50,50};
    static int seriesLength=500;
    static String[] allClassifiers={ //Benchmarks
        "RotF","DTW",
        //Whole series
        "DD_DTW","DTD_C","EE","HESCA",
        //Interval
        "TSF","TSBF","LPS",
        //Shapelet
        "FastShapelets","ST","LearnShapelets",
        //Dictionary
        "BOP","BOSS",
        //Spectral
        "RISE",
        //Combos
        "COTE"};
    static String[] allSimulators={"WholeSeries","Interval","Shapelet","Dictionary"};
    
    public static void createAllScripts(){
//Generates cluster scripts for all combos of classifier and simulator     
       String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\SimulatorScripts\\";
        for(String a:allSimulators){
            OutFile of2=new OutFile(path+a+".txt");
            for(String s:allClassifiers){
                OutFile of = new OutFile(path+s+a+".bsub");
                of.writeLine("#!/bin/csh");
                of.writeLine("#BSUB -q long-eth");
                of.writeLine("#BSUB -J "+s+"[1-200]");
                of.writeLine("#BSUB -oo output/"+a+".out");
                of.writeLine("#BSUB -eo error/"+a+".err");
                of.writeLine("#BSUB -R \"rusage[mem=6000]\"");
                of.writeLine("#BSUB -M 8000");
                of.writeLine("module add java/jdk1.8.0_51");
                of.writeLine("java -jar Simulator.jar "+a+" "+ s+" $LSB_JOBINDEX");                
                
                of2.writeLine("bsub < Scripts/SimulatorExperiments/"+s+a+".bsub");
            }   
        }
    } 
    
    public static Classifier createClassifier(String str) throws RuntimeException{
        Classifier c;
        switch(str){
            case "RandF":
                c=new EnhancedRandomForest();
                break;
            case "RotF":
                c=new RotationForest();
                break;
            case "DTW":
                c=new DTW_1NN();
                break;
             case "DD_DTW":
                c=new DD_DTW();
                break;               
            case "DTD_C":    
                c=new DTD_C();
                break;               
            case "EE":    
                c=new ElasticEnsemble();
                break;                          
            case "TSF":
                c=new TSF();
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "LPS":
                c=new LPS();
                break;
            case "FastShapelets":
                c=new FastShapelets();
                break;
            case "ST":
                c=new ST_Ensemble();
                if(local)
                    ((ST_Ensemble)c).setOneMinuteLimit();
                else
                    ((ST_Ensemble)c).setOneDayLimit();
                break;
            case "LearnShapelets":
                c=new LearnShapelets();
                ((LearnShapelets)c).setParamSearch(false);
                break;
            case "BOP":
                c=new BagOfPatterns();
                break;
            case "BOSS":
                c=new BOSSEnsemble();
                break;
            case "COTE":
                c=new COTE();
                break;
            case "RISE":
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                ((RISE)c).setNosBaseClassifiers(500);
                break;
            case "RISE_HESCA":
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                Classifier base=new HESCA();
                ((RISE)c).setBaseClassifier(base);
                ((RISE)c).setNosBaseClassifiers(20);
                break;
            default:
                throw new RuntimeException(" UNKNOWN CLASSIFIER "+str);
        }
        return c;
    }
    public static Instances simulateData(String str,int seed) throws RuntimeException{
        Instances data;
//        for(int:)
        Model.setGlobalRandomSeed(seed);
        switch(str){
            case "ARMA": case "AR":
                 data=SimulateAR.generateARDataSet(seriesLength, casesPerClass, true);
                break;
            case "Shapelet": 
                if(local){
                    Model.setDefaultSigma(1);
                    seriesLength=300;
                    casesPerClass=new int[]{30,30};
                }
                else{
                    Model.setDefaultSigma(1);
                    seriesLength=300;
                    casesPerClass=new int[]{50,50};
                }
                data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
                break;
            case "Dictionary":
                if(local){
                    Model.setDefaultSigma(0.1);
                    seriesLength=500;
                    casesPerClass=new int[]{50,50};
                }
                else{
                    Model.setDefaultSigma(0.1);
                    seriesLength=500;
                    casesPerClass=new int[]{50,50};
                }
                data=SimulateDictionaryData.generateDictionaryData(seriesLength,casesPerClass);
               break; 
            case "WholeSeries":
 //               data=SimulateWholeSeriesData.generateWholeSeriesData(seriesLength,casesPerClass);
//                break;
           case "WholeSeriesElastic":
 //               data=SimulateWholeSeriesData.generateWholeSeriesData(seriesLength,casesPerClass);
//                break;
        default:
                throw new RuntimeException(" UNKNOWN SIMULATOR ");
            
        }
        return data;
    }
    

//arg[0]: simulator
//arg[1]: classifier
//arg[2]: fold number    
    public static double runSimulationExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
        Classifier c=createClassifier(classifier);
        int fold=Integer.parseInt(args[2])-1;
        Instances data=simulateData(args[0],fold);
        Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
//Do the experiment: find train preds through cross validation
//Then generate all test predictions            
            double acc=singleSampleExperiment(split[0],split[1],c,fold,predictions);
            System.out.println("simulator ="+simulator+" Classifier ="+classifier+" Fold "+fold+" Acc ="+acc);
            return acc;
        }
 //       of.writeString("\n");
        return -1;
    }        
    
    public static void combineTestResults(String classifier, String simulator){
        File f=new File(DataSets.resultsPath+"/"+simulator);
        if(!f.exists() || !f.isDirectory()){
            f.mkdir();
        }
        else{
            boolean results=false;
            for(int i=0;i<100 && !results;i++){
    //Check fold exists            
                f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                if(f.exists())
                    results=true;
            }

            if(results){
                OutFile of=new OutFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+".csv");
                for(int i=0;i<100;i++){
        //Check fold exists            
                    f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                        inf.readLine();
                        inf.readLine();
                        of.writeLine(i+","+inf.readDouble());
                    }
                }
                of.closeFile();
            }
        }
    }
    
    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        double acc=0;
        OutFile p=new OutFile(preds+"/testFold"+sample+".csv");

// hack here to save internal CV for further ensembling   
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(preds+"/trainFold"+sample+".csv");        
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(train);
            int[][] predictions=new int[test.numInstances()][2];
            for(int j=0;j<test.numInstances();j++){
                predictions[j][0]=(int)test.instance(j).classValue();
                test.instance(j).setMissing(test.classIndex());//Just in case ....
            }
            for(int j=0;j<test.numInstances();j++)
            {
                predictions[j][1]=(int)c.classifyInstance(test.instance(j));
                if(predictions[j][0]==predictions[j][1])
                    acc++;
            }
            acc/=test.numInstances();
            String[] names=preds.split("/");
            p.writeLine(names[names.length-1]+","+c.getClass().getName()+",test");
            if(c instanceof SaveCVAccuracy)
                p.writeLine(((SaveCVAccuracy)c).getParameters());
            else if(c instanceof SaveableEnsemble)
                p.writeLine(((SaveableEnsemble)c).getParameters());
            else
                p.writeLine("NoParameterInfo");
            p.writeLine(acc+"");
            for(int j=0;j<test.numInstances();j++){
                p.writeString(predictions[j][0]+","+predictions[j][1]+",");
                double[] dist =c.distributionForInstance(test.instance(j));
                for(double d:dist)
                    p.writeString(","+d);
                p.writeString("\n");
            }
        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes and "+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }

    public static void collateAllResults(){
        DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\";
        for(String s:allClassifiers){
            for(String a:allSimulators){
                combineTestResults(s,a);
            }
        }
        for(String a:allSimulators){
            if(new File(DataSets.resultsPath+a).exists()){
                System.out.println(" Simulation = "+a);
                OutFile of=new OutFile(DataSets.resultsPath+a+"CombinedResults.csv");
                InFile[] ins=new InFile[allClassifiers.length];
                int count=0;
                for(String s:allClassifiers){
                    if(new File(DataSets.resultsPath+a+"\\"+s+".csv").exists()){
                        System.out.println(" Doing "+a+" and "+s);
                        of.writeString(s+",");
                        ins[count++]=new InFile(DataSets.resultsPath+a+"\\"+s+".csv");
                    }
                }
                of.writeString("\n");
                for(int i=0;i<100;i++){
                    for(int j=0;j<count;j++){
                        ins[j].readInt();
                        double acc=ins[j].readDouble();
                        of.writeString(acc+",");
                    }
                    of.writeString("\n");
                }
            }
        }
   }
/** Stand alone method to exactly reproduce shapelet experiment */    
    public static void runShapeletSimulatorExperiment(){
        Model.setDefaultSigma(1);
        seriesLength=300;
        casesPerClass=new int[]{50,50};
        String[] classifiers={"RotF","DTW","FastShapelets","ST","BOSS"};
//            "EE","HESCA","TSF","TSBF","FastShapelets","ST","LearnShapelets","BOP","BOSS","RISE","COTE"};
        OutFile of=new OutFile("C:\\Temp\\ShapeletSimExperiment.csv");
        double prop=0.5;
        of.writeLine("Shapelet Sim, series length= "+seriesLength+" cases class 0 ="+casesPerClass[0]+" class 1"+casesPerClass[0]+" train proportion = "+prop);
        of.writeString("Rep");
        for(String s:classifiers)
            of.writeString(","+s);
        of.writeString("\n");
        for(int i=0;i<100;i++){
            of.writeString(i+",");
//Generate data
            Model.setGlobalRandomSeed(i);
            Instances data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
//Split data
            Instances[] split=InstanceTools.resampleInstances(data, i,0.5);
            for(String str:classifiers){
                Classifier c;
        //Build classifiers            
                switch(str){
                    case "RotF":
                        c=new RotationForest();
                        break;
                    case "DTW":
                        c=new DTW_1NN();
                        break;
                    case "EE":    
                        c=new ElasticEnsemble();
                        break;                          
                    case "TSF":
                        c=new TSF();
                        break;
                    case "TSBF":
                        c=new TSBF();
                        break;
                    case "FastShapelets":
                        c=new FastShapelets();
                        break;
                    case "ST":
                        c=new ST_Ensemble();
                            ((ST_Ensemble)c).setOneMinuteLimit();
                        break;
                    case "LearnShapelets":
                        c=new LearnShapelets();
                        break;
                    case "BOP":
                        c=new BagOfPatterns();
                        break;
                    case "BOSS":
                        c=new BOSSEnsemble();
                        break;
                    case "COTE":
                        c=new COTE();
                        break;
                    case "RISE":
                        c=new RISE();
                        ((RISE)c).setTransformType("PS_ACF");
                        ((RISE)c).setNosBaseClassifiers(500);
                        break;
                    default:
                        throw new RuntimeException(" UNKNOWN CLASSIFIER "+str);
                }
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(c, split[0], split[1]);
                of.writeString(acc+",");
                System.out.println(i+" "+str+" acc ="+acc);
            }
            of.writeString("\n");
        }
        
    }
/**
Parameters: 
*   Train set size from 10 to 100
*   Series length 100 to 1000
 */    
    public static void shapeletParameterTest(){
        for(int seriesLength=100;seriesLength<=1000;seriesLength+=100){
            for(int trainSize=10;trainSize<=100;trainSize+=10){
                
            }
        }
    }
    
    public static void main(String[] args){
//          shapeletParameterTest();
//        runShapeletSimulatorExperiment();
      createAllScripts();
  //      collateAllResults();
        System.exit(0);
        String[] paras;
        if(args.length>0){
            paras=args;
            DataSets.resultsPath=DataSets.clusterPath+"Results/SimulationExperiments/";
            runSimulationExperiment(paras);
        }
        else{
            DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\";
            local=true;
            OutFile oft=new OutFile("C:\\temp\\test.csv");
//            for(String s:allClassifiers){
                String classifier="FastShapelets";
                String simulator="Shapelet";
                for(int i=1;i<=100;i++){
                    classifier="FastShapelets";
                    paras=new String[]{simulator,classifier,""+i};
                    double a=runSimulationExperiment(paras);
                    classifier="ST";
                    paras=new String[]{simulator,classifier,""+i};
                    double b=runSimulationExperiment(paras);
                    classifier="LearnShapelets";
                    paras=new String[]{simulator,classifier,""+i};
                     double c=runSimulationExperiment(paras);
                    oft.writeLine(i+","+a+","+b+","+c);
                    System.out.println(i+","+a+","+b+","+c);
                }
//            }
        }
    }
}
