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
import statistics.simulators.SimulateShapeletData;
import tsc_algorithms.BOSSEnsemble;
import tsc_algorithms.COTE;
import tsc_algorithms.ElasticEnsemble;
import tsc_algorithms.PSACF_Ensemble;
import tsc_algorithms.ST_Ensemble;
import tsc_algorithms.TSF;
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
/**
 *
 * @author ajb
 */
public class SimulationExperiment {
    static int []casesPerClass={50,50};
    static int seriesLength=300;
    static String[] allClassifiers={ //Benchmarks
        "RotF","DTW",
        //Whole series
        "DD_DTW","DTD_C","EE",
        //Interval
        "TSF","TSBF","LPS",
        //Shapelet
        "FastShapelets","ST","LearnShapelets",
        //Dictionary
        "BOP","BOSS",
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
                of.writeLine("#BSUB -J "+s+"[1-100]");
                of.writeLine("#BSUB -oo output/"+a+".out");
                of.writeLine("#BSUB -eo error/"+a+".err");
                of.writeLine("#BSUB -R \"rusage[mem=6000]\"");
                of.writeLine("#BSUB -M 6000");
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
                data=SimulateShapeletData.getShapeletData(seriesLength,casesPerClass);
                break;
            default:
                throw new RuntimeException(" UNKNOWN SIMULATOR ");
            
        }
        return data;
    }
    

//arg[0]: simulator
//arg[1]: classifier
//arg[2]: fold number    
    public static void runSimulationExperiment(String[] args){
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
            }
 //       of.writeString("\n");
    }        
    
    public static void combineTestResults(String classifier, String simulator){
        File f=new File(DataSets.resultsPath+"/"+simulator);
        if(!f.exists() || !f.isDirectory()){
            f.mkdir();
        }
        else{
            boolean results=false;
            for(int i=1;i<=100 && !results;i++){
    //Check fold exists            
                f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                if(f.exists())
                    results=true;
            }

            if(results){
                OutFile of=new OutFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+".csv");
                for(int i=1;i<=100;i++){
        //Check fold exists            
                    f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                        inf.readLine();
                        inf.readLine();
                        of.writeLine(i+","+inf.readDouble());
                    }
                }
            }
        }
    }
    
    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        double acc=0;
        OutFile p=new OutFile(preds+"/testFold"+sample+".csv");

// hack here to save internal CV for furhter ensembling   
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(preds+"/trainFold"+sample+".csv");        
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(train);
            int[][] predictions=new int[test.numInstances()][2];
            for(int j=0;j<test.numInstances();j++)
            {
                predictions[j][0]=(int)test.instance(j).classValue();
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
    }
    public static void main(String[] args){
        //createAllScripts();
        //collateAllResults();
        //System.exit(0);
        String[] paras;
        if(args.length>0){
            paras=args;
            DataSets.resultsPath=DataSets.clusterPath+"Results/SimulationExperiments/";
            runSimulationExperiment(paras);
        }
        else{
            DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\";
            for(String s:allClassifiers){
                for(int i=1;i<=100;i++){
                    paras=new String[]{"Shapelet",s,""+i};
                    runSimulationExperiment(paras);
                }
            }
        }
    }
}
