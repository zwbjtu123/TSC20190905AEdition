/*
Class to run one of various simulations.  
*/
package development;

import tsc_algorithms.depreciated.COTE;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
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
public class SimulationExperiments {
    static boolean local=false;
    static int []casesPerClass={50,50};
    static int seriesLength=300;
    static double trainProp=0.5;
    static String[] allClassifiers={ //Benchmarks
        "RotF","DTW","HESCA",
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
        "COTE","FLATCOTE","HIVECOTE"};
    static String[] allSimulators={"WholeSeries","Interval","Shapelet","Dictionary","ARMA"};
    
    
    public static Classifier createClassifier(String str) throws RuntimeException{
        Classifier c;
        switch(str){
            case "HESCA":
                c=new HESCA();
                break;
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
                   ((ST_Ensemble)c).setOneHourLimit();
                break;
            case "LearnShapelets":
                c=new LearnShapeletsFeb2015Version();
                ((LearnShapeletsFeb2015Version)c).setParamSearch(true);
//                ((LearnShapeletsFeb2015Version)c).fixParameters();
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
            case "FLATCOTE":
                c=new FlatCote();
                break;
            case "HIVECOTE":
                c=new HiveCote();
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
    
    public static void setStandardGlobalParameters(String str){
         switch(str){
            case "ARMA": case "AR":
                break;
            case "Shapelet": 
                casesPerClass=new int[]{50,50};
                seriesLength=300;
                trainProp=0.5;
                Model.setDefaultSigma(1);
                break;
            case "Dictionary":
                casesPerClass=new int[]{30,30};
                seriesLength=1000;
                trainProp=0.5;
                Model.setDefaultSigma(1);
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
                data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
                break;
            case "Dictionary":
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
    public static double runSimulationExperiment(String[] args,boolean useStandard){
   
        String simulator=args[0];
        if(useStandard)
            setStandardGlobalParameters(simulator);
        String classifier=args[1];
        Classifier c=createClassifier(classifier);
        int fold=Integer.parseInt(args[2])-1;
        Instances data=simulateData(args[0],fold);
        Instances[] split=InstanceTools.resampleInstances(data, fold,trainProp);
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
//            System.out.println("simulator ="+simulator+" Classifier ="+classifier+" Fold "+fold+" Acc ="+acc);
            return acc;
        }
 //       of.writeString("\n");
        return -1;
    }        
    
    public static void combineTestResults(String classifier, String simulator){
        int folds=100;
        File f=new File(DataSets.resultsPath+"/"+simulator);
        if(!f.exists() || !f.isDirectory()){
            f.mkdir();
        }
        else{
            boolean results=false;
            for(int i=0;i<folds && !results;i++){
    //Check fold exists            
                f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                if(f.exists())
                    results=true;
            }

            if(results){
                OutFile of=new OutFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+".csv");
                for(int i=0;i<folds;i++){
        //Check fold exists            
                    f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                    if(f.exists() && f.length()>0){
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
        int folds=100;
        for(String a:allSimulators){
            if(new File(DataSets.resultsPath+a).exists()){
                System.out.println(" Simulation = "+a);
                OutFile of=new OutFile(DataSets.resultsPath+a+"CombinedResults.csv");
                InFile[] ins=new InFile[allClassifiers.length];
                int count=0;
                for(String s:allClassifiers){
                    File f=new File(DataSets.resultsPath+a+"\\"+s+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(DataSets.resultsPath+a+"\\"+s+".csv");
                        int lines=inf.countLines();
                        if(lines>=folds){
                            System.out.println(" Doing "+a+" and "+s);
                            of.writeString(s+",");
                            ins[count++]=new InFile(DataSets.resultsPath+a+"\\"+s+".csv");
                        }
                    }
                }
                of.writeString("\n");
                for(int i=0;i<folds;i++){
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
/** 
 * FINAL VERSION
 * Stand alone method to exactly reproduce shapelet experiment which 
 we normally 
 */    
    public static void runShapeletSimulatorExperiment(){
        Model.setDefaultSigma(1);
        seriesLength=300;
        casesPerClass=new int[]{50,50};
        String[] classifiers={"RotF","DTW","FastShapelets","ST","BOSS"};
//            "EE","HESCA","TSF","TSBF","FastShapelets","ST","LearnShapelets","BOP","BOSS","RISE","COTE"};
        OutFile of=new OutFile("C:\\Temp\\ShapeletSimExperiment.csv");
        setStandardGlobalParameters("Shapelet");
        of.writeLine("Shapelet Sim, series length= "+seriesLength+" cases class 0 ="+casesPerClass[0]+" class 1"+casesPerClass[0]+" train proportion = "+trainProp);
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
            Instances[] split=InstanceTools.resampleInstances(data, i,trainProp);
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

/** Method to run the error experiment, default setttings 
 * 
 */    
    public static void runErrorExperiment(String[] args, boolean splitByFold){
        String simulator=args[0];
        String classifier=args[1];

        int e=Integer.parseInt(args[2])-1;
//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator+"Error");
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"Error/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();

        if(splitByFold){
            //E encodes the error and the job number. So 
            int er=e/100;    //Integer division
            double error=(er/10.0);
            int fold=e%100;
            f=new File(predictions+"/testAcc"+e+"_"+fold+".csv");
            if(!f.exists() || f.length()==0){
                OutFile out=new OutFile(predictions+"/testAcc"+e+"_"+fold+".csv");
                Model.setDefaultSigma(error);
                Instances data=simulateData(simulator,50*(e+1)*fold);
                Classifier c=createClassifier(classifier);
                Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
                double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
                out.writeLine(a+"");
            }
        }
        else{   //Do the whole lot in a single job
            double error=(e/10.0);
            System.out.println("Error ="+error);
            
    //Check whether fold already exists, if so, dont do it, just quit
            f=new File(predictions+"/testAcc"+e+".csv");
            if(!f.exists() || f.length()==0){
    //Do the experiment: run over 100 folds
                OutFile out=new OutFile(predictions+"/testAcc"+e+".csv");
                double acc=0;
                double var=0;
                for(int fold=0;fold<100;fold++){
                    Model.setDefaultSigma(error);
                    Instances data=simulateData(simulator,50*(e+1)*fold);
                    Classifier c=createClassifier(classifier);
                    Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
                    double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
                    acc+=a;
                    var+=a*a;
                }
                out.writeLine(acc/100+","+var);
            }
        }
    }

    
    
/** Method to run the error experiment, default setttings 
 * 
 */    
    public static void runLengthExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
//Series length factor
        int l=Integer.parseInt(args[2]);
        seriesLength=10+(1+l)*50;   //l from 1 to 50
//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator+"Length");
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"Length/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();

//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testAcc"+l+".csv");
        if(!f.exists() || f.length()==0){
//Do the experiment: just measure the single fold accuracy
            OutFile out=new OutFile(predictions+"/testAcc"+l+".csv");
            double acc=0;
            double var=0;
            for(int fold=0;fold<100;fold++){
                Instances data=simulateData(simulator,seriesLength);
                Classifier c=createClassifier(classifier);
                Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
                double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
                acc+=a;
                var+=a*a;
            }
            out.writeLine(acc/100+","+var);
        }
    }
    

    public static void trainSetSizeExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
//Series length factor
        int l=Integer.parseInt(args[2]);
        trainProp=(double)(l/10.0);   //l from 1 to 9
//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator+"Length");
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"Length/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();

//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testAcc"+l+".csv");
        if(!f.exists() || f.length()==0){
//Do the experiment: just measure the single fold accuracy
            OutFile out=new OutFile(predictions+"/testAcc"+l+".csv");
            double acc=0;
            double var=0;
            for(int fold=0;fold<100;fold++){
                Instances data=simulateData(simulator,50*(l+1)*fold);
                Classifier c=createClassifier(classifier);
                Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
                double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
                acc+=a;
                var+=a*a;
            }
            out.writeLine(acc/100+","+var);
        }
    }
    
  //<editor-fold defaultstate="collapsed" desc="One off data processing methods">       
    public static void collateErrorResults(){
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\ShapeletError\\";
        OutFile out=new OutFile(path+"CollatedErrorResults.csv");
        out.writeString("Error");
        for(String s:allClassifiers)
            out.writeString(","+s);
        out.writeString("\n");
        for(int i=0;i<=20;i++){
            out.writeString((i/10.0)+"");
            for(String s:allClassifiers){
                File f= new File(path+s+"\\"+"testAcc"+i+".csv");
                if(f.exists() && f.length()>0){
                    InFile inf=new InFile(path+s+"\\"+"testAcc"+i+".csv");
                    double a=inf.readDouble();
                    out.writeString(","+a);
                }
                else
                    out.writeString(",");
            }
            out.writeString("\n");
        }       
    }
      
    public static void collateLengthResults(){
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\ShapeletLength\\";
        OutFile out=new OutFile(path+"CollatedLengthResults.csv");
        out.writeString("Error");
        for(String s:allClassifiers)
            out.writeString(","+s);
        out.writeString("\n");
        for(int i=0;i<10;i++){
            out.writeString((i*50+10)+"");
            for(String s:allClassifiers){
                File f= new File(path+s+"\\"+"testAcc"+i+".csv");
                if(f.exists() && f.length()>0){
                    InFile inf=new InFile(path+s+"\\"+"testAcc"+i+".csv");
                    double a=inf.readDouble();
                    out.writeString(","+a);
                }
                else
                    out.writeString(",");
            }
            out.writeString("\n");
        }       
    }
      

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
                of.writeLine("#BSUB -R \"rusage[mem=7000]\"");
                of.writeLine("#BSUB -M 8000");
                of.writeLine("module add java/jdk1.8.0_51");
                of.writeLine("java -jar Simulator.jar "+a+" "+ s+" $LSB_JOBINDEX");                
                
                of2.writeLine("bsub < Scripts/SimulatorExperiments/"+s+a+".bsub");
            }   
        }
    } 
    
    public static void createErrorORLengthScripts(boolean error){

//Generates cluster scripts for all combos of classifier and simulator     
       String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\SimulatorScripts\\";
//        for(String a:allSimulators)
       String ext;
       if(error)
           ext="Error";
       else
           ext="Length";
        String a ="Shapelet";
        {
            OutFile of2=new OutFile(path+"OC"+a+ext+".txt");
            for(String s:allClassifiers){
                OutFile of = new OutFile(path+"OC"+s+a+ext+".bsub");
                of.writeLine("#!/bin/csh");
                of.writeLine("#BSUB -q short");
                of.writeLine("#BSUB -J "+s+"[1-9]");
                of.writeLine("#BSUB -oo output/OC"+a+".out");
                of.writeLine("#BSUB -eo error/OC"+a+".err");
                of.writeLine("#BSUB -R \"rusage[mem=4000]\"");
                of.writeLine("#BSUB -M 6000");
                of.writeLine(" module add java/jdk/1.8.0_31");
//                of.writeLine("module add java/jdk1.8.0_51");
                of.writeLine("java -jar "+ext+".jar "+a+" "+ s+" $LSB_JOBINDEX");                
                
                of2.writeLine("bsub < Scripts/SimulatorExperiments/"+"OC"+s+a+ext+".bsub");
            }   
        }
    } 
    
  public static void deleteThisMethod(){
        String classifier="LearnShapelets";
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\ShapeletError\\"+classifier+"\\";
        OutFile of=new OutFile(path+classifier+"ZeroError.csv");
        double mean=0;
        for(int folds=0;folds<100;folds++){
            int index=folds;
            InFile inf=new InFile(path+"testAcc"+index+"_"+folds+".csv");
            of.writeLine(inf.readDouble()+"");
        }
        mean/=100;
        of.writeLine("0,"+mean);
        
    }
    public static void collateSingleFoldErrorResults(){
        String classifier="LearnShapelets";
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\ShapeletError\\"+classifier+"\\";
        OutFile of=new OutFile(path+classifier+".csv");
        for(int i=0; i<21;i++){
            double mean=0;
            for(int folds=0;folds<100;folds++){
                int index=i*100+folds;
                InFile inf=new InFile(path+"testAcc"+index+"_"+folds+".csv");
                mean+=inf.readDouble();
            }
            mean/=100;
            of.writeLine(i+","+mean);
        }
    }

    public static void collateSomeStuff(){
        String[] classifiers={"RotF","DTW","BOSS","ST"};
        for(String str:classifiers){
            String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\Dictionary\\"+str+"\\";
            OutFile of=new OutFile(path+str+".csv");
                double mean=0;
                for(int folds=0;folds<200;folds++){
                    File f=new File(path+"testFold"+folds+".csv");
                    if(f.exists() && f.length()>0){
                        InFile inf=new InFile(path+"testFold"+folds+".csv");
                        inf.readLine();
                        inf.readLine();
                        double x=inf.readDouble();
                        of.writeLine(folds+","+x);
                    }
                }
        }
    }
  //</editor-fold>

    
    public static void main(String[] args){
//        deleteThisMethod();
//        collateLengthResults();
//        collateErrorExperiments();
//      collateErrorResults();
//      createErrorORLengthScripts(false);
//     createErrorScripts();
//          shapeletParameterTest();
//        runShapeletSimulatorExperiment();
 //     createAllScripts();
//      collateAllResults();
 //       collateSomeStuff();
  //    System.exit(0);
        String[] paras;
        if(args.length>0){
            paras=args;
            DataSets.resultsPath=DataSets.clusterPath+"Results/SimulationExperiments/";
            double b=runSimulationExperiment(paras,true);
 //           System.out.println(paras[0]+","+paras[1]+","+","+paras[2]+" Acc ="+b);
//            runErrorExperiment(paras,true);
//              runLengthExperiment(paras);
        }
        else{
            DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\";
            local=true;
            String simulator="Dictionary";
            setStandardGlobalParameters(simulator);
            Model.setDefaultSigma(1);
            System.out.println(" Error ="+Model.defaultSigma+" Series Length ="+seriesLength+" training prop ="+trainProp+" nos classes ="+casesPerClass.length);
            for(int i:casesPerClass)
                System.out.println(" "+i);
            String classifier="RotF";
            for(int e=1;e<101;e++){
                classifier="RotF";
                String[] arg={simulator,classifier,e+""};
                double b=runSimulationExperiment(arg,false);
                classifier="DTW";
                String[] arg2={simulator,classifier,e+""};
                double c=runSimulationExperiment(arg2,false);
                classifier="BOSS";
                String[] arg3={simulator,classifier,e+""};
                double d=runSimulationExperiment(arg3,false);
                classifier="ST";
                String[] arg4={simulator,classifier,e+""};
                double a=runSimulationExperiment(arg4,false);
                System.out.println(arg[0]+","+arg[1]+","+","+arg[2]+" RotF acc ="+b+" DTW acc ="+c+" BOSS Acc ="+d+" ST Acc ="+a+"\n\n");
            }
        }
    }
}
