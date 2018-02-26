/**
 *
 * @author ajb
 *local class to run experiments with the UCR-UEA or UCI data


*/
package development;

import vector_classifiers.RotationForestLimitedAttributes;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.BagOfPatterns;
import timeseriesweka.classifiers.DD_DTW;
import timeseriesweka.classifiers.DTD_C;
import timeseriesweka.classifiers.FastDTW_1NN;
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.FastShapelets;
import timeseriesweka.classifiers.FlatCote;
import timeseriesweka.classifiers.HiveCote;
import timeseriesweka.classifiers.LPS;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.NN_CID;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.SAXVSM;
import timeseriesweka.classifiers.ST_HESCA;
import timeseriesweka.classifiers.SlowDTW_1NN;
import timeseriesweka.classifiers.TSBF;
import timeseriesweka.classifiers.TSF;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import vector_classifiers.TunedSVM;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import utilities.ClassifierResults;
import vector_classifiers.HESCA;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import timeseriesweka.classifiers.FastWWS.FastDTWWrapper;
import utilities.GenericTools;
import vector_classifiers.RotationForestBootstrap;
import vector_classifiers.SaveEachParameter;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import vector_classifiers.TunedMLP;
import vector_classifiers.TunedTwoLayerMLP;
import vector_classifiers.TunedXGBoost;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;


    

public class Experiments implements Runnable{
//For threaded version
    String[] args;
    public static int folds=30; 
    static int numCVFolds = 10;
    static boolean debug=true;
    static boolean checkpoint=false;
    static boolean generateTrainFiles=false;
    static Integer parameterNum=0;
    static boolean singleFile=false;
    static boolean foldsInFile=false;
    
public static String threadClassifier="ED";    
public static String[] cmpv2264419={
"adult",
};
public static String[] ajb17pc={
"abalone"
};    
public static String[] cmpv2202398={
"adult"
};
//TODO
/*

*/
public static String[] laptop={
"balloons"  
};
    public static Classifier setClassifier(String classifier, int fold){
        Classifier c=null;
        TunedSVM svm=null;
        switch(classifier){
//TIME DOMAIN CLASSIFIERS   
            
            case "ED":
                c=new ED1NN();
                break;
            case "C45":
                c=new J48();
                break;
            case "NB":
                c=new NaiveBayes();
                break;
            case "SVML":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                ((SMO)c).setKernel(p);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMQ": case "SVMQuad":
                c=new SMO();
                PolyKernel p2=new PolyKernel();
                p2.setExponent(2);
                ((SMO)c).setKernel(p2);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMRBF": 
                c=new SMO();
                RBFKernel rbf=new RBFKernel();
                rbf.setGamma(0.5);
                ((SMO)c).setC(5);
                ((SMO)c).setKernel(rbf);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                
                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "RandFOOB":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRandomForest)c).setEstimateAcc(true);
                ((TunedRandomForest)c).setSeed(fold);
                ((TunedRandomForest)c).setDebug(debug);
                
                break;
            case "RandF":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);
                break;
            case "RotF":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFBootstrap":
                c= new RotationForestBootstrap();
                ((RotationForestBootstrap)c).setNumIterations(200);
                ((RotationForestBootstrap)c).setSeed(fold);
                ((RotationForestBootstrap)c).tuneParameters(false);
                ((RotationForestBootstrap)c).setSeed(fold);
                ((RotationForestBootstrap)c).estimateAccFromTrain(false);
                break;
            case "RotFLimited":
                c= new RotationForestLimitedAttributes();
                ((RotationForestLimitedAttributes)c).setNumIterations(200);
                ((RotationForestLimitedAttributes)c).tuneParameters(false);
                ((RotationForestLimitedAttributes)c).setSeed(fold);
                ((RotationForestLimitedAttributes)c).estimateAccFromTrain(false);
                break;
            case "TunedRandF":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);             
                ((TunedRandomForest)c).setDebug(debug);
                break;
            case "TunedRandFOOB":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedRotF":
                c= new TunedRotationForest();
                ((TunedRotationForest)c).tuneParameters(true);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedSVMRBF":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.RBF);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSVMQuad":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(1089);                
                c= svm;
                break;
            case "TunedSVMLinear":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.LINEAR);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(1089);
                c= svm;
                break;
            case "TunedSVMPolynomial":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.POLYNOMIAL);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSVMKernel":
                svm=new TunedSVM();
                svm.optimiseParas(true);
                svm.optimiseKernel(true);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSingleLayerMLP":
                TunedMLP mlp=new TunedMLP();
                mlp.setParamSearch(true);
                mlp.setTrainingTime(200);
                mlp.setSeed(fold);
                c= mlp;
                break;
            case "TunedTwoLayerMLP":
                TunedTwoLayerMLP mlp2=new TunedTwoLayerMLP();
                mlp2.setParamSearch(true);
                mlp2.setSeed(fold);
                c= mlp2;
                break;
            case "RandomRotationForest1":
                c= new RotationForestLimitedAttributes();
                ((RotationForestLimitedAttributes)c).setNumIterations(200);
                ((RotationForestLimitedAttributes)c).setMaxNumAttributes(100);
                break;
            case "Logistic":
                c= new Logistic();
                break;
            case "HESCA":
                c=new HESCA();
                ((HESCA)c).setRandSeed(fold);
                break;
//ELASTIC CLASSIFIERS     
            case "EE": case "ElasticEnsemble":
                c=new ElasticEnsemble();
                break;
            case "DTW":
                c=new DTW1NN();
                ((DTW1NN )c).setWindow(1);
                break;
            case "SLOWDTWCV":
//                c=new DTW1NN();
                c=new SlowDTW_1NN();
                ((SlowDTW_1NN)c).optimiseWindow(true);
                break;
            case "DTWCV":
//                c=new DTW1NN();
//                c=new FastDTW_1NN();
//                ((FastDTW_1NN)c).optimiseWindow(true);
//                break;
//            case "FastDTWWrapper":
                c= new FastDTWWrapper();
                break;
            case "DD_DTW":
                c=new DD_DTW();
                break;
            case "DTD_C":
                c=new DTD_C();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "MSM":
                c=new MSM1NN();
                break;
            case "TWE":
                c=new MSM1NN();
                break;
            case "WDTW":    
                c=new WDTW1NN();
                break;
                
            case "LearnShapelets": case "LS":
                c=new LearnShapelets();
                break;
            case "FastShapelets": case "FS":
                c=new FastShapelets();
                break;
            case "ShapeletTransform": case "ST": case "ST_Ensemble":
                c=new ST_HESCA();
//Default to 1 day max run: could do this better
                ((ST_HESCA)c).setOneDayLimit();
                
                break;
            case "TSF":
                c=new TSF();
                break;
            case "RISE":
                c=new RISE();
                break;
            case "RISEV2":
                c=new RiseV2();
                ((RiseV2)c).buildFromSavedData(true);
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "BOP": case "BoP": case "BagOfPatterns":
                c=new BagOfPatterns();
                break;
             case "BOSS": case "BOSSEnsemble": 
                c=new BOSS();
                break;
             case "SAXVSM": case "SAX": 
                c=new SAXVSM();
                break;
             case "LPS":
                c=new LPS();
                break; 
             case "FlatCOTE":
                c=new FlatCote();
                break; 
             case "HiveCOTE":
                c=new HiveCote();
                break; 
             case "XGBoost":
                 c=new TunedXGBoost();
                ((TunedXGBoost)c).setSeed(fold);
                ((TunedXGBoost)c).setTuneParameters(false);
                 break;
            case "TunedXGBoost":
                 c=new TunedXGBoost();
                ((TunedXGBoost)c).setSeed(fold);
                ((TunedXGBoost)c).setTuneParameters(true);
                 break;
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
       
//Threaded version
    public void run(){
        try {      
            System.out.print("Running ");
            for(String str:args)
                System.out.print(str+" ");
            System.out.print("\n");
            singleExperiment(args);
//            Experiments.singleClassifierAndFoldTrainTestSplit(args);
        } catch (Exception ex) {
            System.out.println("ERROR, cannot run experiment :");
            for(String str:args)
                System.out.print(str+",");
        }
    }
    
    
/* MUST BE at least Arguments:
    1: Problem path args[0]
    2. Results path args[1]
    3. booleanwWhether to CV to generate train files (true/false)
    4. Classifier =args[3];
    5. String problem=args[4];
    6. int fold=Integer.parseInt(args[5])-1;
Optional    
    7. boolean whether to checkpoint parameter search (true/false)
    8. integer for specific parameter search (0 indicates ignore this) 
    */  
    public static void sanityCheck(String c) throws Exception{
        DataSets.problemPath="\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\TSCProblems";
        DataSets.resultsPath="C:\\Temp\\";
        threadClassifier=c;
        depreciatedThreadedExperiment("TSCProblems",1,10);

    }
    public static void debugExperiment() throws Exception{
//        sanityCheck("DTWCV");
//        System.exit(0);
    //Debug run
        DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/TSCProblems/";
        DataSets.resultsPath="C:\\Temp\\";
        File f=new File(DataSets.resultsPath);
        if(!f.isDirectory()){
            f.mkdirs();
        }
        generateTrainFiles=false;
        checkpoint=false;
        parameterNum=0;
        debug=true;
            String[] newArgs={"FastDTWWrapper","ItalyPowerDemand","1"};
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);
//                for(String str:DataSets.UCIContinuousFileNames){
//                }
        System.exit(0);

        
    }
    
    public static void main(String[] args) throws Exception{

//IF args are passed, it is a cluster run. Otherwise it is a local run, either threaded or not
        debug=false;

//        foldsInFile=true;
        for(String str:args)
            System.out.println(str);
        if(args.length<6){
            boolean threaded=false;
            if(debug){
                debugExperiment();
            }
            else{ 
    //Args 1 and 2 are problem and results path
                String[] newArgs=new String[6];
                newArgs[0]="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/TSCProblems/";//All on the beast now
                newArgs[1]="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/TSCProblems/";
    //Arg 3 argument is whether to cross validate or not and produce train files
                newArgs[2]="false";
    // Arg 4,5,6 Classifier, Problem, Fold             
                newArgs[3]="FastDTWWrapper";
//These are set in the localX method
//              newArgs[4]="Adiac";
//                newArgs[5]="1";
                String[] problems=DataSets.fileNames;
///                problems=new String[]{"Distal"};
                int folds=1;
                if(threaded){//Do problems listed threaded 
                    localThreadedRun(newArgs,problems,folds);
                    
                    
                }
                else //Do the problems listed sequentially
                    localSequentialRun(newArgs,problems,folds);
            }


//Threaded run
 //             threadedExperiment("cmpv2202398");  
 /*           generateTrainFiles=false;
            parameterNum=0;
            threadClassifier="FastDTWWrapper";
            threadedExperiment("ajb17pc",1,1);  
//             threadedExperiment("cmpv2264419");  

// //             threadedExperiment("UCIContinuous");  
//              threadedExperiment("LargeProblems");  
            }
  */          
            
        }
        else{    
            singleExperiment(args);
        }
    
    }
    public static void depreciatedThreadedExperiment(String dataSet, int startFold, int endFold) throws Exception{
        
        int cores = Runtime.getRuntime().availableProcessors();        
        System.out.println("# cores ="+cores);
 //     cores=1; //debug       
        
        ExecutorService executor = Executors.newFixedThreadPool(cores);
        Experiments exp;
        DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/"+dataSet+"/";//Problem Path
        DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/"+dataSet+"/";         //Results path
        String[] problems=DataSets.fileNames;
        parameterNum=0;   
        String classifier=threadClassifier;
        switch(dataSet){
            case "UCIContinuous"://Do all UCI
                problems=DataSets.UCIContinuousFileNames;
            break;
            case "TSCProblems": case "TSC Problems": //Do all Repo
                problems=DataSets.fileNames;
            break;
            
            
        }
        if(dataSet.equals("cmpv2202398")){
            DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/";//Problem Path
            DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/UCIContinuous/";         //Results path
            problems=cmpv2202398;
            
        }
        else if(dataSet.equals("cmpv2264419")){
            DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/";//Problem Path
            DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/UCIContinuous/";         //Results path
            problems=cmpv2264419;
            
        }
        else if(dataSet.equals("ajb17pc")){
            DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/";//Problem Path
            DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/UCIContinuous/";         //Results path
            problems=ajb17pc;
            
        }
            
        ArrayList<String> names=new ArrayList<>();
        for(String str:problems)
            names.add(str);
//        Collections.reverse(names);
        generateTrainFiles=true;
        checkpoint=true;
        for(int i=0;i<names.size();i++){//Iterate over problems
//            if(isPig(names.get(i)))
//                continue;
            for(int j=startFold;j<=endFold;j++){//Iterate over folds
                String[] args=new String[3];
                args[0]=classifier;
                args[1]=names.get(i);
                args[2]=""+j;
                exp=new Experiments();
                exp.args=args;
                executor.execute(exp);
            }
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");            
        
    }
    public static boolean isPig(String str){
//        if(str.equals("adult")||str.equals("miniboone")||str.equals("chess-krvk"))
//            return true;
        return false;
        
    }
    public static void singleExperiment(String[] args) throws Exception{
            DataSets.problemPath=args[0];
            DataSets.resultsPath=args[1];
//Arg 3 argument is whether to cross validate or not and produce train files
            generateTrainFiles=Boolean.parseBoolean(args[2]);
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
                f.setWritable(true, false);
            }
// Arg 4,5,6 Classifier, Problem, Fold             
            String[] newArgs=new String[3];
            for(int i=0;i<3;i++)
                newArgs[i]=args[i+3];
//OPTIONAL
//  Arg 7:  whether to checkpoint        
            checkpoint=false;
            if(args.length>=7){
                String s=args[args.length-1].toLowerCase();
                if(s.equals("true"))
                    checkpoint=true;
            }
//Arg 8: if present, do a single parameter split
            parameterNum=0;
            if(args.length>=8){
                parameterNum=Integer.parseInt(args[7]);
            }
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);        
    }
    /** Run a given classifier/problem/fold combination with associated file set up
 @param args: 
 * args[0]: Classifier name. Create classifier with setClassifier
 * args[1]: Problem name
 * args[2]: Fold number. This is assumed to range from 1, hence we subtract 1
 * (this is because of the scripting we use to run the code on the cluster)
 *          the standard archive folds are always fold 0
 * 
 * NOTES: 
 * 1. this assumes you have set DataSets.problemPath to be where ever the 
 * data is, and assumes the data is in its own directory with two files, 
 * args[1]_TRAIN.arff and args[1]_TEST.arff 
 * 2. assumes you have set DataSets.resultsPath to where you want the results to
 * go It will NOT overwrite any existing results (i.e. if a file of non zero 
 * size exists)
 * 3. This method just does the file set up then calls the next method. If you 
 * just want to run the problem, go to the next method
* */
    public static void singleClassifierAndFoldTrainTestSplit(String[] args) throws Exception{
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        String predictions = DataSets.resultsPath+classifier+"/Predictions/"+problem;
        File f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
        
        //Check whether fold already exists, if so, dont do it, just quit
        if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
            Classifier c=setClassifier(classifier,fold);
            
            //Sample the dataset
            Instances train,test;
            Instances[] data;
//Shapelet special case, hard coded,because all folds are pre-generated             
            if(foldsInFile){
                f=new File(DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN.arff");
                File f2=new File(DataSets.problemPath+problem+"/"+problem+fold+"_TEST.arff");
                if(!f.exists()||!f2.exists())
                    throw new Exception(" Problem files "+DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN.arff not found");
                train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN");
                test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+fold+"_TEST");
                data=new Instances[2];
                data[0]=train;
                data[1]=test;

            }
            else{
    //If there is a train test split, use that. Otherwise, randomly split 50/50            
                f=new File(DataSets.problemPath+problem+"/"+problem+"_TRAIN.arff");
                File f2=new File(DataSets.problemPath+problem+"/"+problem+"_TEST.arff");
                if(!f.exists()||!f2.exists())
                    singleFile=true;
                if(singleFile){
                    Instances all = ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
                    data = InstanceTools.resampleInstances(all, fold, .5);            
                }else{
                    train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
                    test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
                    data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
                }
            }
            if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
            {
                checkpoint=false;
//Check if it already exists, if it does, exit
                if(CollateResults.validateSingleFoldFile(predictions+"/fold"+fold+"_"+parameterNum+".csv")){ //Exit
                    System.out.println("Fold "+predictions+"/fold"+fold+"_"+parameterNum+".csv  already exists");
                    return; //Aready done
                }
            }
            
            double acc = singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
            System.out.println("Classifier="+classifier+", Problem="+problem+", Fold="+fold+", Test Acc,"+acc);
        }
    }
/**
 * 
 * @param train: the standard train fold Instances from the archive 
 * @param test: the standard test fold Instances from the archive
 * @param c: Classifier to evaluate
 * @param fold: integer to indicate which fold. Set to 0 to just use train/test
 * @param resultsPath: a string indicating where to store the results
 * @return the accuracy of c on fold for problem given in train/test
 * 
 * NOTES:
 * 1.  If the classifier is a SaveableEnsemble, then we save the internal cross 
 * validation accuracy and the internal test predictions
 * 2. The output of the file testFold+fold+.csv is
 * Line 1: ProblemName,ClassifierName, train/test
 * Line 2: parameter information for final classifier, if it is available
 * Line 3: test accuracy
 * then each line is
 * Actual Class, Predicted Class, Class probabilities 
 * 
 * 
 */    
    public static double singleClassifierAndFoldTrainTestSplit(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        String testFoldPath="/testFold"+fold+".csv";
        String trainFoldPath="/trainFold"+fold+".csv";
        
        ClassifierResults trainResults = null;
        ClassifierResults testResults = null;
        if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
        {
//If TunedRandForest or TunedRotForest need to let the classifier know the number of attributes 
//n orderto set parameters
            if(c instanceof TunedRandomForest)
                ((TunedRandomForest)c).setNumFeaturesInProblem(train.numAttributes()-1);
            checkpoint=false;
            ((ParameterSplittable)c).setParametersFromIndex(parameterNum);
//            System.out.println("classifier paras =");
            trainFoldPath="/fold"+fold+"_"+parameterNum+".csv";
            generateTrainFiles=true;
        }
        else{
//Only do all this if not an internal fold
    // Save internal info for ensembles
            if(c instanceof SaveableEnsemble)
               ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
            if(checkpoint && c instanceof SaveEachParameter){     
                ((SaveEachParameter) c).setPathToSaveParameters(resultsPath+"/fold"+fold+"_");
            }
        }
        
        try{             
            if(generateTrainFiles){
                if(c instanceof TrainAccuracyEstimate){ //Classifier will perform cv internally
                    ((TrainAccuracyEstimate)c).writeCVTrainToFile(resultsPath+trainFoldPath);
                    File f=new File(resultsPath+trainFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                }
                else{ // Need to cross validate here
                    
                    if(c instanceof RiseV2 && ((RiseV2)c).getBuildFromSavedData()){
                        //Write some internal crossvalidation that can deal with read from files.
                    }else{
                        CrossValidator cv = new CrossValidator();
                        cv.setSeed(fold);
                        int numFolds = Math.min(train.numInstances(), numCVFolds);
                        cv.setNumFolds(numFolds);
                        trainResults=cv.crossValidateWithStats(c,train);
                    }    
                }
            }
            
            //Build on the full train data here
            long buildTime=System.currentTimeMillis();
            c.buildClassifier(train);
            buildTime=System.currentTimeMillis()-buildTime;
            
            if (generateTrainFiles) { //And actually write the full train results if needed
                if(!(c instanceof TrainAccuracyEstimate)){ 
                    OutFile trainOut=new OutFile(resultsPath+trainFoldPath);
                    trainOut.writeLine(train.relationName()+","+c.getClass().getName()+",train");
                    if(c instanceof SaveParameterInfo )
                        trainOut.writeLine(((SaveParameterInfo)c).getParameters()); //assumes build time is in it's param info, is for tunedsvm
                    else 
                        trainOut.writeLine("BuildTime,"+buildTime+",No Parameter Info");
                    trainOut.writeLine(trainResults.acc+"");
                    trainOut.writeLine(trainResults.writeInstancePredictions());
                    //not simply calling trainResults.writeResultsFileToString() since it looks like those that extend SaveParameterInfo will store buildtimes
                    //as part of their params, and so would be written twice
                    trainOut.closeFile();
                    File f=new File(resultsPath+trainFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
            }
            if(parameterNum==0)//Not a single parameter fold
            {  
                //Start of testing, only doing this if the test file doesnt exist
                //This is checked before the buildClassifier also, but we have a special case for the file builder
                //that copies the results over in buildClassifier. No harm in checking again!
                if(!CollateResults.validateSingleFoldFile(resultsPath+testFoldPath)){
                    int numInsts = test.numInstances();
                    int pred;
                    testResults = new ClassifierResults(test.numClasses());
                    double[] trueClassValues = test.attributeToDoubleArray(test.classIndex()); //store class values here

                    for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                        test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)

                        //make prediction
                        double[] probs=c.distributionForInstance(test.instance(testInstIndex));
                        testResults.storeSingleResult(probs);
                    }
                    testResults.finaliseResults(trueClassValues); 

                    //Write results
                    OutFile testOut=new OutFile(resultsPath+testFoldPath);
                    testOut.writeLine(test.relationName()+","+c.getClass().getName()+",test");
                    if(c instanceof SaveParameterInfo)
                      testOut.writeLine(((SaveParameterInfo)c).getParameters());
                    else
                        testOut.writeLine("No parameter info");
                    testOut.writeLine(testResults.acc+"");
                    testOut.writeString(testResults.writeInstancePredictions());
                    testOut.closeFile();
                    File f=new File(resultsPath+testFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
                return testResults.acc;
            }
            else
                 return 0;//trainResults.acc;   
        } catch(Exception e) {
            System.out.println(" Error ="+e+" in method simpleExperiment");
            e.printStackTrace();
            System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
            System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes and "+test.numInstances()+" instances");
            System.out.println(" Classifier ="+c.getClass().getName()+" fold = "+fold);
            System.out.println(" Results path is "+ resultsPath);
                    
            return Double.NaN;
        }
    }    

    
    public static void localSequentialRun(String[] standardArgs,String[] problemList, int folds) throws Exception{
        for(String str:problemList){
                System.out.println("Problem ="+str);
            for(int i=1;i<=folds;i++){
                standardArgs[4]=str;
                standardArgs[5]=i+"";
                singleExperiment(standardArgs);
                
            }
        }
        
        
    }
    public static void localThreadedRun(String[] standardArgs,String[] problemList, int folds) throws Exception{
        int cores = Runtime.getRuntime().availableProcessors();        
        System.out.println("# cores ="+cores);
        cores=cores/2;
        System.out.println("# threads ="+cores);
        ExecutorService executor = Executors.newFixedThreadPool(cores);
        Experiments exp;
        for(String str:problemList){
            for(int i=1;i<=folds;i++){
                String[] args=new String[standardArgs.length];//Need to clone them!
                for(int j=0;j<standardArgs.length;j++)
                    args[j]=standardArgs[j];
                args[4]=str;
                args[5]=i+"";
                exp=new Experiments();
                exp.args=args;
                executor.execute(exp);
            }
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");            
    }
    
   public static void tonyTest() throws Exception{
        Instances all = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\hayes-roth\\hayes-roth");
        Instances[] data = InstanceTools.resampleInstances(all,0, .5);            
        RandomForest rf=new RandomForest();
        rf.setMaxDepth(0);
        rf.setNumFeatures(1);
        rf.setNumTrees(10);
        CrossValidator cv = new CrossValidator();
        ClassifierResults tempResults=cv.crossValidateWithStats(rf,data[0]);
                    tempResults.setName("RandFPara");
                    tempResults.setParas("maxDepth,"+rf.getMaxDepth()+",numFeatures,"+rf.getNumFeatures()+",numTrees,"+rf.getNumTrees());
                    System.out.println(tempResults.writeResultsFileToString());


}
 
}






