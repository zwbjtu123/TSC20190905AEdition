/**
 *
 * @author ajb
 *local class to run experiments with the UCI-UEA data


*/
package development;

import vector_classifiers.RandomRotationForest1;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
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
import utilities.GenericTools;
import vector_classifiers.SaveEachParameter;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;


public class Experiments{
    public static int folds=30; 
    static int numCVFolds = 10;
    static boolean debug=true;
    static boolean checkpoint=false;
    static boolean generateTrainFiles=true;
    static Integer parameterNum=0;
    static boolean UCIData=false;
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
                ((TunedRandomForest)c).tuneTree(false);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRandomForest)c).setSeed(fold);
                
                break;
            case "RandF":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).tuneTree(false);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);
                break;
            case "RotF":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                ((TunedRotationForest)c).tuneFeatures(false);
                ((TunedRotationForest)c).tuneTree(false);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedRandF":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).tuneTree(true);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);
                
                break;
            case "TunedRandFOOB":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).tuneTree(true);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedRotF":
                c= new TunedRotationForest();
                ((TunedRotationForest)c).tuneFeatures(true);
                ((TunedRotationForest)c).tuneTree(true);
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
            case "RandomRotationForest1":
                c= new RandomRotationForest1();
                ((RandomRotationForest1)c).setNumIterations(200);
                ((RandomRotationForest1)c).setMaxNumAttributes(100);
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
            case "DTWCV":
//                c=new DTW1NN();
                c=new FastDTW_1NN();
                ((FastDTW_1NN)c).optimiseWindow(true);
                
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
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
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
       
    public static void main(String[] args) throws Exception{
        for(String str:args)
            System.out.println(str);
        if(args.length<6){//Local run
            if(args!=null){
                System.out.println("Num args passed ="+args.length);
                for(String str:args)
                    System.out.println(str);
            }
            else
                System.out.println("No args passed");
                
            DataSets.problemPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
            DataSets.resultsPath="C:\\Temp\\";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            generateTrainFiles=true;
            checkpoint=true;
            parameterNum=0;
            String[] newArgs={"TunedSVMRBF","ItalyPowerDemand","2"};
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);
            System.exit(0);
        }
        else{    
            DataSets.problemPath=args[0];
            DataSets.resultsPath=args[1];
//Arg 3 argument is whether to cross validate or not and produce train files
            generateTrainFiles=Boolean.parseBoolean(args[2]);
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
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
            System.out.println("Checkpoint ="+checkpoint+" param number ="+parameterNum);
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);
        }
    
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
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
            Classifier c=setClassifier(classifier,fold);
            
            //Sample the dataset
            Instances train,test;
            Instances[] data;
            String uciTest=DataSets.problemPath.toUpperCase();
            if(uciTest.contains("UCI"))
                UCIData=true;
            if(UCIData){
                Instances all = ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
                data = InstanceTools.resampleInstances(all, fold, .5);            
            }else{
                train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
                test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
                data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
            }
            
            if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
            {
                checkpoint=false;
//Check if it already exists, if it does, exit
                f=new File(predictions+"/fold"+fold+"_"+parameterNum+".csv");
                if(f.exists() && f.length()>0){ //Exit
                    System.out.println("Fold "+predictions+"/fold"+fold+"_"+parameterNum+".csv  already exists");
                    return; //Aready done
                }
            }
            
            double acc = singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
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
            checkpoint=false;
            ((ParameterSplittable)c).setParametersFromIndex(parameterNum);
//            System.out.println("classifier paras =");
            testFoldPath="/fold"+fold+"_"+parameterNum+".csv";
            generateTrainFiles=false;
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
                if(c instanceof TrainAccuracyEstimate) //Classifier will perform cv internally
                    ((TrainAccuracyEstimate)c).writeCVTrainToFile(resultsPath+trainFoldPath);
                else{ // Need to cross validate here
                    int numFolds = Math.min(train.numInstances(), numCVFolds);
                    CrossValidator cv = new CrossValidator();
                    cv.setSeed(fold);
                    cv.setNumFolds(numFolds);
                    trainResults=cv.crossValidateWithStats(c,train);
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
                }
            }
            
            //Start of testing
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
            
            return testResults.acc;
        } catch(Exception e) {
            System.out.println(" Error ="+e+" in method simpleExperiment"+e);
            e.printStackTrace();
            System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
            System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

            return Double.NaN;
        }
    }    

    
}










//OLD EXPERIMENTAL CODE SAVED, JUST IN CASE, jamesl

///** Run a given classifier/problem/fold combination with associated file set up
// @param args: 
// * args[0]: Classifier name. Create classifier with setClassifier
// * args[1]: Problem name
// * args[2]: Fold number. This is assumed to range from 1, hence we subtract 1
// * (this is because of the scripting we use to run the code on the cluster)
// *          the standard archive folds are always fold 0
// * 
// * NOTES: 
// * 1. this assumes you have set DataSets.problemPath to be where ever the 
// * data is, and assumes the data is in its own directory with two files, 
// * args[1]_TRAIN.arff and args[1]_TEST.arff 
// * 2. assumes you have set DataSets.resultsPath to where you want the results to
// * go It will NOT overwrite any existing results (i.e. if a file of non zero 
// * size exists)
// * 3. This method just does the file set up then calls the next method. If you 
// * just want to run the problem, go to the next method
//* */
//    public static void singleClassifierAndFoldTrainTestSplit(String[] args) throws Exception{
////first gives the problem file      
//        String classifier=args[0];
//        String problem=args[1];
//        int fold=Integer.parseInt(args[2])-1;
//   
//        File f=new File(DataSets.resultsPath+classifier);
//        if(!f.exists())
//            f.mkdir();
//        String predictions=DataSets.resultsPath+classifier+"/Predictions";
//        f=new File(predictions);
//        if(!f.exists())
//            f.mkdir();
//        predictions=predictions+"/"+problem;
//        f=new File(predictions);
//        if(!f.exists())
//            f.mkdir();
//        
////Check whether fold already exists, if so, dont do it, just quit
//        f=new File(predictions+"/testFold"+fold+".csv");
//        if(!f.exists() || f.length()==0){
//            Classifier c=setClassifier(classifier,fold);
//            
////DO ALL THE SAMPLING HERE NOW        
//            Instances train,test;
//            Instances[] data;
//            String uciTest=DataSets.problemPath.toUpperCase();
//            if(uciTest.contains("UCI"))
//                UCIData=true;
//            if(UCIData){
//                Instances all = ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
//                data = InstanceTools.resampleInstances(all, fold, .5);            
//            }else{
//                train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
//                test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
//                data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
//
//            }
//            
//            if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
//            {
//                checkpoint=false;
////Check if it already exists, if it does, exit
//                f=new File(predictions+"/fold"+fold+"_"+parameterNum+".csv");
//                if(f.exists() && f.length()>0){ //Exit
//                    return; //Aready done
//                }
//            }
//            else{
//                if(generateTrainFiles){
//                    if(c instanceof TrainAccuracyEstimate){
//                        ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
//                    }
//                    else{ // Need to cross validate
//                        int numFolds=data[0].numInstances()>=10?10:data[0].numInstances();
//                        CrossValidator cv = new CrossValidator();
//                        cv.setSeed(fold);
//                        cv.setNumFolds(numFolds);
//            //Estimate train accuracy HERE
////Perform the split here                        
//                        ClassifierResults res=cv.crossValidateWithStats(c,data[0]);
//            //Write to file
//                        OutFile of=new OutFile(predictions+"/trainFold"+fold+".csv");
//                        of.writeLine(data[0].relationName()+","+c.getClass().getName()+",train");
//                        if(c instanceof SaveParameterInfo )
//                            of.writeLine(((SaveParameterInfo)c).getParameters());
//                        else
//                            of.writeLine("No Parameter Info");
//                        of.writeLine(res.acc+"");
//                        
//                        if(res.numInstances()>0){
//                            double[] trueClassVals=res.getTrueClassVals();
//                            double[] predClassVals=res.getPredClassVals();
//                            DecimalFormat df=new DecimalFormat("###.###");
//
//                            for(int i=0;i<data[0].numInstances();i++){
//                                //Basic sanity check
//                                if(data[0].instance(i).classValue()!=trueClassVals[i]){
//                                    throw new Exception("ERROR in TSF cross validation, class mismatch!");
//                                }
//                                of.writeString((int)trueClassVals[i]+","+(int)predClassVals[i]+",");
//                                double[] distForInst=res.getDistributionForInstance(i);
//                                for(double d:distForInst)
//                                    of.writeString(","+df.format(d));
//                                if(i<data[0].numInstances()-1)
//                                    of.writeString("\n");
//                            }
//                       }
//
//                    }
//                }
//            }
//            double acc =singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
//            System.out.println(classifier+","+problem+","+fold+","+acc);
//        }
//    }
///**
// * 
// * @param train: the standard train fold Instances from the archive 
// * @param test: the standard test fold Instances from the archive
// * @param c: Classifier to evaluate
// * @param fold: integer to indicate which fold. Set to 0 to just use train/test
// * @param resultsPath: a string indicating where to store the results
// * @return the accuracy of c on fold for problem given in train/test
// * 
// * NOTES:
// * 1.  If the classifier is a SaveableEnsemble, then we save the internal cross 
// * validation accuracy and the internal test predictions
// * 2. The output of the file testFold+fold+.csv is
// * Line 1: ProblemName,ClassifierName, train/test
// * Line 2: parameter information for final classifier, if it is available
// * Line 3: test accuracy
// * then each line is
// * Actual Class, Predicted Class, Class probabilities 
// * 
// * 
// */    
//    public static double singleClassifierAndFoldTrainTestSplit(Instances train, Instances test, Classifier c, int fold,String resultsPath){
//        double acc=0;
//        int act;
//        int pred;
//        String testFoldPath="/testFold"+fold+".csv";
//        if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
//        {
//            checkpoint=false;
//            ((ParameterSplittable)c).setParametersFromIndex(parameterNum);
////            System.out.println("classifier paras =");
//            testFoldPath="/fold"+fold+"_"+parameterNum+".csv";
//        }
//        else{
////Only do all this if not an internal fold
//    // Save internal info for ensembles
//            if(c instanceof SaveableEnsemble)
//               ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
//            if(checkpoint && c instanceof SaveEachParameter){     
//                ((SaveEachParameter) c).setPathToSaveParameters(resultsPath+"/fold"+fold+"_");
//            }
//        }
//        try{              
//            c.buildClassifier(train);
//            if(debug){
//                if(c instanceof RandomForest)
//                    System.out.println(" Number of features in MAIN="+((RandomForest)c).getNumFeatures());
//            }
//            StringBuilder str = new StringBuilder();
//            DecimalFormat df=new DecimalFormat("##.######");
//            for(int j=0;j<test.numInstances();j++)
//            {
//                act=(int)test.instance(j).classValue();
//                test.instance(j).setClassMissing();//Just in case ....
//                double[] probs=c.distributionForInstance(test.instance(j));
//                pred=0;
//                for(int i=1;i<probs.length;i++){
//                    if(probs[i]>probs[pred])
//                        pred=i;
//                }
//                if(act==pred)
//                    acc++;
//                str.append(act);
//                str.append(",");
//                str.append(pred);
//                str.append(",");
//                for(double d:probs){
//                    str.append(",");
//                    str.append(df.format(d));
//                }
//                if(j<test.numInstances()-1)
//                    str.append("\n");
//            }
//            acc/=test.numInstances();
//            OutFile p=new OutFile(resultsPath+testFoldPath);
//            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
//            if(c instanceof SaveParameterInfo){
//              p.writeLine(((SaveParameterInfo)c).getParameters());
//            }else
//                p.writeLine("No parameter info");
//            p.writeLine(acc+"");
//            p.writeString(str.toString());
//        }catch(Exception e)
//        {
//                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
//                e.printStackTrace();
//                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
//                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");
//
//                System.exit(0);
//        }
//         return acc;
//    }    
