/**
 *
 * @author ajb
 *local class to run experiments with the UCI data


*/
package development;

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
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.FastShapelets;
import timeseriesweka.classifiers.FlatCote;
import timeseriesweka.classifiers.HiveCote;
import timeseriesweka.classifiers.LPS;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.NN_CID;
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
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;


public class RepoExperiments{
    public static int folds=30; 
    static boolean debug=true;
    static boolean generateTrainFiles=true;
    public static Classifier setClassifier(String classifier){
        Classifier c=null;
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
                break;
            case "SVMQ":
                c=new SMO();
                PolyKernel p2=new PolyKernel();
                p2.setExponent(2);
                ((SMO)c).setKernel(p2);
                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "RandF":
                c= new RandomForest();
                ((RandomForest)c).setNumTrees(500);
                break;
            case "RotF":
                c= new RotationForest();
                ((RotationForest)c).setNumIterations(200);
                break;
            case "TunedRandF":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                break;
            case "TunedRotF":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                break;
            case "TunedSVM":
                c= new TunedSVM();
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
                c=new DTW1NN();
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
   
        Classifier c=setClassifier(classifier);
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
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(generateTrainFiles){
                if(c instanceof TrainAccuracyEstimate){
                    ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
                }
                else{ // Need to cross validate
                    int numFolds=train.numInstances()>=10?10:train.numInstances();
                    CrossValidator cv = new CrossValidator();
                    cv.setSeed(fold);
                    cv.setNumFolds(numFolds);
        //Estimate train accuracy HERE
                    ClassifierResults res=cv.crossValidateWithStats(c,train);
        //Write to file
                    OutFile of=new OutFile(predictions+"/trainFold"+fold+".csv");
                    of.writeLine(train.relationName()+","+c.getClass().getName()+",train");
                    if(c instanceof SaveParameterInfo )
                        of.writeLine(((SaveParameterInfo)c).getParameters());
                    else
                        of.writeLine("No Parameter Info");
                   of.writeLine(res.acc+"");
                   if(res.numInstances()>0){
                    double[] trueClassVals=res.getTrueClassVals();
                    double[] predClassVals=res.getPredClassVals();
                    DecimalFormat df=new DecimalFormat("###.###");

                    for(int i=0;i<train.numInstances();i++){
                        //Basic sanity check
                        if(train.instance(i).classValue()!=trueClassVals[i]){
                            throw new Exception("ERROR in TSF cross validation, class mismatch!");
                        }
                        of.writeString((int)trueClassVals[i]+","+(int)predClassVals[i]+",");
                        double[] distForInst=res.getDistributionForInstance(i);
                        for(double d:distForInst)
                            of.writeString(","+df.format(d));
                        of.writeString("\n");
                    }
                   }
                    
                }
            }
            double acc =singleClassifierAndFoldTrainTestSplit(train,test,c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
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
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        try{              
            c.buildClassifier(data[0]);
            if(debug){
                if(c instanceof RandomForest)
                    System.out.println(" Number of features in MAIN="+((RandomForest)c).getNumFeatures());
            }
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=(int)data[1].instance(j).classValue();
                data[1].instance(j).setClassMissing();//Just in case ....
                double[] probs=c.distributionForInstance(data[1].instance(j));
                pred=0;
                for(int i=1;i<probs.length;i++){
                    if(probs[i]>probs[pred])
                        pred=i;
                }
                if(act==pred)
                    acc++;
                str.append(act);
                str.append(",");
                str.append(pred);
                str.append(",,");
                for(double d:probs){
                    str.append(df.format(d));
                    str.append(",");
                }
                str.append("\n");
            }
            acc/=data[1].numInstances();
            OutFile p=new OutFile(resultsPath+"/testFold"+fold+".csv");
            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo){
              p.writeLine(((SaveParameterInfo)c).getParameters());
            }else
                p.writeLine("No parameter info");
            p.writeLine(acc+"");
            p.writeLine(str.toString());
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


    
/* MUST BE FIVE ARGUMENTSArguments:
    1: Problem path args[0]
    2. Results path args[1]
    3. Classifier =args[2];
    4.    String problem=args[3];
    5.    int fold=Integer.parseInt(args[4])-1;
    */  
       
    public static void main(String[] args) throws Exception{
        if(args.length!=6){//Local run
            DataSets.problemPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
            DataSets.resultsPath="c:\\Temp\\";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            
            String[] newArgs={"SVMQ","SonyAIBORobotSurface1","2"};
            RepoExperiments.singleClassifierAndFoldTrainTestSplit(newArgs);
            System.exit(0);
        }
            
        DataSets.problemPath=args[0];
        DataSets.resultsPath=args[1];
//Third argument is whether to cross validate or not
        generateTrainFiles=Boolean.parseBoolean(args[2]);
        File f=new File(DataSets.resultsPath);
        if(!f.isDirectory()){
            f.mkdirs();
        }
        String[] newArgs=new String[args.length-3];
        for(int i=3;i<args.length;i++)
            newArgs[i-2]=args[i];
        RepoExperiments.singleClassifierAndFoldTrainTestSplit(newArgs);
        
    
    }


}

