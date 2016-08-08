/*

 */
package new_COTE_experiments;

import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import utilities.SaveCVAccuracy;
import tsc_algorithms.*;
import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.*;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.OptimisedRotationForest;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.elastic_distance_measures.DTW;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.BagOfPatternsFilter;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class HIVE_COTE_Experiments{
    public static boolean subSample=false;
    public static double sampleProp=0.3;
    public static List<String> sampleProblems=Arrays.asList("HandOutlines","FordA","FordB","NonInvasiveFatalECGThorax1","NonInvasiveFatalECGThorax2");

    public static Classifier setClassifier(String classifier){
        Classifier c=null;
        switch(classifier){
            case "RIF_PS":
                c=new RISE();
                ((RISE)c).setTransformType("PS");
                break;
            case "RIF_ACF":
                c=new RISE();
                ((RISE)c).setTransformType("ACF");
                break;
            case "RIF_PS_ACF": 
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                
                break;
            case "RISE":
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                break;
            case "RISE_HESCA":
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                Classifier base=new HESCA();
                ((RISE)c).setBaseClassifier(base);
                break;
                
            case "FixedIntervalForest":
                c=new FixedIntervalForest();
                ((FixedIntervalForest)c).useCV(true);
                break;
            case "ACF":
                c=new PSACF_Ensemble();
                ((PSACF_Ensemble)c).setClassifierType("WE");
                break;
            case "PS_ACF":
                c=new PSACF_Ensemble();
                ((PSACF_Ensemble)c).setClassifierType("hesca");
                ((PSACF_Ensemble)c).setTransformType("PSACF");
                break;
            case "PS":
                c=new PS_Ensemble();
                ((PS_Ensemble)c).setClassifierType("WE");
                break;
                
             case "TSF": 
                c=new TSF();
                break;
             case "BOSS": case "BOSSEnsemble": 
                c=new BOSSEnsemble();
                break;
             case "WE": case "TimeWE":
                c=new HESCA();
                 break;
             case "RotF": case "RotationForest":
                 c=new RotationForest();
                 ((RotationForest)c).setNumIterations(50);
                 break;
            case "ST":
                c=new ST_Ensemble();
                break;
                 
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
//                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
    
 
   
    public static void testNewProblems(){
           String[] names={"ElectricDeviceOn","EthanolLevel","HeartbeatBIDMC"};

           for(String str: names){
               RotationForest rf= new RotationForest();
               rf.setNumIterations(50);
               DTW_1NN dtw=new DTW_1NN();

               Instances train=ClassifierTools.loadData(DataSets.dropboxPath+"TSC Problems/"+str+"/"+str+"_TRAIN");
               Instances test=ClassifierTools.loadData(DataSets.dropboxPath+"TSC Problems/"+str+"/"+str+"_TEST");
               double acc1=ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
               System.out.println(" Rotation Forest acc ="+acc1);
               double acc2=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
               System.out.println(" DTW acc ="+acc2);
           }
       }
    

/**
 * @param args: Classifier, ProblemName, Fold (1-100)
 */        
    public static void classifierProblemFold(String[] args){
  
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
//Set up file structure if necessary   
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
            Classifier c=setClassifier(classifier);
            Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
            singleSampleExperiment(train,test,c,fold,predictions);
        }
        else
            System.out.println(" Fold "+fold+" already complete");
    }
     public static void singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, sample);
        double acc=0;
        OutFile p=new OutFile(preds+"/testFold"+sample+".csv");
        double[] testLabels=new double[data[1].numInstances()];
        for(int i=0;i<testLabels.length;i++){
            testLabels[i]=data[1].instance(i).classValue();
            data[1].instance(i).setClassMissing();
        }

// Determine what needs to be saved
//Save the train CV accuracy and predictions        
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(preds+"/trainFold"+sample+".csv");        
//Save the internal predictions and accuracies of the components of an ensemble        
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
//Subsample the problem.
        if(subSample && c instanceof SubSampleTrain)
            ((SubSampleTrain)c).subSampleTrain(sampleProp,sample);
        
        try{              
            c.buildClassifier(data[0]);
            int[][] predictions=new int[data[1].numInstances()][2];
            for(int j=0;j<data[1].numInstances();j++)
            {
//HERE: Remove the class value to remove any possibility of using it in classification                
//                predictions[j][0]=(int)data[1].instance(j).classValue();
                predictions[j][1]=(int)c.classifyInstance(data[1].instance(j));
                if(testLabels[j]==predictions[j][1])
                    acc++;
            }
            acc/=data[1].numInstances();
            String[] names=preds.split("/");
            p.writeLine(names[names.length-1]+","+c.getClass().getName()+",test");
            if(c instanceof SaveCVAccuracy)
                p.writeLine(((SaveCVAccuracy)c).getParameters());
            else if(c instanceof SaveableEnsemble)
                p.writeLine(((SaveableEnsemble)c).getParameters());
            else
                p.writeLine("NoParameterInfo");
            p.writeLine(acc+"");
            for(int j=0;j<data[1].numInstances();j++){
                p.writeString((int)testLabels[j]+","+predictions[j][1]+",");
                double[] dist =c.distributionForInstance(data[1].instance(j));
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
    }

    
    public static void main(String[] args) throws Exception{

        try{
            if(args.length>0){ 
    //Cluster run: assume single fold format: Classifier, Problem, Fold
                for (int i = 0; i < args.length; i++) {
                    System.out.println("ARGS ="+i+" = "+args[i]);
                }
                DataSets.resultsPath=DataSets.clusterPath+"Results/";
                File f= new File(DataSets.resultsPath);
                if(args.length>1){
                    if(sampleProblems.contains(args[1])){
                        subSample=true;
                        sampleProp=0.2;
                        System.out.println("SUBSAMPLING "+args[1]+" IF THE CLASSIFIER "+args[0]+" ALLOWS IT");
                    }
                }
                if(!f.exists())
                    f.mkdir();
                DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
                classifierProblemFold(args);
            }
            else{       
    //Local run    
                DataSets.resultsPath=DataSets.dropboxPath+"NewCOTEResults/";
                DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
    //            System.exit(0);
                String problem="ItalyPowerDemand";
                String classifier="RIF_PS";

                if(sampleProblems.contains(problem)){
                    subSample=true;
                    sampleProp=0.05;
                    System.out.println("SUBSAMPLING "+problem+" IF THE CLASSIFIER "+classifier+" ALLOWS IT"+" PROP ="+0.05);
                }
                String[] ar={classifier,problem,"2"};
                classifierProblemFold(ar);
                }
            }catch(Exception e){
                System.out.println("Exception thrown ="+e);
                e.printStackTrace();
            }
        }

    }
