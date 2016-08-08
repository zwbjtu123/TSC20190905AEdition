/*

1. Run HESCA on each channel individually

 */
package AALTD_Competition;

import classifiers.lazy.DTW_D;
import development.DataSets;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import new_COTE_experiments.RISE;
import tsc_algorithms.BOSSEnsemble;
import tsc_algorithms.ElasticEnsemble;
import tsc_algorithms.LearnShapelets;
import tsc_algorithms.ST_Ensemble;
import tsc_algorithms.TSF;
import utilities.ClassifierTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class RunExperiments {
        static String dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD";
        static String resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\DTW\\";
    
     public static Classifier setClassifier(String classifier){
        Classifier c=null;
        String s=classifier.toUpperCase();
        switch(classifier){
             case "ED": 
                 c=new IB1();
//                 c=new kNN(1);
                 break;
             case "C45": 
                 c=new J48();
//                 c=new kNN(1);
                 break;
             case "ROTF": case "ROTATIONFOREST":
                 c=new RotationForest();
                 ((RotationForest)c).setNumIterations(200);
                 break;
             case "RANDF":
                 c=new RandomForest();
                 ((RandomForest)c).setNumTrees(500);
                 break;
            case "DTW":
                c=new DTW_1NN();
                break;
            case "DTW_D":
                DTW_D dist= new DTW_D(3);
                c=new DTW_1NN(dist);
                break;
             case "HESCA": case "WE":
                c=new HESCA();
                 break;
             case "TSF":
                c=new TSF();
                ((TSF)c).setNumTrees(2000);
                 break;
             case "BOSS":
                c=new BOSSEnsemble();
                 break;
             case "RISE":
                c=new RISE();
                ((RISE)c).setNosBaseClassifiers(2000);
                 break;
             case "ST":
                c=new ST_Ensemble();
                long hour =60*60*1000000;
                ((ST_Ensemble)c).setTimeLimit(hour);
                 break;
             case "EE":
                c=new ElasticEnsemble();
                 break;
             case "LS":
                c=new LearnShapelets();
                ((LearnShapelets)c).setParamSearch(true);               
                 break;
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
//                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
   
    public static void concatenatedSeriesExperiment(){    
            Instances train=ClassifierTools.loadData(dataPath+"AALTDChallenge_TRAIN");
            OutFile of=new OutFile(resultsPath+"trainPredConcatV1.csv");

            RotationForest c=new RotationForest();
            c.setNumIterations(50);
            loocv(c,train,of);
    }

    public static void localExperiment(String algorithm) throws Exception{    
        System.out.println("Algorithm = :"+algorithm);
        NormalizeCase nc=new NormalizeCase();
        
        for(int i=0;i<8;i++){
            Instances train=ClassifierTools.loadData(dataPath+i+"_X_TRAIN");
            train=nc.process(train);
            Classifier c=setClassifier(algorithm);
            OutFile of=new OutFile(resultsPath+"trainPredX"+i+".csv");
            System.out.println("SENSOR"+i+" X ::");
            loocv(c,train,of);
            train=ClassifierTools.loadData(dataPath+i+"_Y_TRAIN");
            train=nc.process(train);
            System.out.println("SENSOR"+i+" Y ::");
            of=new OutFile(resultsPath+"trainPredY"+i+".csv");
            loocv(c,train,of);
            of=new OutFile(resultsPath+"trainPredZ"+i+".csv");
            train=ClassifierTools.loadData(dataPath+i+"_Z_TRAIN");
            train=nc.process(train);
            System.out.println("SENSOR"+i+" Z ::");
            loocv(c,train,of);
        
        }
    }
    public static void clusterExperimentAllSplit(String algorithm,int fold) throws Exception{    
        System.out.println(algorithm+" fold "+fold);
        NormalizeCase nc=new NormalizeCase();
        for(int i=0;i<8;i++){
            Instances  train=null;
            OutFile of;
            Classifier c=setClassifier(algorithm);
            String res=resultsPath+"trainPredX"+i+"_"+fold+".csv";
            File f=new File(res);
            if(!f.exists()||f.length()==0){
                train=ClassifierTools.loadData(dataPath+i+"_X_TRAIN");
                train=nc.process(train);
                of=new OutFile(res);
                System.out.println("CHANNEL"+i+" X ::");
                loocvSingleFold(c,train,of,fold);
            }
            res=resultsPath+"trainPredY"+i+"_"+fold+".csv";
            f=new File(res);
            if(!f.exists()||f.length()==0){
                train=ClassifierTools.loadData(dataPath+i+"_Y_TRAIN");
                train=nc.process(train);
                of=new OutFile(res);
                System.out.println("CHANNEL"+i+" Y ::");
                loocvSingleFold(c,train,of,fold);
            }
            res=resultsPath+"trainPredZ"+i+"_"+fold+".csv";
            f=new File(res);
            if(!f.exists()||f.length()==0){
                train=ClassifierTools.loadData(dataPath+i+"_Z_TRAIN");
                train=nc.process(train);
                of=new OutFile(res);
                System.out.println("CHANNEL"+i+" Z ::");
                loocvSingleFold(c,train,of,fold);
            }
        }
    }
    
    
    public static void clusterExperimentAllConcatenated(String algorithm,int fold) throws Exception{    
        System.out.println(algorithm+" fold "+fold);
        NormalizeCase nc=new NormalizeCase();
        Instances  train=null;
        OutFile of;
        Classifier c=setClassifier(algorithm);
        String res=resultsPath+"trainPredAll_"+fold+".csv";
        File f=new File(res);
        if(!f.exists()||f.length()==0){
            train=ClassifierTools.loadData(dataPath+"_TRAIN");
            train=nc.process(train);
            of=new OutFile(res);
            System.out.println("Fold X ::");
            loocvSingleFold(c,train,of,fold);
        }
    }
    
        public static void localExperimentBySensor(String algorithm) throws Exception{    
        System.out.println("Algorithm = :"+algorithm);
        for(int i=0;i<8;i++){
            Instances train=ClassifierTools.loadData(dataPath+"Sensor"+i+"_XYZ_TRAIN");
            Classifier c=setClassifier(algorithm);
            OutFile of=new OutFile(resultsPath+"trainPredXYZ"+i+".csv");
            System.out.println("SENSOR "+i);
            loocv(c,train,of);
        
        }
    }

   public static void localExperimentAllConacenated(String algorithm, Instances train) throws Exception{    
        System.out.println("Algorithm = :"+algorithm);
        Classifier c=setClassifier(algorithm);
        OutFile of=new OutFile(resultsPath+"trainPredAll.csv");
        loocv(c,train,of);
    }

   public static void trainTestExperimentAllConacenated(String algorithm, Instances train, Instances test) throws Exception{    
        System.out.println("Algorithm = :"+algorithm);
        Classifier c=setClassifier(algorithm);
        OutFile of=new OutFile(resultsPath+algorithm+"Test.csv");
        c.buildClassifier(train);
        System.out.println("Classifier built");
        for(Instance in:test){
            int pred=(int)c.classifyInstance(in);
            of.writeString("?,"+(pred+1)+",");
            double[] prob=c.distributionForInstance(in);
            for(double d:prob)
                of.writeString(","+d);
            of.writeString("\n");
        }
        
        
    }
        
    public static void main(String[] args) throws Exception {
        if(args.length>0){
            String algo=args[0];
            int fold=Integer.parseInt(args[1])-1;
            String problemFile=args[2];
            dataPath=DataSets.clusterPath+"TSC Problems/AALTDChallenge/"+problemFile;
            resultsPath=DataSets.clusterPath+"/Results/AALTDChallenge/"+algo+"/";
            File f= new File(resultsPath);
            if(!f.exists())
                f.mkdir();
            clusterExperimentAllConcatenated(algo,fold);
        }
        else{
            String algo="ROTF";
           dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_All_XY";
//            dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SummaryChannels\\AALTD_ALL_SS_TWO_CLASS";
            resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\All Concatenated\\COTEV1\\Temp\\";
            
            Instances train=ClassifierTools.loadData(dataPath+"_TRAIN");
            Instances test=ClassifierTools.loadData(dataPath+"_TEST");
            trainTestExperimentAllConacenated(algo,train,test);
            System.exit(0);

//            dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD";
//            dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\XYZBySensor\\AALTD";
          
            resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\SummaryStats\\"+algo+"\\";
            File f=new File(resultsPath);
            if(!f.exists())
                f.mkdir();
            dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\";

            Instances all=ClassifierTools.loadData(dataPath+"SummaryChannels\\AALTD_ALL_SS_THREE_CLASS_TRAIN");
            localExperimentAllConacenated(algo,all);
            
        }            
        
        
//      dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\";
//      resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\RotF\\";
//      concatenatedSeriesExperiment();
        
        
    }

    public static void loocv(Classifier c, Instances data, OutFile trainResults) {
        DecimalFormat df = new DecimalFormat("##.#######");

        Instances train = new Instances(data);
        int correct = 0;
        int numInstances = data.numInstances();
        int[] predictions = new int[numInstances];
        double[][] probs = new double[numInstances][];
        for (int i = 0; i < numInstances; i++) {
            if(i%2==0)
                System.out.println("Build fold "+i);
            Instance test = train.remove(i);
            try {
                c.buildClassifier(train);
                predictions[i] = (int) c.classifyInstance(test);
                probs[i] = c.distributionForInstance(test);
                if (predictions[i] == (int) test.classValue()) {
                    correct++;
                }
            } catch (Exception e) {
                System.err.println("ERROR BUILDING FOLD " + i + " for data set " + data.relationName());
                e.printStackTrace();
                System.exit(1);
            }
            train.add(i, test);
        }
        trainResults.writeLine(data.relationName() + "," + c.getClass().getName() + ",train");
        if (c instanceof SaveCVAccuracy) {
            trainResults.writeLine(((SaveCVAccuracy) c).getParameters());
        } else if (c instanceof SaveableEnsemble) {
            trainResults.writeLine(((SaveableEnsemble) c).getParameters());
        } else {
            trainResults.writeLine("NoParameterInfo");
        }
        trainResults.writeLine((correct / (double) numInstances) + "");
        System.out.println(" ACCURACY =" + (correct / (double) numInstances));
        for (int i = 0; i < numInstances; i++) {
            trainResults.writeString(((int) train.instance(i).classValue()) + "," + predictions[i] + ",");
            for (int j = 0; j < probs[i].length; j++) {
                trainResults.writeString("," + df.format(probs[i][j]));
            }
            trainResults.writeString("\n");
        }
    }

    public static void loocvSingleFold(Classifier c, Instances data, OutFile trainResults, int fold) {
        DecimalFormat df = new DecimalFormat("##.#######");
        Instances train = new Instances(data);
        int correct = 0;
        int prediction = 0;
        double[] probs;
        Instance test = train.remove(fold);
        try {
            c.buildClassifier(train);
            prediction = (int) c.classifyInstance(test);
            probs = c.distributionForInstance(test);
            if (prediction == (int) test.classValue()) {
                correct++;
            }
            train.add(fold, test);
            trainResults.writeLine(data.relationName() + "," + c.getClass().getName() + ",train");
            if (c instanceof SaveCVAccuracy) {
                trainResults.writeLine(((SaveCVAccuracy) c).getParameters());
            } else if (c instanceof SaveableEnsemble) {
                trainResults.writeLine(((SaveableEnsemble) c).getParameters());
            } else {
                trainResults.writeLine("NoParameterInfo");
            }
            trainResults.writeLine(correct + "");
            System.out.println(" ACCURACY =" + (correct));
            trainResults.writeString(((int) train.instance(fold).classValue()) + "," + prediction + ",");
            for (int j = 0; j < probs.length; j++) {
                trainResults.writeString("," + df.format(probs[j]));
            }
            trainResults.writeString("\n");
        } catch (Exception e) {
            System.err.println("ERROR BUILDING FOLD " + fold + " for data set " + data.relationName());
            e.printStackTrace();
            System.exit(1);
        }
    }
 }
