/*
 Exploratory analysis of classifier performance on the IFR 
Alcohol data

Data Path

*/

package applications;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import new_COTE_experiments.RISE;
import tsc_algorithms.PSACF_Ensemble;
import tsc_algorithms.BOSSEnsemble;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.lazy.kNN;
import tsc_algorithms.COTE;
import tsc_algorithms.ElasticEnsemble;
import tsc_algorithms.PS_Ensemble;
import tsc_algorithms.ST_Ensemble;
import tsc_algorithms.SubSampleTrain;
import tsc_algorithms.TSF;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.SummaryStats;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 *
 * @author ajb
 */
public class Booze {
      static int nosDistilleries=46;
  
public static void generateHeader(){
    OutFile f = new OutFile("C:\\Data\\IFR Spirits\\Header.csv");
    f.writeString("@attribute distillery {");
    f.writeLine("aberfeldy,aberlour,allmalt,amrut,ancnoc,armorik,arran10,arran14,asyla,auchentoshan,balblair,benromach,bernheim,bladnoch,blairathol,cardhu,dutchsinglemalt,elijahcraig,englishwhisky15,englishwhisky9,exhibition,finlaggan,glencadam,glendeveron,glenfarclas,glenfiddich,glengoyne,glenlivet12,glenlivet15,glenmorangie,glenmoray,glenscotia,greatkingst,highlandpark,laphroaig,mackmyra,nikka,oakcross,organic,peatmonster,scapa,smokehead,speyburn,spicetree,talisker,tyrconnell}");
    
    for(int i=0;i<1748;i++){
        f.writeLine("@attribute wavelength"+(226.0+(i+1)/2.0)+" numeric");
    }
    f.writeLine("@attribute abv {40,44,50,55,60}");
    
}  


public static void transformKateStyle(){
    Instances all = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\AllSamples");
//1: Restrict to a region 890-1050

//Baseline shift
    
//Standardise area under the curve    
}
static double splitProp=0.7;

/** this just does a random split, irrespective of distillery. Seems likely
 * that instance based classifiers will do well with it!
 * @param all
 * @param rep
 * @throws Exception 
 */
public static double resampleExperimentBasic(Instances all,int rep,String resultPath, Classifier c) throws Exception{
    all.randomize(new Random());
    Instances train = new Instances(all);
    Instances test= new Instances(all,0);
    int testSize=(int)(all.numInstances()*splitProp);
    for(int i=0;i<testSize;i++)
        test.add(train.remove(0));
    System.out.println(" Train size ="+train.numInstances()+" test size ="+test.numInstances());
    c.buildClassifier(train);
    
 //   double[] cvAcc=c.getCVAccs();
    double acc=ClassifierTools.accuracy(test, c);
    System.out.println("acc ="+acc);
    return acc;
//    of.writeString(rep+","+acc+",");
//    for(int i=0;i<cvAcc.length;i++)
//        of.writeString(cvAcc[i]+",");
    
}

public static void classifyBySummaryStats() throws Exception{
    String resultPath="C:\\Users\\ajb\\Dropbox\\Results\\IFR Spirits\\SummaryStats\\";
    String sourcePath="C:\\Users\\ajb\\Dropbox\\IFR Spirits\\";
    Instances all=ClassifierTools.loadData(sourcePath+"AllSamplesFiveClass");
    
    int nosBottles=46;
    int nosPerBottle=all.numClasses()*4;
    OutFile of=new OutFile(resultPath+"boozeSummaryStatsFiveClassLimited.csv");
//    System.out.println(" All Size ="+all.numInstances()+" num attributes ="+all.numAttributes());
    all.sort(0);
    for(int i=0;i<1300;i++)
        all.deleteAttributeAt(1);
    for(int i=0;i<200;i++)
        all.deleteAttributeAt(all.numAttributes()-2);
    ArrayList<String> names =new ArrayList<>();
    Classifier[] c=ClassifierTools.setDefaultSingleClassifiers(names);
    of.writeString(",");
    for(String s:names)
        of.writeString(s+",");
    of.writeString("\n");
    
    for(int rep=0;rep<nosBottles;rep++){
        Instances train = new Instances(all);
        Instances test= new Instances(all,0);
        for(int i=0;i<nosPerBottle;i++)
            test.add(train.remove(nosPerBottle*rep));
        for(int i=1;i<nosPerBottle;i++)
            if(test.instance(i).value(0)!=test.instance(i).value(0))
                throw new Exception("INCORRECT SPLIT FOR "+test.instance(i).value(0));
        if(rep==0){
                System.out.println(" Test Split ="+test.instance(0).stringValue(0));
                double[] d=test.instance(0).toDoubleArray();
                for(int i=0;i<10;i++)
                    System.out.print(d[i]+",");
                System.out.print("\n");
                    
        }
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);
        SummaryStats ss = new SummaryStats();   
        Instances trainStats=ss.process(train);
        Instances testStats=ss.process(test);
        if(rep==0){
                System.out.println(" Summary Stats");
                double[] d=testStats.instance(0).toDoubleArray();
                for(int i=0;i<d.length;i++)
                    System.out.print(d[i]+",");
                System.out.print("\n");
                    
        }
        if(rep==0){
            OutFile tr=new OutFile("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\Debug\\SummaryStats_TRAIN.arff");
            tr.writeString(trainStats+"");
            tr=new OutFile("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\Debug\\SummaryStats_TEST.arff");
            tr.writeString(testStats+"");
            
        }
        names =new ArrayList<>();
        c=ClassifierTools.setDefaultSingleClassifiers(names);
  //      HESCA we = new HESCA();
        of.writeString(rep+",");
        System.out.println(" Rep ="+rep);
        for(Classifier cl:c){
            double a=ClassifierTools.singleTrainTestSplitAccuracy(cl, trainStats, testStats);
            of.writeString(a+",");
        }
//        double a=ClassifierTools.singleTrainTestSplitAccuracy(we, trainStats, testStats);
        
        of.writeString("\n");
    }
    
}




public static void classifyOnNormalizedRange(Instances all) throws Exception{
    String resultPath="C:\\Users\\ajb\\Dropbox\\Results\\IFR Spirits\\GlobalShape\\";
    String sourcePath="C:\\Users\\ajb\\Dropbox\\IFR Spirits\\";
//    =ClassifierTools.loadData(sourcePath+"AllSamplesFiveClass");
    
    int nosBottles=44;
    int nosPerBottle=all.numClasses()*4;
    OutFile of=new OutFile(resultPath+"globalShapeFiveSampleDTW.csv");
//    System.out.println(" All Size ="+all.numInstances()+" num attributes ="+all.numAttributes());
    ArrayList<String> names =new ArrayList<>();
/*    Classifier[] c=ClassifierTools.setDefaultSingleClassifiers(names);
    of.writeString(",");
    for(String s:names)
        of.writeString(s+",");
*/        

    
    of.writeString(",DTW\n");
    all.sort(0);
    for(int i=0;i<1300;i++)
        all.deleteAttributeAt(1);
    NormalizeCase nc = new NormalizeCase();
    all=nc.process(all);
    for(int rep=0;rep<nosBottles;rep++){
        kNN knn=new kNN();
        knn.setDistanceFunction(new BasicDTW());
        knn.normalise(false);
        Instances train = new Instances(all);
        Instances test= new Instances(all,0);
        for(int i=0;i<nosPerBottle;i++)
            test.add(train.remove(nosPerBottle*rep));
        for(int i=1;i<nosPerBottle;i++)
            if(test.instance(i).value(0)!=test.instance(i).value(0))
                throw new Exception("INCORRECT SPLIT FOR "+test.instance(i).value(0));
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);

  //      HESCA we = new HESCA();
//        names =new ArrayList<>();
//        c=ClassifierTools.setDefaultSingleClassifiers(names);
  //      HESCA we = new HESCA();
        of.writeString(rep+",");
        System.out.println(" Rep ="+rep);
  //      for(Classifier cl:c){
            double a=ClassifierTools.singleTrainTestSplitAccuracy(knn, train, test);
            of.writeString(a+",");
  //      }
        
        of.writeString("\n");
    }
    
}



public static Instances tidyUp(Instances all) throws Exception{
//Remove irrelevant features     
            //1: Distillery name
 //           all.deleteAttributeAt(0);
            //2: Restrict to a region 890-1050: Attributes
            System.out.println(" Number of attributes ="+all.numAttributes());
            System.out.println(" Number of instances ="+all.numInstances());
            System.out.println(" Number of classes ="+all.numClasses());
            System.out.println(" Class index ="+all.classIndex());
            int start=1300;
            for(int i=0;i<start;i++)
                all.deleteAttributeAt(1);
//Normalise
            Attribute att=all.attribute(0);
            if(att.isNominal())
                System.out.println("Att "+att+" is Nominal ");
            NormalizeCase nc = new NormalizeCase();
            Instances newInst=nc.process(all);
            System.out.println(" Number of attributes ="+all.numAttributes());
            System.out.println(" Number of instances ="+all.numInstances());
            System.out.println(" Number of classes ="+all.numClasses());
            System.out.println(" Class index ="+all.classIndex());
//Save to file
            return newInst;
            
         }

//Default to cluster 
    static String sourcePath="/gpfs/home/ajb/TSC Problems/EthanolLevel/EthanolLevel";
    static String resultsPath="/gpfs/home/ajb/EthanolLevelResults/";


public static void singleFold(String[] args) throws Exception{
    int rep=Integer.parseInt(args[1])-1;
    Instances all=ClassifierTools.loadData(sourcePath);
    EthanolFoldCreator eth=new EthanolFoldCreator();
    eth.deleteFirstAtt(true);
    Instances[] split=eth.createSplit(all, rep);    
    Classifier c= getClassifier(args[0]);
    File f=new File(resultsPath+args[0]);
    if(!f.exists())
        f.mkdir();
    singleSampleExperiment(split[0],split[1],c,rep,resultsPath+args[0]);    
    
//    ST_Ensemble st = new ST_Ensemble();
//    st.createTransformData(split[0],ShapeletTransformFactory.dayNano);//TIME
    
//Save full shapelet set for each fold.     

//Build classifier    
//    st.doSTransform(false);
//    st.buildClassifier(split[0]);
//Save output in the correct format    
 //   OutFile of= new OutFile(resultsPath+args[0]+rep+".csv");
//    
    
}

public static void mergeBoozeFiles(String path,int nosBottles){
    OutFile of = new OutFile(path+"combinedBooze.csv");
    for(int i=0;i<nosBottles;i++){
        InFile f =new InFile(path+"booze"+i+".csv");
        of.writeLine(f.readLine());   }
}
public static void firstExperiment(){
//   generateHeader();
    Instances all = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\TwoClassSpirits");
    J48 rf = new J48();
    all.deleteAttributeAt(0);
    double[][] acc;    
//    double[][] acc=ClassifierTools.crossValidationWithStats(rf, all, 10);
//   System.out.println(" C4.5 10 fold acc ="+acc[0][0]);
//Normalise cases
    NormalizeCase nc= new NormalizeCase();
    try {
 //       all=nc.process(all);
    } catch (Exception ex) {
        Logger.getLogger(Booze.class.getName()).log(Level.SEVERE, null, ex);
    }
    kNN k=new kNN(1);
    k.setCrossValidate(false);
    k.normalise(true);
    acc=ClassifierTools.crossValidationWithStats(k, all, 10);
   System.out.println(" 1-NN 10 fold acc ="+acc[0][0]);
    k=new kNN(100);
    k.setCrossValidate(true);
    k.normalise(true);
    acc=ClassifierTools.crossValidationWithStats(k, all, 10);
   System.out.println(" kNN 10 fold acc ="+acc[0][0]);
//    acc=ClassifierTools.crossValidationWithStats(new RandomForest(), all, 10);
//   System.out.println(" Rand forest 10 fold acc ="+acc[0][0]);
          
   } 


public static void main(String[] args) throws Exception{
    if(args.length<1){
        
        sourcePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\EthanolLevel\\EthanolLevel";
        resultsPath="C:\\Users\\ajb\\Dropbox\\NewCOTEResults\\EthanolLevel\\";
        
        String[] ar={"RIF_ACF","1"};
        singleFold(ar);
    }
    else
        singleFold(args);
}
     public static void singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        double acc=0;
        OutFile p=new OutFile(preds+"/testFold"+sample+".csv");

// Determine what needs to be saved
//Save the train CV accuracy and predictions        
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(preds+"/trainFold"+sample+".csv");        
//Save the internal predictions and accuracies of the components of an ensemble        
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");

        if(c instanceof ST_Ensemble)
            ((ST_Ensemble)c).setShapeletOutputFile(preds+"/ShapeletSetFold"+sample+".csv");

//Subsample the problem.
//        if(subSample && c instanceof SubSampleTrain)
  //          ((SubSampleTrain)c).subSampleTrain(sampleProp,sample);
        
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
    }
        
    

public static Classifier getClassifier(String str){
    Classifier c;
    switch(str){
        case "ST": case "ShapeletTransform":
            ST_Ensemble st = new ST_Ensemble();
            st.doSTransform(true);
            st.setTimeLimit(ShapeletTransformFactory.dayNano);
            c=st;   
            break;
        case "BOSS":
            c=new BOSSEnsemble();
            break;
        case "TSF":
            c=new TSF();
            break;
            case "RIF_PS":
                c=new RISE();
                ((RISE)c).setTransformType("PS");
                break;
            case "RIF_ACF":
                c=new RISE();
                ((RISE)c).setTransformType("ACF");
                break;
            case "ACF":
                c=new PSACF_Ensemble();
                ((PSACF_Ensemble)c).setClassifierType("WE");
                break;
            case "PS":
                c=new PS_Ensemble();
                ((PS_Ensemble)c).setClassifierType("WE");
                break;
        default:
            c=new DTW_1NN();
    }
    return c;
  }
 
}
