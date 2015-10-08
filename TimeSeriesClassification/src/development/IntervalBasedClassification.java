/*
Interval based DTW

Debug: 1: Make sure it isnot some data set corruption. Use full interval same interv
 */
package development;

import fileIO.*;
import java.io.File;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.*;
import weka.core.elastic_distance_measures.DTW;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class IntervalBasedClassification{
    
//1. Train test full enumeration for a classifier with standard enumeration order
    public static void intervalClassifier(Instances train, Instances test, Classifier c, OutFile results){
        int start=0;
        int end=train.numAttributes()-1;
        int minInterval=3;
        int bestStart=0;
        int bestEnd=end;
        double bestAcc=0;
        double bestSD=0;
//        double sd=train.numInstances()
        System.out.println(" Number of attributes (inc class)"+train.numAttributes());
//Eval full interval first. 
        double[][] a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        bestAcc=a[0][0];  
        double fullTrainAcc=bestAcc;
        System.out.println(" FULL INTERVAL (0,"+(train.numAttributes()-2)+"} acc = "+bestAcc);
        if(bestAcc<1){ //No point continuing if full interval is 1!
            for(int i=start;i<end-minInterval;i++){
                Instances intervalTrain=new Instances(train);
                for(int k=0;k<i;k++)   //Delete attributes before start
                    intervalTrain.deleteAttributeAt(0);
                for(int j=end-1;j>=i+minInterval;j--){  //Evaluate interval from i to j
                    intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
                    double[][] temp =ClassifierTools.crossValidationWithStats(c, intervalTrain,intervalTrain.numInstances());
    //Only accept the new interval if it is significantly better.
                    if(temp[0][0]>bestAcc){ 
    //Test is not really appropriate because not normal and not independent!                    
                        if(classifierTest(temp,a,TestType.ZTEST))
                        {
                            bestAcc=temp[0][0];
                            bestStart=i;
                            bestEnd=j;
                            bestSD=temp[1][0];
                            a=temp;
                            System.out.println(" Best interval so far: ("+bestStart+" "+bestEnd+")"+" has train acc ="+bestAcc);
                        }
                    }
                } 
            }
        }
//Eval on test
//Evaluate full interval test and reduced interval test
        Instances intervalTrain=new Instances(train);
        Instances intervalTest=new Instances(test);
        for(int k=0;k<bestStart;k++){   //Delete attributes before start
            intervalTrain.deleteAttributeAt(0);
            intervalTest.deleteAttributeAt(0);
        }
        for(int k=bestEnd+1;k<end;k++){   //Delete attributes after end
            intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
            intervalTest.deleteAttributeAt(intervalTrain.numAttributes()-2);
        }
//        System.out.println(" best start ="+bestStart+" best end ="+bestEnd+" full end ="+end);
//        System.out.println(" Train data size ="+intervalTrain.numAttributes());
//        System.out.println(" Test data size ="+intervalTrain.numAttributes());
        double fullIntervalTestAcc=ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
        double testAcc=ClassifierTools.singleTrainTestSplitAccuracy(c, intervalTrain, intervalTest);
        a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        System.out.println(" Full data CV accuracy ="+a[0][0]);
        results.writeLine(bestStart+","+bestEnd+","+fullTrainAcc+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
        System.out.println(bestStart+","+bestEnd+","+fullTrainAcc+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
    }
    

   public static void intervalClassifierAlternativeOrder(Instances train, Instances test, Classifier c, OutFile results){
        int start=0;
        int end=train.numAttributes()-1;
        int minInterval=3;
        int bestStart=0;
        int bestEnd=end;
        double bestAcc=0;
        double bestSD=0;
//        double sd=train.numInstances()
        System.out.println(" Number of attributes (inc class)"+train.numAttributes());
//Eval full interval first. 
        double[][] a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        bestAcc=a[0][0];  
        double fullTrainAcc=bestAcc;
        System.out.println(" FULL INTERVAL (0,"+(train.numAttributes()-2)+"} acc = "+bestAcc);
        if(bestAcc<1){ //No point continuing!
            for(int j=end-1;j>=minInterval;j--){ //Evaluate all j length intervals
                   System.out.println("\n  Eval all "+j+" length intervals");
//There are end-i intervals, all will start from 0, and end at                 
                for(int i=0;i<end-j;i++){  //Evaluate interval from i to j
  //                  System.out.print("  ("+i+" "+(j+i-1)+")");
//Evaluate full interval test and reduced interval test: will slow things down a lot!
                    Instances intervalTrain=new Instances(train);
                    for(int k=0;k<i;k++){   //Delete attributes before start
                        intervalTrain.deleteAttributeAt(0);
                    }
                    for(int k=j+i;k<end;k++){   //Delete attributes after end
                        intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
                    }
                    System.out.print(" "+(intervalTrain.numAttributes()-1));
                    double[][] temp =ClassifierTools.crossValidationWithStats(c, intervalTrain,intervalTrain.numInstances());
                    //                     System.out.println(" interval ("+i+" "+j+")"+" has train acc ="+a[0][0]);
                    //                  System.out.println("\t Number of attributes ="+(intervalTrain.numAttributes()-1)+" acc ="+a[0][0]);

                    //Only accept the new interval if it is significantly better.
                    if(temp[0][0]>bestAcc){ 
                    //Test is not really appropriate because not normal and not independent!                    
                    if(classifierTest(temp,a,TestType.ZTEST))
                    {
                        bestAcc=temp[0][0];
                        bestStart=i;
                        bestEnd=j;
                        bestSD=temp[1][0];
                        a=temp;
                        System.out.println(" Best interval so far: ("+bestStart+" "+bestEnd+")"+" has train acc ="+bestAcc);
                    }
                    }                                    
                } 
            }
        }
//Eval on test
//Evaluate full interval test and reduced interval test
        Instances intervalTrain=new Instances(train);
        Instances intervalTest=new Instances(test);
        for(int k=0;k<bestStart;k++){   //Delete attributes before start
            intervalTrain.deleteAttributeAt(0);
            intervalTest.deleteAttributeAt(0);
        }
        for(int k=bestEnd;k<end;k++){   //Delete attributes after end
            intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
            intervalTest.deleteAttributeAt(intervalTrain.numAttributes()-2);
        }
//        System.out.println(" best start ="+bestStart+" best end ="+bestEnd+" full end ="+end);
//        System.out.println(" Train data size ="+intervalTrain.numAttributes());
//        System.out.println(" Test data size ="+intervalTrain.numAttributes());
        double fullIntervalTestAcc=ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
        double testAcc=ClassifierTools.singleTrainTestSplitAccuracy(c, intervalTrain, intervalTest);
        a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        System.out.println(" Full data CV accuracy ="+a[0][0]);
        results.writeLine(bestStart+","+bestEnd+","+fullTrainAcc+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
        System.out.println(bestStart+","+bestEnd+","+fullTrainAcc+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
    }
    
    
    public static void allSpectral(){
        OutFile results= new OutFile(DataSets.resultsPath+"IntervalBasedDTW\\EDTrainTest.csv");
//        for(String s:DataSets.spectral){
            String s="Coffee";{
            results.writeString(s+",");
            System.out.println(s+",");

            Instances train=ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TEST");
            Classifier kNN=new kNN();
            NormalizeCase nc=new NormalizeCase();
            try {
                train=nc.process(train);
                test=nc.process(test);
                intervalClassifierAlternativeOrder(train,test,kNN,results);
            } catch (Exception ex) {
                Logger.getLogger(IntervalBasedClassification.class.getName()).log(Level.SEVERE, null, ex);
            }
       }
        
    }
//For cluster run
//Arg 1: array fold number
//Arg 2: file name
//Arg 3: distance measure
    public static void clusterRun(String[] args){
//
        int fold=Integer.parseInt(args[0]);
//Second gives the problem file      
        String s=args[1];
//third gives the distance function: 0=ED, 1=DTW, 2= DTWCV  
        int distanceMeasure=Integer.parseInt(args[2]);
        String dist="ED";
        if(distanceMeasure==1)
            dist="DTW";
        else if(distanceMeasure==2)
            dist="DTWCV";
        System.out.println(s+" "+dist+" "+fold);
        File f= new File(DataSets.clusterPath+"Results/Interval/"+s);
        if(!f.isDirectory())
            f.mkdir();
        f= new File(DataSets.clusterPath+"Results/Interval/"+s+"/"+dist+VERSION);
        if(!f.isDirectory())
            f.mkdir();
        
        OutFile results= new OutFile(DataSets.clusterPath+"Results/Interval/"+s+"/"+dist+VERSION+"/"+s+"_"+dist+"_"+fold+".csv");
        results.writeString(s+",");
        Instances train=ClassifierTools.loadData(DataSets.clusterPath+"TSC Problems/"+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.clusterPath+"TSC Problems/"+s+"/"+s+"_TEST");
//Form randomised stratified split
        if(fold!=1){
            Random r = new Random(fold);
            Instances all = new Instances(train);
            all.addAll(test);
            Map<Double, Integer> trainDistribution = InstanceTools.createClassDistributions(train);
            Map<Double, Instances> classBins = InstanceTools.createClassInstancesMap(all);
            //empty instances.
            Instances tr = new Instances(all, 0);
            Instances te = new Instances(all, 0);
             Iterator<Double> keys = classBins.keySet().iterator();
             while(keys.hasNext())
             {
                 double classVal = keys.next();
                 int occurences = trainDistribution.get(classVal);
                 Instances bin = classBins.get(classVal);
                 bin.randomize(r); //randomise the bin.
                 
                 tr.addAll(bin.subList(0,occurences));//copy the first portion of the bin into the train set
                 te.addAll(bin.subList(occurences, bin.size()));//copy the remaining portion of the bin into the test set.
             }
            train=tr;
            test=te;
        }
        
        Classifier knn=new kNN();
        if(distanceMeasure==1)
            ((kNN)knn).setDistanceFunction(new DTW());
        if(distanceMeasure==2)
            knn = new DTW_1NN();
        NormalizeCase nc=new NormalizeCase();
        try {
            train=nc.process(train);
            test=nc.process(test);
            intervalClassifierAlternativeOrder(train,test,knn,results);
        } catch (Exception ex) {
            System.out.println(" Error : "+ex);
        }
       
        
    }


    public static void debugRun(){
//First arg gives the dataset number: 1 to 82
        int fold=2;
//Second gives the problem file      
        String s="Coffee";
//third gives the distance function: 0=ED, 1=DTW, 2= DTWCV  
        int distanceMeasure=0;
        String dist="ED";
        if(distanceMeasure==1)
            dist="DTW";
        else if(distanceMeasure==2)
            dist="DTWCV";
        System.out.println(s+" "+dist+" "+fold);
        String str="C:/Users/ajb/Dropbox/";

        File f= new File(str+"Results/IntervalBasedDTW/Test/"+s);
        if(!f.isDirectory())
            f.mkdir();
        f= new File(str+"Results/IntervalBasedDTW/Test/"+s+"/"+dist+VERSION);
        if(!f.isDirectory())
            f.mkdir();
        
        OutFile results= new OutFile(str+"Results/IntervalBasedDTW/Test/"+s+"/"+dist+VERSION+"/"+s+"_"+dist+"_"+fold+".csv");
        results.writeString(s+",");
        Instances train=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(str+"TSC Problems/"+s+"/"+s+"_TEST");
//Form randomised stratified split
        for(fold=1;fold<10;fold++){
            Random r = new Random(fold);
            Instances all = new Instances(train);
            all.addAll(test);
            Map<Double, Integer> trainDistribution = InstanceTools.createClassDistributions(train);
            Map<Double, Instances> classBins = InstanceTools.createClassInstancesMap(all);
            //empty instances.
            Instances tr = new Instances(all, 0);
            Instances te = new Instances(all, 0);
             Iterator<Double> keys = classBins.keySet().iterator();
             while(keys.hasNext())
             {
                 double classVal = keys.next();
                 int occurences = trainDistribution.get(classVal);
                 Instances bin = classBins.get(classVal);
                 bin.randomize(r); //randomise the bin.
                 
                 tr.addAll(bin.subList(0,occurences));//copy the first portion of the bin into the train set
                 te.addAll(bin.subList(occurences, bin.size()));//copy the remaining portion of the bin into the test set.
             }
            train=tr;
            test=te;
            Classifier knn=new kNN();
            if(distanceMeasure==1)
                ((kNN)knn).setDistanceFunction(new DTW());
            if(distanceMeasure==2)
                knn = new DTW_1NN();
            NormalizeCase nc=new NormalizeCase();
            try {
                train=nc.process(train);
                test=nc.process(test);
                intervalClassifierDebugRun(train,test,knn,results);
            } catch (Exception ex) {
                System.out.println(" Error : "+ex);
            }
        }
   
    }


public static void intervalClassifierDebugRun(Instances train, Instances test, Classifier c, OutFile results){
         int start=0;
       int end=train.numAttributes()-1;
        int minInterval=3;
        int bestStart=0;
        int bestEnd=end;
        double bestAcc=0;
//        double sd=train.numInstances()
        System.out.println(" Number of attributes (inc class)"+train.numAttributes());
//Eval full interval first. 
        double[][] a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        bestAcc=a[0][0];  
        double fullTrainAcc=bestAcc;
        System.out.println(" FULL INTERVAL (0,"+(train.numAttributes()-2)+"} acc = "+bestAcc);
        if(bestAcc<1){ //No point continuing!
            for(int i=start;i<end-minInterval;i++){
                Instances intervalTrain=new Instances(train);
                for(int k=0;k<i;k++)   //Delete attributes before start
                    intervalTrain.deleteAttributeAt(0);
                for(int j=end-1;j>=i+minInterval;j--){  //Evaluate interval from i to j
                    intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
                    double[][] temp=a;
//                    double[][] temp =ClassifierTools.crossValidationWithStats(c, intervalTrain,intervalTrain.numInstances());
//                    System.out.println(" interval ("+i+" "+j+")"+" has train acc ="+a[0][0]);
 //                   System.out.println("\t Number of attributes ="+(intervalTrain.numAttributes()-1)+" acc ="+a[0][0]);

                    //Only accept the new interval if it is significantly better.
                    if(temp[0][0]>bestAcc){ 
                    //Test is not really appropriate because not normal and not independent!                    
                    if(classifierTest(temp,a,TestType.ZTEST))
                    {
                        bestAcc=temp[0][0];
                        bestStart=i;
                        bestEnd=j;
                        a=temp;
                        System.out.println(" Best interval so far: ("+bestStart+" "+bestEnd+")"+" has train acc ="+bestAcc);
                    }
                    }                                    
                } 
            }
        }
//Eval on test
//Evaluate full interval test and reduced interval test
        Instances intervalTrain=new Instances(train);
        Instances intervalTest=new Instances(test);
        for(int k=0;k<bestStart;k++){   //Delete attributes before start
            intervalTrain.deleteAttributeAt(0);
            intervalTest.deleteAttributeAt(0);
        }
        for(int k=bestEnd;k<end;k++){   //Delete attributes after end
            intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
            intervalTest.deleteAttributeAt(intervalTrain.numAttributes()-2);
        }
//        System.out.println(" best start ="+bestStart+" best end ="+bestEnd+" full end ="+end);
//        System.out.println(" Train data size ="+intervalTrain.numAttributes());
//        System.out.println(" Test data size ="+intervalTrain.numAttributes());
        double fullIntervalTestAcc=ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
        double testAcc=ClassifierTools.singleTrainTestSplitAccuracy(c, intervalTrain, intervalTest);
        a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        System.out.println(" Full data CV accuracy ="+a[0][0]);
        results.writeLine(bestStart+","+bestEnd+","+fullTrainAcc+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
        System.out.println(bestStart+","+bestEnd+","+fullTrainAcc+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
    }

    
    enum TestType {BEST,ZTEST,MCNEMAR};
    static boolean classifierTest(double[][]a, double[][] b, TestType t){
        switch(t){
            case BEST:
                if(a[0][0]>b[0][0]) return true;
                return false;
            case ZTEST:
                double p=(a[0][0]+b[0][0])/2; //Pooled proportion
                double se=Math.sqrt(p*(1-p)/(2*(a[0].length-1)));     //Standard error
                double z=(a[0][0]-b[0][0])/se; 
                if(z>1.96) return true;
                return false;
            case MCNEMAR:
                throw new UnsupportedOperationException("MCNEMAR Not supported yet."); //To change body of generated methods, choose Tools | Templates.
            default:    
                throw new UnsupportedOperationException("Unknown test."); //To change body of generated methods, choose Tools | Templates.
        }
    }
    public static void combineResults(String path,String[] sourceFiles, String type,int reps){
        
       OutFile fullResults=new OutFile(path+type+VERSION+".csv");
        for(String s:sourceFiles){
//Chck it exists
            File inf=new File(path+"/"+s+"/"+type+VERSION);
            if(inf.exists()){//Collate the files
                OutFile results=new OutFile(path+s+type+VERSION+".csv");
                combineFiles(path+"/"+s+"/"+type+VERSION+"/"+s+"_"+type,results,reps);
                results.closeFile();
                InFile f = new InFile(path+s+type+".csv");
                int r=f.countLines();
                f.closeFile();
                f = new InFile(path+s+type+".csv");
    //Read in the accuracies, find mean and SD of difference             
                String name="";
                int start=0,end=0;
                double intervalTrainAcc=0;
                double sd=0;
                double fullTestAcc=0;
                double fullTrainAcc=0;
                double fullTestSS=0;
                double intervalTestAcc=0;
                double intervalTestSS=0;
                double[] diff=new double[r];
                for(int j=0;j<r;j++){
                    name=f.readString();
                    start+=f.readInt();
                    end+=f.readInt();
                    fullTrainAcc+=f.readDouble();
                    intervalTrainAcc+=f.readDouble();
                    double a=f.readDouble();
                    fullTestAcc+=a;
                    double b=f.readDouble();
                    intervalTestAcc+=b;
                    diff[j]=a-b;
                }
                start/=r;end/=r;fullTrainAcc/=r;intervalTrainAcc/=r;intervalTestAcc/=r;fullTestAcc/=r;
                fullResults.writeString(name+","+start+","+end+","+fullTrainAcc+","+intervalTrainAcc+","+fullTestAcc+","+intervalTestAcc+",,");
                for(double d:diff)
                    fullResults.writeString(d+",");
                fullResults.writeString("\n");
            }
        }
    }
    
    public static void combineFiles(String sourcePath, OutFile destination,int reps){
        for(int i=1;i<=reps;i++){
            File test=new File(sourcePath+"_"+i+".csv");
            if(test.exists()){
                InFile f = new InFile(sourcePath+"_"+i+".csv");
                destination.writeLine(f.readLine());
            }
        }
    }
    public static String VERSION="V1";
    
    public static void main(String[] args){
        VERSION="V3";
        debugRun();
        System.exit(0);
        combineResults("C:\\Users\\ajb\\Dropbox\\Results\\IntervalBasedDTW\\",DataSets.spectral,"ED",100);
        clusterRun(args);
        allSpectral();
//        

        
    }
}
