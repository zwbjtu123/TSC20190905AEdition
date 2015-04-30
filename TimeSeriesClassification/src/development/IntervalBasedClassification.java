/*
ACM Trans/ICDM paper: Interval based DTW
 */
package development;

import fileIO.*;
import java.io.File;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
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
public class IntervalBasedClassification {
    
//1. Train test full enumeration for 1-NN ED
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
        System.out.println(" FULL INTERVAL (0,"+(train.numAttributes()-2));
//Eval full interval first. 
        double[][] a =ClassifierTools.crossValidationWithStats(c, train,train.numInstances());
        bestAcc=a[0][0];  
//Find critical region for test.         
        double var=train.numInstances()*.5*.5;
        var=Math.sqrt(var)*1.96;
        for(int i=start;i<end-minInterval;i++){
//            System.out.println(" Evaluating starting from: "+i);
            Instances intervalTrain=new Instances(train);
            for(int k=0;k<i;k++)   //Delete attributes before start
                intervalTrain.deleteAttributeAt(0);
            for(int j=end-1;j>=i+minInterval;j--){  //Evaluate interval from i to j
                    
                a =ClassifierTools.crossValidationWithStats(c, intervalTrain,intervalTrain.numInstances());
//                    System.out.println(" interval ("+i+" "+j+")"+" has train acc ="+a[0][0]);
//                System.out.println("\t Number of attributes ="+(intervalTrain.numAttributes()-1)+" acc ="+a[0][0]);

//Only accept the new interval if it is significantly better.
                if(a[0][0]>bestAcc){ 
                    double p=(a[0][0]+bestAcc)/2; //Pooled proportion
                    double se=Math.sqrt(p*(1-p)/(2*intervalTrain.numInstances()));     //Standard error
                    double z=(a[0][0]-bestAcc)/se; 
//Test is not really appropriate because not normal and not independent!                    
                    if(z>1.96)
                    {
                        bestAcc=a[0][0];
                        bestStart=i;
                        bestEnd=j;
                        bestSD=a[1][0];
                        System.out.println(" Best interval so far: ("+bestStart+" "+bestEnd+")"+" has train acc ="+bestAcc);
                    }
                }
//Delete the last attribute at the end               
//                System.out.println("Num atts (inc class) ="+intervalTrain.numAttributes()+"  Class index ="+intervalTrain.classIndex()+"  Deleting attribute"+(intervalTrain.numAttributes()-2));
                 intervalTrain.deleteAttributeAt(intervalTrain.numAttributes()-2);
                // Estimate CV accuracyof classi
                
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
        results.writeLine(bestStart+","+bestEnd+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
        System.out.println(bestStart+","+bestEnd+","+bestAcc+","+fullIntervalTestAcc+","+testAcc);
    }
    
    public static void allSpectral(){
        OutFile results= new OutFile(DataSets.resultsPath+"IntervalBasedDTW\\EDTrainTest.csv");
        for(String s:DataSets.spectral){
//            String s="Coffee";
            results.writeString(s+",");
            System.out.println(s+",");
            Instances train=ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TEST");
            Classifier kNN=new kNN();
            NormalizeCase nc=new NormalizeCase();
            try {
                train=nc.process(train);
                test=nc.process(test);
                intervalClassifier(train,test,kNN,results);
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
//First arg gives the dataset number: 1 to 82
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
        f= new File(DataSets.clusterPath+"Results/Interval/"+s+"/"+dist);
        if(!f.isDirectory())
            f.mkdir();
        
        OutFile results= new OutFile(DataSets.clusterPath+"Results/Interval/"+s+"/"+dist+"/"+s+"_"+dist+"_"+fold+".csv");
        results.writeString(s+",");
        Instances train=ClassifierTools.loadData(DataSets.clusterPath+"TSC Problems/"+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.clusterPath+"TSC Problems/"+s+"/"+s+"_TEST");
//Form randomised split
        if(fold!=1){
            Instances all = new Instances(train);
            all.addAll(test);
            int testSize = test.numInstances();
            Random r = new Random(fold);
            all.randomize(r);
            //Form new Train/Test split
            Instances tr = new Instances(all);
            Instances te = new Instances(all, 0);
            for (int j = 0; j < testSize; j++)
                te.add(tr.remove(0));
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
            intervalClassifier(train,test,knn,results);
        } catch (Exception ex) {
            System.out.println(" Error : "+ex);
        }
       
        
    }
    public static void combineResults(String path,String[] sourceFiles, String type,int reps){
        
       OutFile fullResults=new OutFile(path+type+".csv");
        for(String s:sourceFiles){
//Chck it exists
            File inf=new File(path+"/"+s+"/"+type);
            if(inf.exists()){//Collate the files
                OutFile results=new OutFile(path+s+type+".csv");
                combineFiles(path+"/"+s+"/"+type+"/"+s+"_"+type,results,reps);
                results.closeFile();
                InFile f = new InFile(path+s+type+".csv");
    //Read in the accuracies, find mean and SD of difference             
                String name="";
                int start=0,end=0;
                double trainAcc=0;
                double sd=0;
                double fullTestAcc=0;
                double fullTestSS=0;
                double intervalTestAcc=0;
                double intervalTestSS=0;
                double[] diff=new double[reps];
                for(int j=0;j<reps;j++){
                    name=f.readString();
                    start+=f.readInt();
                    end+=f.readInt();
                    trainAcc+=f.readDouble();
                    sd+=f.readDouble();
                    double a=f.readDouble();
                    fullTestAcc+=a;
                    fullTestSS+=a*a;
                    double b=f.readDouble();
                    intervalTestAcc+=b;
                    intervalTestSS+=b*b;
                    diff[j]=a-b;
                }
                start/=reps;end/=reps;trainAcc/=reps;fullTestAcc/=reps;intervalTestAcc/=reps;
                fullResults.writeString(name+","+start+","+end+","+trainAcc+","+fullTestAcc+","+fullTestSS+","+intervalTestAcc+","+intervalTestSS+",,");
                for(double d:diff)
                    fullResults.writeString(d+",");
                fullResults.writeString("\n");
            }
        }
    }
    
    public static void combineFiles(String sourcePath, OutFile destination,int reps){
        for(int i=1;i<=reps;i++){
            InFile f = new InFile(sourcePath+"_"+i+".csv");
            destination.writeLine(f.readLine());
        }
    }
    public static void main(String[] args){
        allSpectral();
//        combineResults("C:\\Users\\ajb\\Dropbox\\Results\\IntervalBasedDTW\\",DataSets.spectral,"ED",100);
        System.exit(0);
        clusterRun(args);
        
    }
}
