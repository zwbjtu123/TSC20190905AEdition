/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package old_development;

import development.DataSets;
import fileIO.OutFile;
import statistics.simulators.*;
import utilities.ClassifierTools;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class TimeDomainClassification {
    
    public static void EuclideanVsDTWParameterTest(String s, boolean warp){
        OutFile of = new OutFile(s);
        int[] sizes={10,100,300,500,1000,2000};
        int testSize=200;
        double[][] modelParas={{0,0,1},{0,0,1}};
        int seriesLength=100;
        int reps=10;
        int[] testCases={testSize/2,testSize/2};
        double paraInterval=0.1;
        double[] paraDiffs={0.01,0.02,0.05,0.1};
        for(int i=0; i<1/paraInterval;i++){ 
            modelParas[0][1]=0;
            modelParas[1][1]=0;
            modelParas[0][2]=1;
            modelParas[1][2]=1;
            for(int j=0; j<1/paraInterval;j++){
                for(int k=0; k<1/paraInterval;k++){
                    of.writeString(modelParas[0][0]+","+modelParas[0][1]+","+modelParas[0][2]);
//For each parameter set
                    for(int q=0;q<modelParas[0].length;q++){
                        for(int p=0;p<modelParas[0].length;p++)
                            modelParas[1][p]=modelParas[0][p];
                        modelParas[1][q]+=paraDiffs[0];
                        for(int p=0;p<sizes.length;p++){
                            int n=sizes[p];
                            double accEuclid=0;
                            double accDTW=0;
                            for(int m=0;m<reps;m++){
                                DataSimulator ds;
                                if(warp){
                                    SimulateTimeDomain sim = new SimulateTimeDomain(modelParas);
                                    sim.setWarping();
                                    ds=sim;
                                }
                                else
                                    ds = new SimulateTimeDomain(modelParas);
                                int[] casesPerClass={n/2,n/2};
                                Instances train=ds.generateDataSet(seriesLength, casesPerClass);
                                Instances test=ds.generateDataSet(seriesLength, testCases);
                                kNN euclid= new kNN(1);
                                DTW_1NN dtw= new DTW_1NN();
                                dtw.optimiseWindow(false);
                                accEuclid+=ClassifierTools.singleTrainTestSplitAccuracy(euclid, train, test);
                                accDTW+=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
                            }
                            of.writeString(n+","+accEuclid/reps+","+accDTW/reps);
                            System.out.println(n+","+accEuclid/reps+","+accDTW/reps);
                        }
                    }
                    of.writeString("\n");                    
                }
                modelParas[0][1]+=paraInterval;
            }
            modelParas[0][0]+=paraInterval;
        }
        
        
        }
        
    //</editor-fold>
  
    
    
  //<editor-fold defaultstate="collapsed" desc="Does 1NN Euclidean tend to DTW as train set size increase?">    
    public static void EuclideanVsDTWSimulation(String s, boolean warp){
        OutFile of = new OutFile(s);
        int startN=10, endN=5000, increment=10;
        int testSize=100;
        double[][] modelParas={{0,0.02,1},{0,0.019,1}};
        int seriesLength=100;
        int reps=10;
        int[] testCases={testSize/2,testSize/2};
        for(int n=startN;n<=endN;n+=increment){
            System.out.println(" Evaluating for "+n+" training cases");
            //Generate random data set with n training cases and testSize test casesj+

            double accEuclid=0;
            double accDTW=0;
            for(int j=0;j<reps;j++){
                DataSimulator ds;
                if(warp){
                    SimulateTimeDomain sim = new SimulateTimeDomain(modelParas);
                    sim.setWarping();
                    ds=sim;
                }
                else
                    ds = new SimulateTimeDomain(modelParas);
                int[] casesPerClass={n/2,n/2};
                Instances train=ds.generateDataSet(seriesLength, casesPerClass);
                Instances test=ds.generateDataSet(seriesLength, testCases);
                kNN euclid= new kNN(1);
                DTW_1NN dtw= new DTW_1NN();
                dtw.optimiseWindow(false);
                 accEuclid+=ClassifierTools.singleTrainTestSplitAccuracy(euclid, train, test);
                accDTW+=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
            }
            of.writeLine(n+","+accEuclid/reps+","+accDTW/reps);
            System.out.println(n+","+accEuclid/reps+","+accDTW/reps);
        }
        
    }
    //</editor-fold>
    
    public static void nearestNeighbour(boolean crossV){
           String[] files=DataSets.fileNames;
            OutFile of;
            if(crossV)
                of= new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\CV_NN_UCR.csv");
            else
                of= new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\1_NN_UCR.csv");
                 
            try{
                for(int i=0;i<files.length;i++){
                    System.gc();
                    System.out.println(" Problem file ="+files[i]);
                    Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+files[i]+"\\"+files[i]+"_TRAIN");
                    Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+files[i]+"\\"+files[i]+"_TEST");
                    NormalizeCase nc = new NormalizeCase();
                    train=nc.process(train);
                    test=nc.process(test);
                    kNN knn=null;
                    double a;
                    if(crossV){
                        knn= new kNN(100);
                        knn.setCrossValidate(crossV);
                        knn.buildClassifier(train);
                        knn.normalise(false);
                        a=ClassifierTools.accuracy(test,knn);
                   }
                    else{
                        knn= new kNN(1);
                        knn.normalise(false);
                        knn.buildClassifier(train);
                        a=ClassifierTools.accuracy(test,knn);
                    }                        
                    of.writeLine(files[i]+","+a);
                    System.out.println(files[i]+","+a);
                }
               }catch(Exception e){
               System.out.println(" Error ="+e);
               e.printStackTrace();
               System.exit(0);
           }
        
    }

    public static void main(String[] args){
        
        String str="ARSENAL";
        switch(str){
            case "TOTTS":
                
                break;
            case "ARSENAL":
                
        }
        
//        nearestNeighbour(true);
//        nearestNeighbour(false);
/*        OutFile of = new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\EuclidVsDTWResults\\test.csv");
        double[][] modelParas={{0,0.02,1},{0,0.019,1}};
        int seriesLength=100;
        Model.setDefaultSigma(1);
        SimulateTimeDomain sim=new SimulateTimeDomain(modelParas);
//        sim.setWarping();
        DataSimulator ds= sim;
       System.out.println("Model 0 variance ="+ds.getModels().get(0).getVariance());
       int[] casesPerClass={1,1};
       Instances train=ds.generateDataSet(seriesLength, casesPerClass);
       of.writeString(train.toString()); 
        Model.setDefaultSigma(1);
  */      
       String results="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\EuclidVsDTWResults\\WithWarpingTest1.csv"; 
        EuclideanVsDTWSimulation(results,true);
    }
    
    
}
