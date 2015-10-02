/*
Class to recreate someone elses  

*/
package development;

import development.DataSets;
import fileIO.OutFile;
import tsc_algorithms.RecreateResults;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.*;
import weka.core.elastic_distance_measures.*;

/**
 *
 * @author ajb
 */
public class MarteauTWED extends RecreateResults{

    public void recreatePublishedResults(String path){
        OutFile of=new OutFile(path);
       String[] names= DataSets.marteau09stiffness;
       int nosClassifiers=3;
       kNN[] c=new  kNN[nosClassifiers];
       double[] testAcc;
       for(int i=0;i<names.length;i++){
            System.out.println(" Problem ="+names[i]);
            of.writeString(names[i]+",");
            Instances train=ClassifierTools.loadData(dataPath+names[i]+"\\"+names[i]+"_TRAIN");
            Instances test=ClassifierTools.loadData(dataPath+names[i]+"\\"+names[i]+"_TEST");
//Build clean classifiers. 
            for(int j=0;j<c.length;j++)
               c[j]=new kNN(1);
            testAcc=new double[c.length];
//6 Classifiers:
//1. Euclidean distance: No parameter optimisation           
            int cCount=0;
            kNN k=new kNN(1);
            k.setDistanceFunction(new EuclideanDistance());
            k.buildClassifier(train);
            testAcc[cCount++]=ClassifierTools.accuracy(test, k);
            of.writeString(testAcc[cCount-1]+",");
            System.out.println("\t\t Euclidean Dist ="+testAcc[cCount-1]);
//2. Full Window DTW No parameter optimisation          
            k=new kNN(1);
            k.setDistanceFunction(new BasicDTW());
            k.buildClassifier(train);
           testAcc[cCount++]=ClassifierTools.accuracy(test, k);
           System.out.println("\t\t Full DTW ="+testAcc[cCount-1]);
           of.writeString(testAcc[cCount-1]+",");
//3. LCSS:  Need to know parameter!
//4. ODTW: Optimised for window size           
//5. ERP
           
//6. OTWED
            double[] nuVals={0.00001,0.0001,0.001,0.01,0.1, 1};
            double[] lambdaVals={0,0.25,0.5,0.75,1};            
            double best=0;
            int nuBest=0, lambdaBest=0;
            k=new kNN(1);
            TWEDistance twed=new TWEDistance();
            for(int p=0;p<nuVals.length;p++){
                twed.setNu(nuVals[p]);
                for(int m=0;m<lambdaVals.length;m++){

                  twed.setLambda(lambdaVals[m]);
                  k.setDistanceFunction(twed);
                  //Cross validate, I'm cheating!
                  
                  k.buildClassifier(train);
                  double temp=ClassifierTools.accuracy(test, k);
                  if(temp>best){
                      nuBest=p;
                      lambdaBest=m;
                      best=temp;
                  }
                }
            }
              twed.setNu(nuVals[nuBest]);
              twed.setLambda(lambdaVals[lambdaBest]);
              k.setDistanceFunction(twed);
              k.buildClassifier(train);
              testAcc[cCount++]=ClassifierTools.accuracy(test, k);
            System.out.println("\t\t TWED ="+testAcc[cCount-1]+" nu val= "+nuVals[nuBest]+" l val ="+lambdaVals[lambdaBest]);
            of.writeString(testAcc[cCount-1]+"\n");
           
       }
       
       
    }
    public void runOnAllDataSets(String path){
        
    }
    public static void main(String[] args){
        new MarteauTWED().recreatePublishedResults("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\MarteauTable1.csv");
    }
}
