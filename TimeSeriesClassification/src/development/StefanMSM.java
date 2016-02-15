/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import development.DataSets;
import fileIO.OutFile;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.MSMDistance;




public class StefanMSM{

    /* Compares MSM to cDTW, DTW, ERP, Euclid*/
    public void recreatePublishedResults(String path){
        OutFile of=new OutFile(path);
       String[] names= DataSets.stefan13movesplit;
       int nosClassifiers=3;
       kNN[] c=new  kNN[nosClassifiers];
       double[] testAcc;
       for(int i=0;i<names.length;i++){
            System.out.println(" Problem ="+names[i]);
            of.writeString(names[i]+",");
            Instances train=ClassifierTools.loadData(DataSets.problemPath+names[i]+"\\"+names[i]+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.problemPath+names[i]+"\\"+names[i]+"_TEST");
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
//3. cDTW:  Need to know parameter!   
//5. ERP
           
//6. MSM: For each of the 20 data sets, the value for c was chosen from the
//           set {0.01, 0.1, 1, 10, 100}
            k=new kNN(1);
            double[] paras={0.01, 0.1, 1, 10, 100};
            MSMDistance msm= new MSMDistance();
            int best=0;
            double bestAcc=0;
            for(int p=0;p<paras.length;p++){
                msm.setC(paras[p]);
                k=new kNN(1);
                k.setDistanceFunction(msm);
                
                
                  k.buildClassifier(train);
                  double temp=ClassifierTools.accuracy(test, k);
/*                  if(temp>best){
                      nuBest=p;
                      lambdaBest=m;
                      best=temp;
                  }
  */              }
            }           
            
            
/*            k.setDistanceFunction(msm);
            k.buildClassifier(train);
            testAcc[cCount++]=ClassifierTools.accuracy(test, k);
            System.out.println("\t\t TWED ="+testAcc[cCount-1]+" nu val= "+nuVals[nuBest]+" l val ="+lambdaVals[lambdaBest]);
            of.writeString(testAcc[cCount-1]+"\n");
  */         
       }
       
       
    
}
