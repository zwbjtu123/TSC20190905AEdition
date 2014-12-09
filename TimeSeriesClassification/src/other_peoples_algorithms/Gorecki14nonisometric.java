/*
 Recreates the results in
@article{gorecki14nonisometric,
  title={Non-isometric transforms in time series classification using DTW},
  author={T. Gorecki and M. Luczak},
  journal={Knowledge-Based Systems},
  volume={online first},
  year={2014}
}
 */

package other_peoples_algorithms;

import development.DataSets;
import development.TimeSeriesClassification;
import static development.TimeSeriesClassification.UCRProblems;
import fileIO.InFile;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.lazy.DTW_kNN;
import weka.classifiers.lazy.kNN;
import weka.core.DTW_DistanceBasic;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.*;

/**
Combines DTW distance/DTW distance between derivatives with DTW distance
between transforms of time series: cosine, sine and Hilbert transform. Forms  a weighted sum of two distances, finding weights by some form of pairwise optimization which might be like the SMO/SVM technique, although this is not that clear, as it then describes setting alpha through cross validation.

Add three new BasicFilters in weka.filters.timeseries:  Sin, Cosine and Hilbert
 */
public class Gorecki14nonisometric {
//To test the three transforms
        public static void testTransforms(){
            String s="Beef";
            OutFile of1 = new OutFile("C:\\Users\\ajb\\Dropbox\\test\\BeefCosine_TRAIN.arff");
            OutFile of2 = new OutFile("C:\\Users\\ajb\\Dropbox\\test\\BeefCosine_TEST.arff");
            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TRAIN");			
            Cosine cosTransform= new Cosine();
            Sine sinTransform=new Sine();
            Hilbert hilbertTransform= new Hilbert();
            System.out.println(" Data set ="+s);
            try {
                Instances cosTrain=cosTransform.process(train);
                Instances cosTest=cosTransform.process(test);
                of1.writeString(cosTrain+"");
                of2.writeString(cosTest+"");
                System.out.println(" Cosine trans complete");
                kNN a=new DTW_kNN(1);
                a.normalise(false);
                a.buildClassifier(cosTrain);
                double acc=ClassifierTools.accuracy(cosTest, a);
                System.out.println(" Cosine acc ="+acc);

                Instances sinTrain=sinTransform.process(train);
                Instances sinTest=sinTransform.process(test);
                System.out.println(" Sine trans complete");
                a=new DTW_kNN(1);
                a.normalise(false);
                a.buildClassifier(sinTrain);
                acc=ClassifierTools.accuracy(sinTest, a);
                System.out.println(" Sine acc ="+acc);

                Instances hilbertTrain=hilbertTransform.process(train);
                Instances hilbertTest=hilbertTransform.process(test);
                System.out.println(" Hilbert trans complete");              
                a=new DTW_kNN(1);
                a.normalise(false);
                a.buildClassifier(hilbertTrain);
                acc=ClassifierTools.accuracy(hilbertTest, a);
                System.out.println(" Hilbert acc ="+acc);
                
                
            } catch (Exception ex) {
                Logger.getLogger(Gorecki14nonisometric.class.getName()).log(Level.SEVERE, null, ex);
            }
        
        
    }

//to recreate columns 6,7, 8 of table 2    
//Combine DTW with other transform with a weighting alpha
//where alpha in [0,1], searched for increments of 0.01  
//Note only find the distances once for each alpha          
    public static void combinedDistanceClassifiersTable2(){
        OutFile of = new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\gorecki14nonisometricRecreation.csv");
        double[] alphaVals=new double[100];
        for(int i=1;i<alphaVals.length;i++)
            alphaVals[i]=0.01+alphaVals[i];
        double[] mistakes=new double[alphaVals.length];
        for(String s:DataSets.ucrSmall){
            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TRAIN");			
            Cosine cosTransform= new Cosine();
            Sine sinTransform=new Sine();
            Hilbert hilbertTransform= new Hilbert();
            of.writeString(s+",");
            System.out.println(" Data set ="+s);
            try {
                Instances cosTrain=cosTransform.process(train);
                System.out.println(" Cosine trans complete");
                Instances sinTrain=sinTransform.process(train);
                System.out.println(" Sine trans complete");
                Instances hilbertTrain=hilbertTransform.process(train);
                System.out.println(" Hilbert trans complete");              
                for(int i=0;i<train.numInstances();i++){ //
                    double[] dtwDist=new double[train.numInstances()];
                    double[] dtwcDist=new double[train.numInstances()];
                    double[] dtwsDist=new double[train.numInstances()];
                    double[] dtwhDist=new double[train.numInstances()];
//Find distances between element i and the rest                    
                    for(int j=0;j<train.numInstances();j++){
//                        if(i!=j){
//                            dtwDist[j]=dist;
                        
                    }
                }
                
            } catch (Exception ex) {
                Logger.getLogger(Gorecki14nonisometric.class.getName()).log(Level.SEVERE, null, ex);
            }
            of.writeString("\n");
            
            
        }
        
    }

     public static void singleDistanceClassifiersTable2(){
            OutFile of = new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\gorecki14Table2Combined.csv");
        for(String s:DataSets.ucrSmall){
            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TRAIN");			
            Cosine cosTransform= new Cosine();
            Sine sinTransform=new Sine();
            Hilbert hilbertTransform= new Hilbert();
            of.writeString(s+",");
            System.out.println(" Data set ="+s);
            try {
                Instances cosTrain=cosTransform.process(train);
                Instances cosTest=cosTransform.process(test);
                System.out.println(" Cosine trans complete");
                kNN a=new DTW_kNN(1);
                a.normalise(false);
                a.buildClassifier(cosTrain);
                double acc=ClassifierTools.accuracy(cosTest, a);
                of.writeString(acc+",");
                System.out.println(" Cosine acc ="+acc);

                Instances sinTrain=sinTransform.process(train);
                Instances sinTest=sinTransform.process(test);
                System.out.println(" Sine trans complete");
                a=new DTW_kNN(1);
                a.normalise(false);
                a.buildClassifier(sinTrain);
                acc=ClassifierTools.accuracy(sinTest, a);
                of.writeString(acc+",");
                System.out.println(" Sine acc ="+acc);

                Instances hilbertTrain=hilbertTransform.process(train);
                Instances hilbertTest=hilbertTransform.process(test);
                System.out.println(" Hilbert trans complete");              
                a=new DTW_kNN(1);
                a.normalise(false);
                a.buildClassifier(hilbertTrain);
                acc=ClassifierTools.accuracy(hilbertTest, a);
                of.writeString(acc+",");
                System.out.println(" Hilbert acc ="+acc);
                
                
            } catch (Exception ex) {
                Logger.getLogger(Gorecki14nonisometric.class.getName()).log(Level.SEVERE, null, ex);
            }
            of.writeString("\n");
            
            
        }
        
    }

//Requires  derivativeDistanceCVClassifierTable2 to have been run  
     public static void derivativeDistanceTestClassifierTable2(){
        InFile inf = new InFile("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\gorecki14Table2CombinedTrainCVParasFull.csv");
        OutFile of = new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\gorecki14Table2DD_Results.csv");
        of.writeString("Problem,TrainCVError,TestMistakes,TestError");

        for(String s:DataSets.ucrNames){
            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TRAIN");
            String problem=inf.readString();
            if(!s.equals(problem)){
                System.out.println("ERROR: file mismatch: "+s+" and from CV file :"+problem);
                System.exit(0);
            }
//           of.writeLine(s+","+n+","+df.format(bestAlpha)+","+mistakes[bestAlpha]+","+df.format(mistakes[bestAlpha]/(double)n)+","+df.format(alpha[bestAlpha]));
            int n=inf.readInt();
            n=inf.readInt();
            n=inf.readInt();
            double trainCVError=inf.readDouble();
            double alpha=inf.readDouble();
//            String rest=inf.readLine();
            double a=Math.cos(alpha);
            double b=Math.sin(alpha);
            of.writeString(problem+",");
            int mistakes=0;
//Find DTW on test and test differential            
            for(Instance ins:test){
                double[] temp=ins.toDoubleArray();
//Recover and remove target class for test instance
                double testClass=temp[temp.length-1];  
                double[] testData=new double[temp.length-1];
                System.arraycopy(temp, 0, testData, 0, temp.length-1);
                double[] testDiffs= new double[testData.length-1];
                for(int j=0;j<testDiffs.length;j++)
                    testDiffs[j]=testData[j+1]-testData[j];
//Find nearest neighbour in train data
                double minDist = Double.MAX_VALUE;
                double pred=0;
                for(Instance tr:train){
//Recover and remove target class for train instance
                    temp=tr.toDoubleArray();
                    double trainClass=temp[temp.length-1];  
                    double[] trainData=new double[temp.length-1];
                    System.arraycopy(temp, 0, trainData, 0, temp.length-1);
                    double[] trainDiffs= new double[trainData.length-1];
                    for(int j=0;j<trainDiffs.length;j++)
                        trainDiffs[j]=trainData[j+1]-trainData[j];                
//Find DTW distance for dataand diffs  
                    DTW_DistanceBasic dtw= new DTW_DistanceBasic();
                    double d1=dtw.distance(testData,trainData,Double.MAX_VALUE);
                    double d2=dtw.distance(testDiffs,trainDiffs,Double.MAX_VALUE);;
                    double dist =a*d1+b*d2;
                    if(dist<minDist){
                        minDist=dist;
                        pred=trainClass;
                    }
               }
               if(pred!=testClass)
                   mistakes++;
        }               
        of.writeString(trainCVError+","+mistakes+","+(mistakes/(double)test.numInstances())+"\n");
        System.out.print(problem+","+trainCVError+","+mistakes+","+(mistakes/(double)test.numInstances())+"\n");
        
     }
 //Get instance to classify
 
     }
     public static void derivativeDistanceCVClassifierTable2(){
        OutFile of = new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\gorecki14Table2CombinedTrainCVParasFull.csv");
        DecimalFormat df = new DecimalFormat("###.####");
        of.writeString(",bestAlphaIndex,nosMistakes,bestTrainError,bestALphaValue");

        for(String s:DataSets.ucrNames){
            of.writeString(s+",");
            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+s+"\\"+s+"_TRAIN");			
            double step = 0.01;
            int k = (int)((Math.PI/2)/step);
            double[] alpha=new double[k];
            for(int i=1;i<alpha.length;i++)
                alpha[i]=alpha[i-1]+step;
            double[] a =new double[k];
            double[] b =new double[k];
            for(int i=0;i<alpha.length;i++){
                a[i]=Math.cos(alpha[i]);
                b[i]=Math.sin(alpha[i]);
            }
            int n = train.numInstances();
            int[] mistakes=new int[k];
            for(int i=0;i<n;i++){
//Get instance to classify
                double[] temp=train.instance(i).toDoubleArray();
//Recover and remove target class
                double actual1=temp[temp.length-1];  
                double[] data1=new double[temp.length-1];
                System.arraycopy(temp, 0, data1, 0, temp.length-1);
                double[] diffs1= new double[data1.length-1];
                for(int j=0;j<diffs1.length;j++)
                    diffs1[j]=data1[j+1]-data1[j];
                double[] minDists = new double[k];
                for(int j=0;j<k;j++)
                   minDists[j]=Double.MAX_VALUE;     
                double[] preds = new double[k];
                for(int j=0;j<n;j++){
                    if(i!=j){
//Recover data2  and diffs2                     
                        temp=train.instance(j).toDoubleArray();
//Recover and remove target class
                        double actual2=temp[temp.length-1];  
                        double[] data2=new double[temp.length-1];
                        System.arraycopy(temp, 0, data2, 0, temp.length-1);
                        double[] diffs2= new double[data2.length-1];
                        for(int m=0;m<diffs1.length;m++)
                            diffs2[m]=data2[m+1]-data2[m];
//Find the DTW differences     
                        DTW_DistanceBasic dtw= new DTW_DistanceBasic();
                        double d1=dtw.distance(data1,data2,Double.MAX_VALUE);
                        double d2=dtw.distance(diffs1,diffs2,Double.MAX_VALUE);;
//Update the min dist so for and the predicted for all weights
                        for(int m=0;m<minDists.length;m++){
                            double dist=a[m]*d1+b[m]*d2;
                            if(dist<minDists[m]){
                                minDists[m]=dist;
                                preds[m]=actual2;
                            }
                        }
                    }
                }
//Count whether predictions were correct for each para weight  
                for(int m=0;m<minDists.length;m++){
                   // System.out.print(preds[m]+",");
                    if(actual1!=preds[m])
                        mistakes[m]++;
                    }
            }
//Find best alpha, save to file with the training error
            int bestAlpha=0;
            for(int m=1;m<mistakes.length;m++){
              if(mistakes[m]<mistakes[bestAlpha])
                  bestAlpha=m;
            }            
            of.writeLine(s+","+n+","+df.format(bestAlpha)+","+mistakes[bestAlpha]+","+df.format(mistakes[bestAlpha]/(double)n)+","+df.format(alpha[bestAlpha]));
            System.out.println(s+","+n+","+df.format(bestAlpha)+","+mistakes[bestAlpha]+","+df.format(mistakes[bestAlpha]/(double)n)+","+df.format(alpha[bestAlpha]));
        }
     }
    public static void main(String[] args){
//        testTransforms();
//        singleDistanceClassifiersTable2();
//        derivativeDistanceClassifierTable2();
        derivativeDistanceTestClassifierTable2();
    }
    
}
