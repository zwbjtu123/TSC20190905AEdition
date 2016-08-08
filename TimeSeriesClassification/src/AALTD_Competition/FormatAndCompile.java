/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AALTD_Competition;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.Random;
import transformations.PCA;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.Differences;
import weka.filters.timeseries.SummaryStats;

/**
 *
 * @author ajb
 */
public class FormatAndCompile {

   public static void combineSingleFolds(String source, int cases){
        int folds=cases;
//1. Check all folds are present
        boolean allPresent=true;
        File f=new File(source+"trainPredAll.csv");
        if(!f.exists() || f.length()==0){
            OutFile out=new OutFile(source+"trainPredAll.csv");
            for(int k=0;k<folds;k++){
                f=new File(source+"trainPredAll_"+k+".csv");
                if(!f.exists()){
                   allPresent=false;
                   System.out.println("MISSING "+source+"trainPredAll_"+k+".csv");
               }
           }       
            if(allPresent)//We have all folds, can proceed to collate
            {
                String[] lines=new String[folds];
                double correct=0;
                int c=0;
                for(int k=0;k<folds;k++){
                    InFile inf=new InFile(source+"trainPredAll_"+k+".csv");
                    if(k==0){
                        out.writeLine(inf.readLine());
                        out.writeLine(inf.readLine());
                       }else{
                        inf.readLine();
                        inf.readLine();
                    }
                     c=inf.readInt();
                     if(c==1)
                         correct++;
                    lines[k]=inf.readLine();
                }
                out.writeLine((correct/(double)folds)+"");
                for(int i=0;i<folds;i++)
                    out.writeLine(lines[i]);
            }
        }
   }

   public static void formatTwoClassPredictionProblem(){
        Instances data=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\\\AllConcatenated\\AALTD_All_XYZ_OffsetRemoved_TRAIN");
        Instances first=new Instances(data);
        int count=0;
        while(count<first.numInstances()){
            int t=(int)first.instance(count).classValue();
            if(t==1 ||  t==2)
                count++;
            else
                first.delete(count);
        }
        int nos=50;//Remove the left hand sensor
        for(int i=0;i<8;i++){
            for(int j=0;j<nos;j++){
                first.deleteAttributeAt(i*nos);
            }
        }
//Delete all zero attributes
        int[] features=InstanceTools.removeConstantTrainAttributes(first);
        for(int i=0;i<features.length;i++)
            System.out.println("Deleted att :"+features[i]);
        OutFile out = new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_All_XYZ_TwoClass_TRAIN.arff");
        out.writeLine(first.toString());
        Instances second=new Instances(data);
        for(Instance ins:second){
            int t=(int)ins.classValue();
            if(t>2)
                ins.setClassValue(0);
        }

//Remove the left hand sensor
        for(int i=0;i<8;i++){
            for(int j=0;j<nos;j++){
                second.deleteAttributeAt(i*nos);
            }
        }
        features=InstanceTools.removeConstantTrainAttributes(second);
        
        out = new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_All_XYZ_ThreeClass_TRAIN.arff");
        out.writeLine(second.toString());
        
   }
    
   public static void combineSingleFoldsIntoChannels(String source){
        String[] axis={"X","Y","Z"};
        int channels=8;
        int folds=180;
//1. Check all folds are present
        boolean allPresent=true;
        for(int i=0;i<axis.length;i++){
            for(int j=0;j<channels;j++){
                for(int k=0;k<folds;k++){
                    File f=new File(source+"trainPred"+axis[i]+j+"_"+k+".csv");
                    if(!f.exists()){
                       allPresent=false;
                       System.out.println("MISSING "+source+"trainPred"+axis[i]+j+"_"+k+".csv");
                   }
               }       
            }
        }
        if(allPresent)//We have all folds, can proceed to collate
        {
            for(int i=0;i<axis.length;i++){
                for(int j=0;j<channels;j++){
                    OutFile out=new OutFile(source+"trainPred"+axis[i]+j+".csv");
                    String[] lines=new String[folds];
                    for(int k=0;k<folds;k++){
                        InFile inf=new InFile(source+"trainPred"+axis[i]+j+"_"+k+".csv");
                        if(k==0){
                            out.writeLine(inf.readLine());
                            out.writeLine(inf.readLine());
                            inf.readLine();
                           }else{
                            inf.readLine();
                            inf.readLine();
                            inf.readLine();
                        }
                        lines[k]=inf.readLine();
                    }
//Work out CV accuracy
                    int correct=0;
                    for(int k=0;k<folds;k++){
                        String[] split=lines[k].split(",");
                        if(split[0].equals(split[1]))
                            correct++;
                    }
                    out.writeLine((((double)correct)/folds)+"");
                    for(int k=0;k<folds;k++)
                       out.writeLine(lines[k]);     
                }
            }
        }
   }

   public static void ensembleSingleSensorAndDimension(String source, OutFile results){
        Instances data=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD0_X_TRAIN");
        String[] axis={"X","Y","Z"};
        int channels=8;
//Open all and scan in header + CV acc       
        InFile[][] all=new InFile[3][channels];
        double[][] cvAcc=new double[3][channels];
        double[][] allProbs=new double[data.numInstances()][data.numClasses()];
        int[][] allPreds=new int[2][data.numInstances()];
        results.writeLine("combo1ChannelVote");
        double correct=0;
           for(int j=0;j<channels;j++){
              for(int i=0;i<axis.length;i++){
                  all[i][j]=new InFile(source+"trainPred"+axis[i]+j+".csv");
                  all[i][j].readLine();
                  all[i][j].readLine();
                  cvAcc[i][j]=all[i][j].readDouble();
              }
           }
        
        for(int n=0;n<data.numInstances();n++){
           int classVal=(int)data.instance(n).classValue();
           for(int j=0;j<channels;j++){
              for(int i=0;i<axis.length;i++){
//Read in the data for this instance
                  String[] str=all[i][j].readLine().split(",");
//Sanity check classes correct
                  if(classVal!=Integer.parseInt(str[0])){
                      System.out.println("FUCKSHITARSE: Not alligned: debug");
                      System.exit(0);
            //Prediction weighted
                  }
                  int pred=Integer.parseInt(str[1]);
                  allPreds[0][n]=classVal;
//Weight by probs
                  for(int k=0;k<data.numClasses();k++)
                        allProbs[n][k]+=cvAcc[i][j]*Double.parseDouble(str[3+k]);
               }
           }
           double sum=0;
            for(int k=0;k<data.numClasses();k++)
                  sum+=allProbs[n][k];
            for(int k=0;k<data.numClasses();k++)
                 allProbs[n][k]/=sum;
           allPreds[1][n]=0;
           for(int i=1;i<allProbs[n].length;i++){
               if(allProbs[n][allPreds[1][n]]<allProbs[n][i])
                   allPreds[1][n]=i;
           }
           if(allPreds[0][n]==allPreds[1][n])
               correct++;
       }
       results.writeLine(correct/data.numInstances()+"");       
        for(int n=0;n<data.numInstances();n++){
            results.writeString(allPreds[0][n]+","+allPreds[1][n]+",");       
            for(int i=0;i<allProbs[n].length;i++)
                results.writeString(","+allProbs[n][i]);
            results.writeString("\n");
        }
   } 

   public static void ensembleSingleSensor(String source, OutFile results){
        Instances data=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\XYZBySensor\\AALTDSensor0_XYZ_TRAIN");
        String[] axis={"X","Y","Z"};
        int sensors=8;
//Open all and scan in header + CV acc       
        InFile[] all=new InFile[sensors];
        double[] cvAcc=new double[sensors];
        double[][] allProbs=new double[data.numInstances()][data.numClasses()];
        int[][] allPreds=new int[2][data.numInstances()];
        results.writeLine("combo1ChannelVote");
        double correct=0;
        for(int i=0;i<sensors;i++){
               all[i]=new InFile(source+"trainPredXYZ"+i+".csv");
               all[i].readLine();
               all[i].readLine();
               cvAcc[i]=all[i].readDouble();
        }
        
        for(int n=0;n<data.numInstances();n++){
           int classVal=(int)data.instance(n).classValue();
           for(int i=0;i<sensors;i++){
//Read in the data for this instance
                String[] str=all[i].readLine().split(",");
                //Sanity check classes correct
                if(classVal!=Integer.parseInt(str[0])){
                  System.out.println("FUCKSHITARSE: Not alligned: debug");
                  System.exit(0);
                //Prediction weighted
                }
                int pred=Integer.parseInt(str[1]);
                allPreds[0][n]=classVal;
                //Weight by probs
                for(int k=0;k<data.numClasses();k++)
                    allProbs[n][k]+=cvAcc[i]*Double.parseDouble(str[3+k]);
            }
            double sum=0;
            for(int k=0;k<data.numClasses();k++)
                  sum+=allProbs[n][k];
            for(int k=0;k<data.numClasses();k++)
                 allProbs[n][k]/=sum;
           allPreds[1][n]=0;
           for(int i=1;i<allProbs[n].length;i++){
               if(allProbs[n][allPreds[1][n]]<allProbs[n][i])
                   allPreds[1][n]=i;
           }
           if(allPreds[0][n]==allPreds[1][n])
               correct++;
       }
       results.writeLine(correct/data.numInstances()+"");       
        for(int n=0;n<data.numInstances();n++){
            results.writeString(allPreds[0][n]+","+allPreds[1][n]+",");       
            for(int i=0;i<allProbs[n].length;i++)
                results.writeString(","+allProbs[n][i]);
            results.writeString("\n");
        }
   } 
   


   public static void ensembleCote(String source, Instances data,OutFile results, String[] components){

        DecimalFormat df= new DecimalFormat("##.######");
        int classifiers=components.length;
        System.out.println(" Nos classifiers ="+classifiers);
        System.out.println(" Nos cases ="+data.numInstances());
//Open all and scan in header + CV acc       
        InFile[] all=new InFile[classifiers];
        double[] cvAcc=new double[classifiers];
        double[][] classAccs=new double[classifiers][];
        double[][] allProbs=new double[data.numInstances()][data.numClasses()];
        int[][] allPreds=new int[2][data.numInstances()];
        results.writeLine(data.relationName());
        results.writeString("COTE,,");
        for(String str:components)
            results.writeString(str+",");
        results.writeLine("Concat data");
        double correct=0;
        for(int i=0;i<classifiers;i++){
               all[i]=new InFile(source+components[i]+"\\trainPredAll.csv");
               all[i].readLine();
               all[i].readLine();
               cvAcc[i]=all[i].readDouble();
               System.out.println(" Classifier "+components[i]+" CV Acc ="+cvAcc[i]);
        classAccs[i]=findAccuracyPerClass(source+components[i]+"\\trainPredAll.csv",data.numClasses(),data.numInstances());
        
               all[i]=new InFile(source+components[i]+"\\trainPredAll.csv");
               all[i].readLine();
               all[i].readLine();
               all[i].readLine();
        }
    
        int power=4;
        for(int n=0;n<data.numInstances();n++){
           int classVal=(int)data.instance(n).classValue();
           for(int i=0;i<classifiers;i++){
//Read in the data for this instance
                String[] str=all[i].readLine().split(",");
                //Sanity check classes correct
                if(classVal!=Integer.parseInt(str[0])){
                  System.out.println("FUCKSHITARSE: Not alligned: debug");
                  System.exit(0);
                //Prediction weighted
                }
                allPreds[0][n]=classVal;
                //Weight by probs
                if(useProbsToEnsemble){
                    for(int k=0;k<data.numClasses();k++)
//                        allProbs[n][k]+=Math.pow(cvAcc[i],power)*Double.parseDouble(str[3+k]);
                        allProbs[n][k]+=Math.pow(classAccs[i][k],power)*Double.parseDouble(str[3+k]);                    
                }
                else
                    allProbs[n][Integer.parseInt(str[1])]+=Math.pow(classAccs[i][Integer.parseInt(str[1])],power);                    
                
            }
            double sum=0;
            for(int k=0;k<data.numClasses();k++)
                  sum+=allProbs[n][k];
            for(int k=0;k<data.numClasses();k++)
                 allProbs[n][k]/=sum;
           allPreds[1][n]=0;
           for(int i=1;i<allProbs[n].length;i++){
               if(allProbs[n][allPreds[1][n]]<allProbs[n][i])
                   allPreds[1][n]=i;
           }
           if(allPreds[0][n]==allPreds[1][n])
               correct++;
       }
       results.writeString(correct/data.numInstances()+",");       
       for(double d: cvAcc)
           results.writeString(","+d); 
       for(double[] cd: classAccs){
           results.writeString(",");
           for(double d:cd)
               results.writeString(","+d);
       }
       results.writeString("\n");       
           
        for(int n=0;n<data.numInstances();n++){
            results.writeString(allPreds[0][n]+","+allPreds[1][n]+",");       
            for(int i=0;i<allProbs[n].length;i++)
                results.writeString(","+df.format(allProbs[n][i]));
            results.writeString("\n");
        }
   } 
   public static void createZeroed() throws Exception{
        int sensors=8;
        OutFile normedByDimension=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTDZeroed_TRAIN.arff");
        Instances[] bySensor=new Instances[sensors];
        for(int j=0;j<sensors;j++){
            Instances x,y,z; 
            x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_X_TRAIN");
            y=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Y_TRAIN");
            z=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Z_TRAIN");
            for(Instance ins:x){
                double first=ins.value(0);
                for(int i=0;i<ins.numAttributes()-1;i++)
                    ins.setValue(i, ins.value(i)-first);
            }
            for(Instance ins:y){
                double first=ins.value(0);
                for(int i=0;i<ins.numAttributes()-1;i++)
                    ins.setValue(i, ins.value(i)-first);
            }
            for(Instance ins:z){
                double first=ins.value(0);
                for(int i=0;i<ins.numAttributes()-1;i++)
                    ins.setValue(i, ins.value(i)-first);
            }
            bySensor[j]=new Instances(x);
            bySensor[j].setClassIndex(-1);
            bySensor[j].deleteAttributeAt(bySensor[j].numAttributes()-1); 
            bySensor[j]=Instances.mergeInstances(bySensor[j], y);
            bySensor[j].setClassIndex(-1);
            bySensor[j].deleteAttributeAt(bySensor[j].numAttributes()-1); 
            bySensor[j]=Instances.mergeInstances(bySensor[j], z);
            bySensor[j].setClassIndex(bySensor[j].numAttributes()-1);
            bySensor[j].setRelationName("AALTDSensorZeroed"+j);
        }
        Instances all=new Instances(bySensor[0]);
        all.setClassIndex(-1);
        all.deleteAttributeAt(all.numAttributes()-1); 
        for(int j=1;j<sensors-1;j++){
            all=Instances.mergeInstances(all,bySensor[j]);
            all.setClassIndex(-1);
            all.deleteAttributeAt(all.numAttributes()-1); 
        }
        all=Instances.mergeInstances(all,bySensor[sensors-1]);
        all.setClassIndex(all.numAttributes()-1);
        all.setRelationName("AALTD_ZERO");
        normedByDimension.writeString(all.toString());
   }
   
   
   
   public static void createDifference() throws Exception{
        int sensors=8;
        OutFile normedByDimension=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTDDiff_TRAIN.arff");
        Instances[] bySensor=new Instances[sensors];
        for(int j=0;j<sensors;j++){
            Instances x,y,z; 
            x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_X_TRAIN");
            y=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Y_TRAIN");
            z=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Z_TRAIN");
            Differences nc=new Differences();
            nc.setOrder(1);
            nc.setAttName("Sensor"+j+"X");
            x=nc.process(x);
            nc.setAttName("Sensor"+j+"Y");
            y=nc.process(y);
            nc.setAttName("Sensor"+j+"Z");
            z=nc.process(z);
            bySensor[j]=new Instances(x);
            bySensor[j].setClassIndex(-1);
            bySensor[j].deleteAttributeAt(bySensor[j].numAttributes()-1); 
            bySensor[j]=Instances.mergeInstances(bySensor[j], y);
            bySensor[j].setClassIndex(-1);
            bySensor[j].deleteAttributeAt(bySensor[j].numAttributes()-1); 
            bySensor[j]=Instances.mergeInstances(bySensor[j], z);
            bySensor[j].setClassIndex(bySensor[j].numAttributes()-1);
            bySensor[j].setRelationName("AALTDSensor"+j);
        }
        Instances all=new Instances(bySensor[0]);
        all.setClassIndex(-1);
        all.deleteAttributeAt(all.numAttributes()-1); 
        for(int j=1;j<sensors-1;j++){
            all=Instances.mergeInstances(all,bySensor[j]);
            all.setClassIndex(-1);
            all.deleteAttributeAt(all.numAttributes()-1); 
        }
        all=Instances.mergeInstances(all,bySensor[sensors-1]);
        all.setClassIndex(all.numAttributes()-1);
        all.setRelationName("AALTD_DIFF");
        normedByDimension.writeString(all.toString());
   }
   
public static void createSeparateDimensionProblems() throws Exception{
        int sensors=8;
        OutFile[] out=new OutFile[3];
        out[0]=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_X_TRAIN.arff");
        out[1]=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_Y_TRAIN.arff");
        out[2]=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_Z_TRAIN.arff");
        Instances byDimension=null;
        String[] dim={"X","Y","Z"};
        for(int i=0;i<3;i++){
            for(int j=0;j<sensors;j++){
                Instances x; 
                x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_"+dim[i]+"_TRAIN");
    //            NormalizeCase nc=new NormalizeCase();
    //            x=nc.process(x);
                if(j==0)
                    byDimension=new Instances(x);
                else{
                    byDimension.setClassIndex(-1);
                    byDimension.deleteAttributeAt(byDimension.numAttributes()-1); 
                    byDimension=Instances.mergeInstances(byDimension, x);
                    byDimension.setClassIndex(byDimension.numAttributes()-1);
                }
            }
            byDimension.setRelationName("AALTD_"+dim[i]+"_ONLY");
            out[i].writeString(byDimension.toString());
        }
   }
   

public static void createNormalisedByDimension() throws Exception{
        int sensors=8;
        OutFile normedByDimension=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTDNormed_TRAIN.arff");
        Instances[] bySensor=new Instances[sensors];
        for(int j=0;j<sensors;j++){
            Instances x,y,z; 
            x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_X_TRAIN");
            y=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Y_TRAIN");
            z=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Z_TRAIN");
            NormalizeCase nc=new NormalizeCase();
            x=nc.process(x);
            y=nc.process(y);
            z=nc.process(z);
            bySensor[j]=new Instances(x);
            bySensor[j].setClassIndex(-1);
            bySensor[j].deleteAttributeAt(bySensor[j].numAttributes()-1); 
            bySensor[j]=Instances.mergeInstances(bySensor[j], y);
            bySensor[j].setClassIndex(-1);
            bySensor[j].deleteAttributeAt(bySensor[j].numAttributes()-1); 
            bySensor[j]=Instances.mergeInstances(bySensor[j], z);
            bySensor[j].setClassIndex(bySensor[j].numAttributes()-1);
            bySensor[j].setRelationName("AALTDSensor"+j);
        }
        Instances all=new Instances(bySensor[0]);
        all.setClassIndex(-1);
        all.deleteAttributeAt(all.numAttributes()-1); 
        for(int j=1;j<sensors-1;j++){
            all=Instances.mergeInstances(all,bySensor[j]);
            all.setClassIndex(-1);
            all.deleteAttributeAt(all.numAttributes()-1); 
        }
        all=Instances.mergeInstances(all,bySensor[sensors-1]);
        all.setClassIndex(all.numAttributes()-1);
        all.setRelationName("AALTD_NORMED");
        normedByDimension.writeString(all.toString());
   }
   
   
   public static void formatChannelProblems() throws Exception{
        int sensors=8;
           for(int j=0;j<sensors;j++){
               OutFile byChannel=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\XYZBySensor\\AALTDSensor"+j+"_XYZ_TRAIN.arff");
               Instances x,y,z; 
                  x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_X_TRAIN");
                  y=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Y_TRAIN");
                  z=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24Sensors\\AALTD"+j+"_Z_TRAIN");
                 NormalizeCase nc=new NormalizeCase();
                 x=nc.process(x);
                 y=nc.process(y);
                 z=nc.process(z);
                  Instances all=new Instances(x);
                  all.setClassIndex(-1);
                  all.deleteAttributeAt(all.numAttributes()-1); 
                  all=Instances.mergeInstances(all, y);
                  all.setClassIndex(-1);
                  all.deleteAttributeAt(all.numAttributes()-1); 
                  all=Instances.mergeInstances(all, z);
                  all.setClassIndex(all.numAttributes()-1);
                  all.setRelationName("AALTDSensor"+j);
                  byChannel.writeLine("% AALTD data by sensor. First 50 readings are the X axis, next 50 the Y and last 50 the Z");
                  
                  byChannel.writeLine(all.toString()); 
           }
   }
   public static void renameAnnoyingAaronAttributeNames(){
        int sensors=8;
       for(int i=0;i<sensors;i++){
           Instances x,y,z; 
           x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24SensorsAnnoyingAaronNames\\AALTD"+i+"_X_TRAIN");
           OutFile rename=new OutFile("C:\\Temp\\AALTD"+i+"_X_TRAIN.arff");
           for(int j=0;j<x.numAttributes()-1;j++)
               x.renameAttribute(j,"Sensor"+i+"X_"+j);
           rename.writeString(x.toString());
           y=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24SensorsAnnoyingAaronNames\\AALTD"+i+"_Y_TRAIN");
           rename=new OutFile("C:\\Temp\\AALTD"+i+"_Y_TRAIN.arff");
           for(int j=0;j<y.numAttributes()-1;j++)
               y.renameAttribute(j,"Sensor"+i+"Y_"+j);
           rename.writeString(y.toString());
           z=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\24SensorsAnnoyingAaronNames\\AALTD"+i+"_Z_TRAIN");
           rename=new OutFile("C:\\Temp\\AALTD"+i+"_Z_TRAIN.arff");
           for(int j=0;j<z.numAttributes()-1;j++)
               z.renameAttribute(j,"Sensor"+i+"Z_"+j);
           rename.writeString(z.toString());
       }
   }
   
   public static double[] findAccuracyPerClass(String trainPredAll, int nosClasses, int cases){
       double[] accPerClass=new double[nosClasses];
       int[] countPerClass=new int[nosClasses];
       int correct=0;
       InFile inf=new InFile(trainPredAll);
       System.out.println(inf.readLine());
       System.out.println(inf.readLine());
       double cvAcc=inf.readDouble();
       for(int i=0;i<cases;i++){   
           int actual=inf.readInt();
           int pred=inf.readInt();
           countPerClass[actual]++;
           if(actual==pred){
               accPerClass[actual]++;
               correct++;
           }
           inf.readLine();
       }
       for(int i=0;i<nosClasses;i++)
           accPerClass[i]/=countPerClass[i];
       for(int i=0;i<nosClasses;i++)
           System.out.println("ACC CLASS "+i+" = "+accPerClass[i]+" count = "+countPerClass[i]);
       System.out.println(" OVERALL ACCURACY ="+correct/(double)cases);
       inf.closeFile();
       return accPerClass;
   }
   
   public static void makePCA(){
       Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_All_XY_TRAIN.arff");
       PCA pca=new PCA(); 
       pca.setVariance(.99);
       all.setClassIndex(-1);
       all.deleteAttributeAt(all.numAttributes()-1);
       Instances trans=pca.transform(all);
       OutFile out=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_PCA_XY_TRAIN.arff");
       out.writeLine(trans.toString());
       
   }
   
   public static void buildCompleteFromSingles(String[] components, String path){        
       Instances data=ClassifierTools.loadData(dataPath+"_TRAIN");
       for(String str: components)
            combineSingleFolds(rootPath+path+"\\"+str+"\\",data.numInstances());        
       OutFile out=new OutFile(rootPath+"COTE_"+path+".csv");
        ensembleCote(rootPath+path+"\\", data,out,components);        
       
   }
   
   public static void buildCompleteFromTrainPredAllFile(String[] components, String path){  
       Instances data=ClassifierTools.loadData(dataPath+"_TRAIN");
       OutFile out=new OutFile(rootPath+"COTE_"+path+".csv");
        ensembleCote(rootPath+path+"\\", data,out,components);        
   }
   

public static void summaryStats() throws Exception{
        String[] axis={"X","Y","Z"};
        int channels=8;
        Instances all=null;
        Instances all2=null;
        int nosChannels=0;
        for(int j=0;j<axis.length;j++){
            Instances[] trainSensor=new Instances[channels];
            Instances[] testSensor=new Instances[channels];
/* LEFT HAND SIDE */        
            for(int i=0;i<channels;i+=2){

                trainSensor[i]=ClassifierTools.loadData(dataPath+i+"_"+axis[j]+"_TRAIN");
                testSensor[i]=ClassifierTools.loadData(dataPath+i+"_"+axis[j]+"_TEST");
                System.out.println(" First att name ="+testSensor[i].attribute(0).name());
//Find mean, min, max
                SummaryStats ss= new SummaryStats();
                SummaryStats ss2= new SummaryStats();
                Instances temp=ss.process(trainSensor[i]);
                Instances temp2=ss2.process(testSensor[i]);
 
                if(i==0 && axis[j].equals(axis[0])){
                    all=new Instances(temp);
                    all2=new Instances(temp2);
                }
                else{
                    all.setClassIndex(-1);
                    all.deleteAttributeAt(all.numAttributes()-1); 
                    all=Instances.mergeInstances(all,temp);
                    all.setClassIndex(all.numAttributes()-1);
                    all2.setClassIndex(-1);
                    all2.deleteAttributeAt(all2.numAttributes()-1); 
                    all2=Instances.mergeInstances(all2,temp2);
                    all2.setClassIndex(all2.numAttributes()-1);
                }
            }

/* RIGHT HAND SIDE      */              
            for(int i=1;i<channels;i+=2){
                trainSensor[i]=ClassifierTools.loadData(dataPath+i+"_"+axis[j]+"_TRAIN");
                testSensor[i]=ClassifierTools.loadData(dataPath+i+"_"+axis[j]+"_TEST");
                
//Find mean, min, max
                SummaryStats ss= new SummaryStats();
                Instances temp=ss.process(trainSensor[i]);                
                Instances temp2=ss.process(testSensor[i]);

                all.setClassIndex(-1);
                all.deleteAttributeAt(all.numAttributes()-1); 
                all=Instances.mergeInstances(all,temp);
                all.setClassIndex(all.numAttributes()-1);
                all2.setClassIndex(-1);
                all2.deleteAttributeAt(all2.numAttributes()-1); 
                all2=Instances.mergeInstances(all2,temp2);
                all2.setClassIndex(all2.numAttributes()-1);
            }
        }

        System.out.println(" Number of attributes ="+(all.numAttributes()-1));
        
//Find out the differences betwenn continguous maxima
        for(int j=0;j<axis.length;j++){
            for(int i=1;i<channels;i++){
                for(Instance ins:all){
                    double peak1=ins.value(j*6*8+(i-1)*6+5);
                    double peak2=ins.value(j*6*8+(i)*6+5);
//                    if(i==1)
//                        System.out.println("Dimension ="+axis[j]+" Peak "+i+" Pos="+(j*6*8+(i)*6+5)+" = "+peak2+" Peak "+(i-1)+" Pos ="+(j*6*8+(i-1)*6+5)+" = "+peak2);
//Overwrite 3rd and 4th moment, dont really need it!
                    double trough1=ins.value(j*6*8+(i-1)*6+4);
                    double trough2=ins.value(j*6*8+(i)*6+4);
                    ins.setValue(j*6*8+(i-1)*6+2, peak2-peak1);
                    ins.setValue(j*6*8+(i-1)*6+3, trough2-trough1);

                }
                all.renameAttribute(j*6*8+(i-1)*6+2, "PeakDiff_"+axis[j]+"_"+i+"_"+(i-1));
                all.renameAttribute(j*6*8+(i-1)*6+3, "TroughDiff"+axis[j]+"_"+i+"_"+(i-1));
            }
            for(int i=1;i<channels;i++){
                for(Instance ins:all2){
                    double peak1=ins.value(j*6*8+(i-1)*6+5);
                    double peak2=ins.value(j*6*8+(i)*6+5);
//                    if(i==1)
//                        System.out.println("Dimension ="+axis[j]+" Peak "+i+" Pos="+(j*6*8+(i)*6+5)+" = "+peak2+" Peak "+(i-1)+" Pos ="+(j*6*8+(i-1)*6+5)+" = "+peak2);
//Overwrite 3rd and 4th moment, dont really need it!
                    double trough1=ins.value(j*6*8+(i-1)*6+4);
                    double trough2=ins.value(j*6*8+(i)*6+4);
                    ins.setValue(j*6*8+(i-1)*6+2, peak2-peak1);
                    ins.setValue(j*6*8+(i-1)*6+3, trough2-trough1);
                }
                all2.renameAttribute(j*6*8+(i-1)*6+2, "PeakDiff_"+axis[j]+"_"+i+"_"+(i-1));
                all2.renameAttribute(j*6*8+(i-1)*6+3, "TroughDiff"+axis[j]+"_"+i+"_"+(i-1));
            }
            
        }
//DELETE LEFT HAND SIDE
        for(int j=0;j<axis.length;j++){
            for(int i=0;i<4;i++){
                for(int k=0;k<6;k++){
                    all.deleteAttributeAt(j*24);
                    all2.deleteAttributeAt(j*24);
                }
            }
        }        
        
        all.setRelationName("SummaryStatsThreeClass");
        all2.setRelationName("SummaryStatsThreeClass");
        OutFile out=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SummaryChannels\\AALTD_ALL_SS_TRAIN.arff");
        out.writeLine(all.toString());
        out=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SummaryChannels\\AALTD_ALL_SS_THREE_CLASS_TRAIN.arff");
        Instances threeClass=new Instances(all);
        for(Instance ins:threeClass)
            if(ins.classValue()!=1 && ins.classValue()!=2)
                ins.setClassValue(0.0);
        out.writeLine(threeClass.toString());
        int count=0;
        while(count<threeClass.numInstances()){
            int c=(int)threeClass.instance(count).classValue();
            if(c!=1 && c!=2)
                threeClass.delete(count);
            else
                count++;
        }
        out=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SummaryChannels\\AALTD_ALL_SS_TWO_CLASS_TRAIN.arff");
        out.writeLine(threeClass.toString());
        System.out.println("Number of channels ="+nosChannels);
        
//Find out the differences betwenn continguous maxima
        for(int j=0;j<axis.length;j++){
            for(int i=1;i<nosChannels;i++){
                for(Instance ins:all2){
                    double peak1=ins.value(j*6*8+(i-1)*6+5);
                    double peak2=ins.value(j*6*8+(i)*6+5);
//                    if(i==1)
//                        System.out.println("Dimenion ="+axis[j]+" Peak "+i+" Pos="+(j*6*8+(i)*6+5)+" = "+peak2+" Peak "+(i-1)+" Pos ="+(j*6*8+(i-1)*6+5)+" = "+peak2);
//Overwrite 3rd and 4th moment, dont really need it!
                    double trough1=ins.value(j*6*8+(i-1)*6+4);
                    double trough2=ins.value(j*6*8+(i)*6+4);
                    ins.setValue(j*6*8+(i-1)*6+2, peak2-peak1);
                    ins.setValue(j*6*8+(i-1)*6+3, trough2-trough1);

                    }
                all2.renameAttribute(j*6*8+(i-1)*6+2, "PeakDifference_"+axis[j]+"_"+i+"_"+(i-1));
                all2.renameAttribute(j*6*8+(i-1)*6+3, "TroughDifference"+axis[j]+"_"+i+"_"+(i-1));
            }
        }
        out=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SummaryChannels\\AALTD_ALL_SS_THREE_CLASS_TEST.arff");
        out.writeLine(all2.toString());
        out=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SummaryChannels\\AALTD_ALL_SS_TWO_CLASS_TEST.arff");
        out.writeLine(all2.toString());
   }
//   public static String rootPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\All Concatenated\\";
//   public static String dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_All_";
   public static String rootPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\All Concatenated\\";
   public static String dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD";
   
   
   public static void hackTheShitOutOfTwoClassResults(){
       Instances data=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AllConcatenated\\AALTD_All_XYZ_OffsetRemoved_TRAIN.arff");
       InFile res=new InFile("C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\SummaryStats\\RotF\\trainPredAll.csv");
       OutFile newRes=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\All Concatenated\\XYZ_OffsetRemoved\\SUMMARY\\trainPredAll.csv");
       newRes.writeLine(res.readLine());
       newRes.writeLine(res.readLine());
       newRes.writeLine(res.readLine());
       String dummy=",0,0,0,0,0,0"; 
       for(Instance ins:data){
           if(ins.classValue()==1 || ins.classValue()==2){
               newRes.writeLine(res.readLine());
           }
           else{
               int actual=(int)ins.classValue();
                      int wrongClass;
               if(actual==0) wrongClass=3;
               else if (actual==0)wrongClass=0;
               else wrongClass=actual+1;
               newRes.writeLine(actual+","+wrongClass+","+dummy);
           }
               
       }
       
   }
   public static void hackTheShitOutOfThreeClassResults(){
       Instances data=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_TRAIN.arff");
       InFile res=new InFile("C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\COTEV1\\SUMMARY\\trainPredAll.csv");
       InFile testRes=new InFile("C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\COTEV1\\SUMMARY\\SummaryTest.csv");
       OutFile newRes=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\COTEV1\\SUMMARY\\trainPredAllHACKED.csv");
       OutFile newTestRes=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\COTEV1\\SUMMARY\\SummaryTestHACKED.csv");
       newRes.writeLine(res.readLine());
       newRes.writeLine(res.readLine());
       newRes.writeLine(res.readLine());
       newTestRes.writeLine(testRes.readLine());
       newTestRes.writeLine(testRes.readLine());
       newTestRes.writeLine(testRes.readLine());
       DecimalFormat df= new DecimalFormat("##.####");
       for(Instance ins:data){
           String line=res.readLine();
           String[] p=line.split(",");
           int act=Integer.parseInt(p[0]);
           int pred=Integer.parseInt(p[1]);
           double[] probs=new double[data.numClasses()];
           for(int i=0;i<probs.length;i++)
               probs[i]=Double.parseDouble(p[i+3]);
//Distribute class 0 across others           
           double split=probs[0]/4;
           probs[0]=probs[3]=probs[4]=probs[5]=split;
           int actual=(int)ins.classValue();//Just to make sure 
           if(pred==0){//Make it wrong
               int wrongClass;
               if(actual==0) wrongClass=3;
               else if (actual==0)wrongClass=0;
               else wrongClass=actual+1;
               newRes.writeString(actual+","+wrongClass+",");
           }
           else
               newRes.writeString(actual+","+pred+",");
           for(int i=0;i<probs.length;i++)
               newRes.writeString(","+df.format(probs[i]));
            newRes.writeString("\n");
       }
        data=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_TEST.arff");
       for(Instance ins:data){
           String line=testRes.readLine();
           String[] p=line.split(",");
           int pred=Integer.parseInt(p[1]);
           double[] probs=new double[data.numClasses()];
           for(int i=0;i<probs.length;i++)
               probs[i]=Double.parseDouble(p[i+3]);
//Distribute class 0 across others           
           double split=probs[0]/4;
           probs[0]=probs[3]=probs[4]=probs[5]=split;
           int actual=(int)ins.classValue();//Just to make sure 
           if(pred!=2 || pred!=3){//Make it wrong
               Random r= new Random();
               actual=r.nextInt(5)+1;
              while(actual==2 || actual == 3)
                   actual=r.nextInt(5)+1;
               newTestRes.writeString("?,"+actual+",");
           }
           else
               newTestRes.writeString("?,"+pred+",");
           for(int i=0;i<probs.length;i++)
               newTestRes.writeString(","+df.format(probs[i]));
            newTestRes.writeString("\n");
       }
       
   }   
   
   public static boolean useProbsToEnsemble=true;
   public static void buildCOTETestPredictions(String path,String[] components){
//Get the CV accs from the train data 
//Any instance train instance set will do for this stage        
       Instances train = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_TRAIN");
       Instances test = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\AALTD_TEST");
       OutFile out=new OutFile(path+"CoteTrainPreds.csv");
       ensembleCote(path,train,out,components);
//Load all the test predictions
       double[] cvAcc=new double[components.length];
       int[][] testPred=new int[components.length][test.numInstances()];
       double[][][] testProbs=new double[components.length][test.numInstances()][train.numClasses()];
       double[][] classCVAcc=new double[components.length][train.numClasses()];
       InFile inf=new InFile(path+"CoteTrainPreds.csv");
       out=new OutFile(path+"CoteTestPreds.csv");
       
       out.writeLine(inf.readLine());
       out.writeLine(inf.readLine());
       String t=inf.readLine();
       out.writeLine(t);
       String [] str=t.split(",");
       int start=2;
       for(int i=0;i<components.length;i++)
                  cvAcc[i]=Double.parseDouble(str[start++]);
       start++;
       for(int i=0;i<components.length;i++){
           for(int j=0;j<train.numClasses();j++)
               classCVAcc[i][j]=Double.parseDouble(str[start++]);
           start++;
       }
       for(int i=0;i<components.length;i++){
           InFile preds=new InFile(path+components[i]+"\\"+components[i]+"Test.csv");
           System.out.println("Processing "+components[i]+" Test File");
           preds.readLine();
           preds.readLine();
           preds.readLine();
           for(int j=0;j<test.numInstances();j++){
               str = preds.readLine().split(",");
               testPred[i][j]=Integer.parseInt(str[1])-1;
               for(int k=0;k<train.numClasses();k++){
                 testProbs[i][j][k]=Double.parseDouble(str[k+3]);
               }
           }
       }
       int power=4;
       for(int j=0;j<test.numInstances();j++){
           double[] weightedVote=new double[train.numClasses()];
           for(int i=0;i<components.length;i++){
                if(useProbsToEnsemble){
                    for(int k=0;k<train.numClasses();k++)
//                        allProbs[n][k]+=Math.pow(cvAcc[i],power)*Double.parseDouble(str[3+k]);
                        weightedVote[k]+=Math.pow(classCVAcc[i][k],power)*testProbs[i][j][k];                    
                }
                else
                    weightedVote[testPred[i][j]]+=classCVAcc[i][testPred[i][j]];                    
               
           }
               
           int coteVote=0;
           for(int k=1;k<train.numClasses();k++){
               if(weightedVote[k]>weightedVote[coteVote])
                   coteVote=k;
           }
           out.writeString("?,"+(coteVote+1)+",");
           for(int k=0;k<train.numClasses();k++)
                out.writeString(","+weightedVote[k]);   
           out.writeString("\n");  
//Get all component weights
       }
      
   }
   public static void main(String[] args) throws Exception {
       String[] components=new String[]{"RISE","EE","TSF","HESCA"};
       
//     buildCOTETestPredictions(path,components);
//hackTheShitOutOfThreeClassResults();
//System.exit(0);
     
       
//      dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\AALTDChallenge\\SplitBySensorAndDimension\\AALTD_";
//       summaryStats();
//hackTheShitOutOfThreeClassResults();
//System.exit(0);
//        formatTwoClassPredictionProblem();
//        
//        System.exit(0);
        
     
//        buildCompleteFromTrainPredAllFile(components,"XYZ");        
//        buildCompleteFromTrainPredAllFile(components,"XY");        
        components=new String[]{"ST","BOSS","TSF","RISE","HESCA","EE","SUMMARY"};
        components=new String[]{"RISE","EE","TSF"};
//        rootPath="C:\\Users\\ajb\\Dropbox\\Results\\AALTDChallenge\\SummaryStats\\";
        String target="COTEV1";
//        String target="XYZ_leftRight_OffsetRemoved";
        useProbsToEnsemble=true;        
        buildCompleteFromSingles(components,target);
//     buildCompleteFromTrainPredAllFile(components,target);
     
       System.out.println("COTE SUMMARY ::::::");
    findAccuracyPerClass(rootPath+"\\COTE_"+target+".csv",6,180);
     buildCOTETestPredictions(rootPath+"\\"+target+"\\",components);
//Delete all zero attributes
/*        Instances data=ClassifierTools.loadData(dataPath+"XY_OffsetRemoved_TRAIN");
        int[] features=InstanceTools.removeConstantTrainAttributes(data);
        for(int i=0;i<features.length;i++)
            System.out.println("Deleted att :"+features[i]);
        OutFile out = new OutFile(dataPath+"XY_OffsetRemoved_TRAIN.arff");
        out.writeLine(data.toString());
 */   
    
    }
}
