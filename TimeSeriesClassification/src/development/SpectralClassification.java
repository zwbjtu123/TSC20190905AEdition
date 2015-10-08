package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import net.sourceforge.sizeof.SizeOf;
import utilities.AttributeSelectionTools;
import utilities.ClassifierTools;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.spectral_distance_functions.*;
import weka.filters.timeseries.*;

/**
 *
 * @author ajb
 */
public class SpectralClassification {

    public static void transformOnCluster(String fileName){
        String directoryName="TSC_Problems";
        String resultName="ChangeTransformedTSC_Problems";
        DecimalFormat df = new DecimalFormat("###.###");
        System.out.println("************** ACF TRANSFORM ON "+fileName+"   *******************");
        Instances test;
        Instances train;
        test=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/"+fileName+"/"+fileName+"_TEST");
        train=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/"+fileName+"/"+fileName+"_TRAIN");			
        OutFile o2=null,o3=null;
        o2=new OutFile(DataSets.clusterPath+resultName+"/PS"+fileName+"/PS"+fileName+"_TRAIN.arff");
        o3=new OutFile(DataSets.clusterPath+resultName+"/PS"+fileName+"/PS"+fileName+"_TEST.arff");

        Instances header;
        try{
            PowerSpectrum ps= new PowerSpectrum();    
            train=ps.process(train);
            test=ps.process(test);
            header=new Instances(train,0);
            o2.writeLine(header.toString());
            o3.writeLine(header.toString());
            for(int j=0;j<train.numInstances();j++)
                o2.writeLine(train.instance(j).toString());
            for(int j=0;j<test.numInstances();j++)
                o3.writeLine(test.instance(j).toString());
        }catch(Exception e){
            System.err.println("Error in transforming");
            System.exit(0);
        }

    }
    
    
    public static void transformAllDataSets(String resultsPath, boolean saveTransforms){
             DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
            System.out.println("************** POWER SPECTRUM TRANSFORM ON ALL*******************");
            ArrayList<String> names=new ArrayList<>();
            String[] fileNames=DataSets.fileNames;
            for(String s:names){
                of.writeString(s+",");
                System.out.print(s+"\t");
            }
                of.writeString("\n");
                System.out.print("\n");
                for(int i=0;i<fileNames.length;i++)
                {
                     Instances test=null;
                     Instances train=null;
                    try{
                           test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileNames[i]+"\\"+fileNames[i]+"_TEST");
                            train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileNames[i]+"\\"+fileNames[i]+"_TRAIN");			
                            OutFile o2=null,o3=null;
                            if(saveTransforms){
                                File f = new File("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS"+fileNames[i]);
                                if(!f.isDirectory())//Test whether directory exists
                                    f.mkdir();
                                o2=new OutFile("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS"+fileNames[i]+"\\PS"+fileNames[i]+"_TRAIN.arff");
                                o3=new OutFile("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS"+fileNames[i]+"\\PS"+fileNames[i]+"_TEST.arff");
                            }
                            System.gc();
                            PowerSpectrum ps= new PowerSpectrum();    
                            System.out.print("Transforming train "+fileNames[i]+" .... ");

                           train=ps.process(train);
                           System.gc();
                            System.out.print("Transforming test "+fileNames[i]+" .... ");
                            test=ps.process(test);
                           System.gc();
                            if(saveTransforms){
    //Need to do this instance by instance to save memory
                                Instances header=new Instances(train,0);
                                o2.writeLine(header.toString());
                                o3.writeLine(header.toString());
                                for(int j=0;j<train.numInstances();j++)
                                    o2.writeLine(train.instance(j).toString());
                                for(int j=0;j<test.numInstances();j++)
                                    o3.writeLine(test.instance(j).toString());
                            }

    //Save results to file
    //Train Classifiers
                            System.out.print(" Classifying ....\n ");
                            Classifier[] c= ClassifierTools.setDefaultSingleClassifiers(names);
                            of.writeString(fileNames[i]+",");
                            for(int j=0;j<c.length;j++){
                                c[j].buildClassifier(train);
                                double a=utilities.ClassifierTools.accuracy(test,c[j]);
                                System.out.print(a+"\t");
                                of.writeString(a+",");
                            }                                    
                                System.out.print("\n");
                                of.writeString("\n");
                    }catch(Exception e){
                            System.out.println(" Error in accuracy ="+e);
                            e.printStackTrace();
                            System.exit(0);
                    } catch(OutOfMemoryError m){
                        System.out.println("OUT OF MEMORY ERROR");
                        m.printStackTrace();
                        Runtime runtime = Runtime.getRuntime();
                        long totalMemory = runtime.totalMemory();
                        long freeMemory = runtime.freeMemory();
                        long maxMemory = runtime.maxMemory();
                        long usedMemory = totalMemory - freeMemory;
    //Summarise memory                    
                        System.out.println(" Total ="+totalMemory+" used = "+usedMemory);
                        System.out.println(" Problem ="+fileNames[i]);
                        try{
                            
                            long testSize=SizeOf.iterativeSizeOf(test);
                            long trainSize=SizeOf.iterativeSizeOf(train);
                            
                            System.out.println("Train set size ="+trainSize);
                            System.out.println("Test set size ="+testSize);                            
                            System.out.println(" USED ="+usedMemory/1000000+" Main Data ="+(testSize+trainSize)/1000000);
     
                            System.exit(0);
                        }catch(Exception e){
                             System.out.println(" Error in memory sizeOf ="+e);
                            e.printStackTrace();
                            System.exit(0);
                            
                        }
                    }
               }     
                                
             }

    public static Classifier[] setDefaultSingleClassifiers(ArrayList<String> names){
            ArrayList<Classifier> sc2=new ArrayList<>();
            kNN k = new kNN(100);
            k.setCrossValidate(true);
            k.normalise(false);
            k.setDistanceFunction(new EuclideanDistance());
            sc2.add(k);
            names.add("kNN_Euclid");
            k = new kNN(100);
            k.setCrossValidate(true);
            k.normalise(true);
            k.setDistanceFunction(new LogNormalisedDistance());
            sc2.add(k);
            names.add("kNN_LogNnorm");
            k = new kNN(100);
            k.setCrossValidate(true);
            k.normalise(true);
            k.setDistanceFunction(new LikelihoodRatioDistance());
            sc2.add(k);
            names.add("kNN_Likelihood Ratio");
            k = new kNN(100);
            k.setCrossValidate(true);
            k.normalise(true);
            k.setDistanceFunction(new KullbackLeiberDistance());
            sc2.add(k);
            names.add("kNN_KL");
         
            
            Classifier c;
            sc2.add(new NaiveBayes());
            names.add("NB");
            sc2.add(new J48());
            names.add("C45");
            c=new SMO();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            ((SMO)c).setKernel(kernel);
            sc2.add(c);
            names.add("SVML");
            c=new SMO();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            ((SMO)c).setKernel(kernel);
            sc2.add(c);
            names.add("SVMQ");
            c=new RandomForest();
            ((RandomForest)c).setNumTrees(200);
            sc2.add(c);
            names.add("RandF200");
            RotationForest f=new RotationForest();
            f.setNumIterations(50);
            sc2.add(f);
            names.add("RotF50");

            Classifier[] sc=new Classifier[sc2.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=sc2.get(i);

            return sc;
    } 

    public static void trainCrossValAccuracies(String resultsPath){
            DecimalFormat df= new DecimalFormat("###.#####");
            OutFile of = new OutFile(resultsPath);
            System.out.println("************** POWER SPECTRUM TRAIN CV ACCURACIES*******************");
            ArrayList<String> names=new ArrayList<>();
            Classifier[] c= ClassifierTools.setDefaultSingleClassifiers(names);
            String[] fileNames=DataSets.fileNames;
            String path="C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS";
            int folds=5;
            System.out.print("\t\t");
            for(String s:names){
                of.writeString(s+",");
                System.out.print(s+"\t");
            }
                of.writeString("\n");
                System.out.print("\n");
                for(int i=0;i<fileNames.length;i++)
                {
                    try{
                        Instances train=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TRAIN");
                        train.randomize(new Random());
    //Train Classifiers
                        System.out.print((i+1)+"  "+fileNames[i]+"\t");
                        c= ClassifierTools.setDefaultSingleClassifiers(names);
                        of.writeString(fileNames[i]+",");
                        for(int j=0;j<c.length;j++){
                            double[][] a=ClassifierTools.crossValidationWithStats(c[j], train, folds);
                            System.out.print(df.format(a[0][0])+"  \t");
                            of.writeString(a[0][0]+",");
                        }                                    
                            System.out.print("\n");
                            of.writeString("\n");
                   }catch(Exception e){
                        System.out.println(" Error in accuracy ="+e);
                        e.printStackTrace();
                        System.exit(0);
                    }              
                }     
                                
             }
    
/**
 * Produce the train/test accuracies for the spectral data sets reduced
 * with a filter. 
 * */
    
    public static void accuraciesWithFilteredData(String resultsPath){
             DecimalFormat df= new DecimalFormat("###.#####");
            OutFile of = new OutFile(resultsPath+"TrainCV.csv");
            OutFile of2 = new OutFile(resultsPath+"Test.csv");
            String[] fileNames=DataSets.fileNames;
            String path="C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS";
            int folds=5;
             ArrayList<String> names=new ArrayList<>();
             AttributeSelectionTools at= new AttributeSelectionTools();
//             at.setEvaluation(new InfoGainAttributeEval());
//             at.setSearch(new Ranker());
             
             Classifier[] c= ClassifierTools.setDefaultSingleClassifiers(names);
            System.out.print("\t\t");
            for(String s:names){
                of.writeString(s+",");
                of2.writeString(s+",");
                System.out.print(s+"\t");
            }
                System.out.print("\n");
                for(int i=0;i<fileNames.length;i++)
                {
                    try{
                        Instances train=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TRAIN");
                        Instances test=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TEST");
                        train.randomize(new Random());
                        
                         
                        System.out.print("\n"+(i+1)+"  "+fileNames[i]+"\t");
                        c= ClassifierTools.setDefaultSingleClassifiers(names);
                        of.writeString("\n"+fileNames[i]+",");
                        of2.writeString("\n"+fileNames[i]+",");
                        Instances reducedTrain=at.filterTrainSet(train);    
                        Instances reducedTest=at.filterTestSet(test);
                        for(int j=0;j<c.length;j++){
    //Estimate the train CV Accuracy
                           double trainAcc=at.crossValidateAccuracy(train,c[j],folds);
                           System.out.print("("+df.format(trainAcc)+",");
                            of.writeString(trainAcc+",");
//Test Accuracy
                            c[j].buildClassifier(reducedTrain);
                            double testAcc=ClassifierTools.accuracy(reducedTest,c[j]);
                            System.out.print(df.format(testAcc)+"  \t");
                            of2.writeString(testAcc+",");
                        }
                        
                        
                   }catch(Exception e){
                        System.out.println(" Error in accuracy ="+e);
                        e.printStackTrace();
                        System.exit(0);
                    }              
                }            
        

    }
    
/**
 * Produce the train/test accuracies for the spectral data sets reduced
 * with a with a wrapper specific to the classifier. 
 * */
 
    public static void accuraciesWithWrapper(String resultsPath){
             DecimalFormat df= new DecimalFormat("###.#####");
            OutFile of = new OutFile(resultsPath+"TrainCV.csv");
            OutFile of2 = new OutFile(resultsPath+"Test.csv");
            String[] fileNames=DataSets.fileNames;
            String path="C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS";
            int folds=5;
             ArrayList<String> names=new ArrayList<>();
             AttributeSelectionTools at= new AttributeSelectionTools();
             WrapperSubsetEval eval=new WrapperSubsetEval();
             at.setEvaluation(eval);
//             at.setSearch(new Ranker());
             Classifier[] c= ClassifierTools.setDefaultSingleClassifiers(names);
            System.out.print("\t\t");
            for(String s:names){
                of.writeString(s+",");
                of2.writeString(s+",");
                System.out.print(s+"\t");
            }
                System.out.print("\n");
                for(int i=fileNames.length-1;i>0;i--)
                {
                    try{
                        Instances train=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TRAIN");
                        Instances test=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TEST");
                        train.randomize(new Random());
                        
                         
                        System.out.print("\n"+(i+1)+"  "+fileNames[i]+"\t");
                        c= ClassifierTools.setDefaultSingleClassifiers(names);
                        of.writeString("\n"+fileNames[i]+",");
                        of2.writeString("\n"+fileNames[i]+",");
                        for(int j=0;j<c.length;j++){
    //Estimate the train CV Accuracy
                         eval.setClassifier(c[j]);   
                          Instances reducedTrain=at.filterTrainSet(train);    
                          Instances reducedTest=at.filterTestSet(test);
//                          double trainAcc=at.crossValidateAccuracy(train,c[j],folds);
//                           System.out.print("("+df.format(trainAcc)+",");
 //                           of.writeString(trainAcc+",");
//Test Accuracy
                            c[j].buildClassifier(reducedTrain);
                            double testAcc=ClassifierTools.accuracy(reducedTest,c[j]);
                            System.out.print(df.format(testAcc)+"  \t");
                            of2.writeString(testAcc+",");
                        }
                        
                        
                   }catch(Exception e){
                        System.out.println(" Error in accuracy ="+e);
                        e.printStackTrace();
                        System.exit(0);
                    }              
                }            
        

    }
    
    
    public static void testPredictionsWithFilter(String resultsPath){
            OutFile of;
            String[] fileNames=DataSets.fileNames;
            String path="C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PS";
             ArrayList<String> names=new ArrayList<>();
             AttributeSelectionTools at= new AttributeSelectionTools();
             Classifier[] c= ClassifierTools.setDefaultSingleClassifiers(names);
                System.out.print("\n");
                for(int i=0;i<fileNames.length;i++)
                {
                    try{
                        Instances train=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TRAIN");
                        Instances test=utilities.ClassifierTools.loadData(path+fileNames[i]+"\\"+"PS"+fileNames[i]+"_TEST");
                        of=new  OutFile(resultsPath+fileNames[i]+"TestPredictions.csv");
                        for(String s:names){
                            of.writeString(s+",");
                        }
                        of.writeString("Actual\n");
                        train.randomize(new Random());
                        System.out.print("\n"+(i+1)+"  "+fileNames[i]+"\t");
                        c= ClassifierTools.setDefaultSingleClassifiers(names);
                        Instances reducedTrain=at.filterTrainSet(train);    
                        Instances reducedTest=at.filterTestSet(test);
                        double[][] preds=new double[c.length][reducedTest.numInstances()];
                        for(int j=0;j<c.length;j++){
                            c[j].buildClassifier(reducedTrain);
                            for(int k=0;k<reducedTest.numInstances();k++)
                                preds[j][k]=c[j].classifyInstance(reducedTest.instance(k));
                        }
                        for(int k=0;k<reducedTest.numInstances();k++){
                            for(int j=0;j<c.length;j++){
                                of.writeString((int)(preds[j][k])+",");
                            }
                            of.writeString((int)(reducedTest.instance(k).classValue())+"\n");
                        }
                        
                   }catch(Exception e){
                        System.out.println(" Error in accuracy ="+e);
                        e.printStackTrace();
                        System.exit(0);
                    }              
                }        
    }


/** Generates a separate output file for each data set.
 * Outputs four predictions and four weights for each instance
 * Predictions are equal, best, weighted and cutoff weighted
 * Weights are the average CV train accuracy for classifiers 
 * advocating the winning class    **/
    public static void spectralEnsembleCombination(String path){

            InFile cv=new InFile(path+"AllCV_TrainAccuracies.csv");
            String first=cv.readLine();
            String[] temp=first.split(",");
            String[] classifierNames=new String[temp.length-1];
            for(int i=0;i<classifierNames.length;i++){
                classifierNames[i]=temp[i+1];
                System.out.println(" Classifier ="+classifierNames[i]);
            }
            String[] fileNames=DataSets.fileNames;
            double[][] cvAccuracy=new double[fileNames.length][classifierNames.length];
            //Load all Train CV accuracies
            for(int i=0;i<fileNames.length;i++){
                String name=cv.readString();
               //Check data allignment
                if(!name.equals(fileNames[i])){
                    System.out.println(" Error, file misallignment");
                    System.out.println("From file ="+name+" from array ="+fileNames[i]);
                    System.exit(0);
                }
               //Read in accuracies
                for(int j=0;j<classifierNames.length;j++){
                    cvAccuracy[i][j]=cv.readDouble();
                }
                System.out.print("\n");
             }
            //Load all Test predictions and form ensemble predictions
            for(int i=0;i<fileNames.length;i++){
                InFile predFile=new InFile(path+"AllPredictions\\"+fileNames[i]+"TestPredictions.csv");
                //Count the number of lines
                int nosCases=predFile.countLines()-1;
                predFile=new InFile(path+"AllPredictions\\"+fileNames[i]+"TestPredictions.csv");
                String header=predFile.readLine();
                String[] c=header.split(",");
                int nosClassifiers=c.length-1;
//Check names etc tally
                if(nosClassifiers!=classifierNames.length){
                     System.out.println(" Error, nos classifiers misallignment");
                    System.out.println("From file ="+fileNames[i]+" nos c="+nosClassifiers+" should be ="+classifierNames.length);
                    System.exit(0);
                }
                for(int k=0;k<classifierNames.length;k++)
                if(!classifierNames[k].equals(c[k])){
                     System.out.println(" Error, classifier name misallignment");
                    System.out.println("From file ="+fileNames[i]+" in file="+classifierNames[k]+" read in ="+c[k]);
                    System.exit(0);
                }
                InstancePredictions[] preds=new InstancePredictions[nosCases];
                InstancePredictions.cvWeights=cvAccuracy[i];
//Find best
                int best=0;
                for(int j=1;j<InstancePredictions.cvWeights.length;j++)
                    if(InstancePredictions.cvWeights[j]>InstancePredictions.cvWeights[best])
                        best=j;
                InstancePredictions.bestClassifierName=classifierNames[best];
                InstancePredictions.bestClassifierIndex=best;
                System.out.println(" Best Classifier for "+fileNames[i]+" is "+InstancePredictions.bestClassifierName+" with CV Acc ="+InstancePredictions.cvWeights[best]);
                for(int j=0;j<nosCases;j++){
                    preds[j]=new InstancePredictions(nosClassifiers);
                    for(int k=0;k<nosClassifiers;k++){
                        preds[j].predicted[k]=predFile.readInt();
                        if(k==InstancePredictions.bestClassifierIndex)
                            preds[j].bestPrediction=preds[j].predicted[k];
                        preds[j].actual=predFile.readInt();
                }
                int[] equalWeight=new int[nosCases];                
                int[] prop=new int[nosCases];
                int[] cutoffProp=new int[nosCases];
                
                int[] predCount;
                double[] weightCount;

 /*               for(int j=0;j<nosCases;j++){
//Horrible hack to avoid storing or calculating the number of classes.
                    //If the problem has more than 200 classes then it will crash
                    int maxNosClasses=200;
                    predCount=new int[maxNosClasses]; 
                    weightCount=new double[maxNosClasses];
                    for(int k=0;k<nosClassifiers;k++){
                        predCount[preds[j][k]]++;
                        weightCount[preds[j][k]]+=cvAccuracy[i][j];
                    }
//Find max value.                    
                    int max=0;
                    int maxWeighted=0;
                    for(int k=0;k<predCount.length;k++){
                        if(predCount[k]>predCount[max])
                            max=k;
                        if(weightCount[k]>weightCount[maxWeighted])
                            maxWeighted=k;
                    }
//Look for duplicates
                    ArrayList<Integer> best=new ArrayList<>();
                    ArrayList<Integer> bestW=new ArrayList<>();
                    for(int k=0;k<predCount.length;k++){
                        if(predCount[k]==predCount[max])
                            best.add(k);
                        if(weightCount[k]==weightCount[maxWeighted])
                            bestW.add(k);
                    }
                    if(best.size()==1)
                        equalWeight[j]=max;
                    else//This will crash if Math.random returns 1
                        equalWeight[j]=best.get((int)(Math.random()*best.size()));
                    if(bestW.size()==1)
                        prop[j]=maxWeighted;
                    else//This will crash if Math.random returns 1
                        prop[j]=bestW.get((int)(Math.random()*bestW.size()));*/
                }
            }
    }
    public static void spectralEnsemble(String resultsPath){
//Set the periodogram specific parameters.         
        ArrayList<String> names = new ArrayList<>();
        Classifier[] c =setDefaultSingleClassifiers(names);
        
        WeightedEnsemble w= new WeightedEnsemble(c,names);
        OutFile of=new OutFile(resultsPath);
        String rootPath="C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\";
        String[] files=DataSets.fileNames;
        
        of.writeString("Data Set,");
        for(String s: names)
            of.writeString(s+",");
        
        for(int i=0;i<files.length;i++){
            of.writeString(files[i]+",");
            Instances train=ClassifierTools.loadData(rootPath+"PS"+files[i]+"\\"+"PS"+files[i]+"_TRAIN");
            Instances test=ClassifierTools.loadData(rootPath+"PS"+files[i]+"\\"+"PS"+files[i]+"_TEST");
            w=new WeightedEnsemble(c,names);
            try{
                w.buildClassifier(train);
                double a=ClassifierTools.accuracy(test, w);
                System.out.println(files[i]+"\t Accuracy ="+a);
            }catch(Exception e){
               System.out.println("Exception = "+e);
               e.printStackTrace();
               System.exit(0);
            }
        }
            
    }
    public static void buildClassifier(String fileName){
        String directoryName="Power Spectrum Transformed TSC Problems";
        Instances test=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/PS"+fileName+"/PS"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/PS"+fileName+"/PS"+fileName+"_TRAIN");			
        ArrayList<String> names= new ArrayList<>();
        Classifier[] c =setDefaultSingleClassifiers(names); 
        WeightedEnsemble    w=new WeightedEnsemble(c,names);
        OutFile of = new OutFile("ps/"+fileName+"PSAcc.csv");
            try{
                w.buildClassifier(train);
                double a=ClassifierTools.accuracy(test, w);
                System.out.println(fileName+"\t Accuracy ="+a);
                of.writeString(fileName+","+a);
            }catch(Exception e){
               System.err.println("Exception = "+e);
               e.printStackTrace();
               System.exit(0);
            }
        
    }
//                 int index=Integer.parseInt(args[0])-1;
  //            System.out.println("Input ="+index);
    //           transformOnCluster(DataSets.fileNames[index]);
     
    public static void mergeFiles(){
        String[] files=DataSets.fileNames;
        OutFile result=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\EnsembleAccuracy\\TestAcc.csv");
        InFile f;
        for(int i=0;i<files.length;i++){
             File file = new File("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\EnsembleAccuracy\\"+files[i]+"PSAcc.csv");
             if(file.exists()){
                f=new InFile("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\EnsembleAccuracy\\"+files[i]+"PSAcc.csv");
                String s=f.readLine();
                result.writeLine(s);
             }
        }
    }
    public static void main(String[] args){
//transformAllDataSets("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\ICDM_TestAccuries.csv", true);
        mergeFiles();
        System.exit(0);
        int index=Integer.parseInt(args[0])-1;
        buildClassifier(DataSets.fileNames[index]);
 //       spectralEnsemble("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\ensemble.csv");
//        spectralEnsembleCombination("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\");        
//        testPredictionsWithFilter("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\AllPredictions\\");
//        accuraciesWithWrapper("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\WrapperSpectrumResults2");
//       accuraciesWithFilteredData("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\FilteredSpectrumResults");
 //       trainCrossValAccuracies("C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\FullSpectrumTrainCVResults.csv");
    }
    
    public static class InstancePredictions{
        static double[] cvWeights;
        static String bestClassifierName;
        static int bestClassifierIndex;
        int actual;
        int[] predicted; 
        int bestPrediction;
        int equalWeightPrediction;
        int cvWeightPrediction;
        int cvCutoffPrediction;
        
        
        InstancePredictions(int nosClassifiers){
            predicted=new int[nosClassifiers];
        }
    }
}
