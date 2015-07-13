/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import AaronTest.LocalInfo;
import grabocka_reproduction.LearnShapelets;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.*;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;

/**
 *
 * @author raj09hxu
 */
public class ResamplingExperiments {


    private static final String dotdotSlash = ".." + File.separator;
    private static String[] datasets;
    
    public static final int noSamples = 100;

    public static void main(String args[]) {
        
        //ive already 
        //datasets = removeSubArray(DataSets.ucrNames, DataSets.ucrSmall);
        datasets = DataSets.ucrSmall;

        final String ucrLocation         = dotdotSlash + dotdotSlash + "75 Data sets for Elastic Ensemble DAMI Paper";
        final String resampleLocation    = dotdotSlash + dotdotSlash + "resampled data sets";
        final String transformLocation   = dotdotSlash + dotdotSlash + "resampled transforms";
        final String accuraciesLocation  = dotdotSlash + dotdotSlash + "resampled accuracies";
        final String resultsLocation     = dotdotSlash + dotdotSlash + "resampled results";
        
        /*File fDir = new File(ucrLocation);
        final File[] ds = fDir.listFiles();

        for (File d : ds) 
        {
            String fileExtension = File.separator + d.getName() + File.separator + d.getName();
            createResmapleSets(ucrLocation + fileExtension, resampleLocation + fileExtension, );
        }
       */

        //create all the shapelets sets for the small resamples.
        int inputVal    = Integer.parseInt(args[0]) - 1;
        int sampleSize  = Integer.parseInt(args[1]);
        int classifier    = Integer.parseInt(args[2]);

        //1565 / 100 = 15
        //1565 % 100 = 65
        int index = inputVal / sampleSize;
        int fold = inputVal % sampleSize;

        System.out.println("creating resample for " + datasets[index] + " for fold: " + fold);
        
        String fileExtension = File.separator + datasets[index] + File.separator + datasets[index];

        FullShapeletTransform transform = null;
        LearnShapelets ls = null;
        
        Object classObj;
        
        if(classifier == 0)
        {
            transform = new FullShapeletTransform();
            classObj = transform;
        }
        else if(classifier == 1)
        {
            transform = new BalancedClassShapeletTransform();
            transform.setClassValue(new BinarisedClassValue());
            classObj = transform;
        }
        else
        {
            ls = new LearnShapelets();
            classObj = ls;
        }
        
        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + classObj.getClass().getSimpleName() + fileExtension;
        String samplePath       = resampleLocation + fileExtension;
        String transformPath    = transformLocation + classifierDir;
        String accuracyPath     = accuraciesLocation   + classifierDir;
        String resultsPath      = resultsLocation   + classifierDir;
        
        System.out.println(samplePath);
        
        //for shapelets.
        /*if(classifier != 2)
        {
            createShapeletsOnResample(samplePath, transformPath, fold, transform); 

            //save path in this instance is where the transformed data is.
            //createWeightedEnsembleAccuracies(transformPath, accuracyPath, fold);
        }
        //for grabocka.
        else
        {
            createLearnShapeleteAccuracies(samplePath, resultsPath, fold);
        }*/
        
        fileVerifier(transformLocation, classObj);
        //createAndWriteAccuracies(transformPath, resultsPath, sampleSize);
        //createAccuracies(accuracyPath, resultsPath);
        
        /*try {
            collateData(resultsLocation, classObj);
        } catch (IOException ex) {
            System.out.println("IOError");
        }*/
    }
    
    public static String[] removeSubArray(String[] datasets, String[] subDatasets)
    {
        ArrayList<String> list = new ArrayList<>();
        
        
        for(String data : datasets)
        {
            boolean match = false;
            for(String data1 : subDatasets)
            {
                if(data.equalsIgnoreCase(data1))
                {
                    match = true;
                    break;
                }
            }
            if(!match)
                list.add(data);
        }
        
        return list.toArray(new String[list.size()]);
    }
    
    public static void collateData(String resultsLocation, Object classObj) throws IOException
    {
        String resultsPath;
        
        String classifierDir = File.separator + classObj.getClass().getSimpleName();
        
        //create the file.
        File output = new File(resultsLocation  + classifierDir +"results.csv");
        output.createNewFile();
        
        try (PrintWriter pw = new PrintWriter(output)) {
            pw.printf("dataset,accuracy\n");
            
            for (String smallDataset : datasets) {
                String fileExtension = File.separator + smallDataset + File.separator + smallDataset;
                resultsPath = resultsLocation  + classifierDir + fileExtension;
                
                File f = new File(resultsPath+".csv");
                

                //if the file doesn't exist skip it.
                if(!f.exists()) continue;

                System.out.println(f);
                
                Scanner sc  = new Scanner(f);
                
                double avg = 0;
                int i=0;
                
                //skip the header.
                if(!sc.hasNextLine())
                   continue;
                 
                sc.nextLine();
                
                while(sc.hasNextLine())
                {
                    String line = sc.nextLine();
                    //should be size 2.
                    String[] split = line.split(",");
                    
                    double value = Double.parseDouble(split[1]);
                    avg += value;
                    i++;
                }
                
                if(avg != 0 || i != 0)
                    avg /= i;
                
                //write the average and the file to the csv.
                pw.printf("%s,%f\n", smallDataset, avg);
            }
        }
    }
    
    public static void fileVerifier(String transformLocation, Object classObj)
    {
        String list ="";
        
        int sampleSize = 100;
        
        for(int i=1; i < datasets.length * sampleSize; i++)
        {
            int index = i / sampleSize;
            int fold = i % sampleSize;
            
            String fileExtension = File.separator + datasets[index] + File.separator + datasets[index];
            String transformPath = transformLocation + File.separator + classObj.getClass().getSimpleName() + fileExtension;
            
            
            File f = new File(transformPath + fold + "_TRAIN.arff");
            System.out.println(f);
            if(!f.exists())
            {
                list += (i+1) + ",";
            }
        }
        
        System.out.println(list); 
    }
    
    
    public static void createAndWriteAccuracies(String transformPath, String resultsPath, int totalFolds) {
        try {
            //create our accuracy file.
            File f = new File(resultsPath+".csv");
            f.getParentFile().mkdirs();
            f.createNewFile();
            PrintWriter pw = new PrintWriter(f);
            pw.printf("%s,%s\n","fold","accuracy");
            
            for(int fold=0; fold<totalFolds; fold++)
            {
            //get the train and test instances for each dataset.
                Instances test = utilities.ClassifierTools.loadData(transformPath + fold + "_TEST");
                Instances train = utilities.ClassifierTools.loadData(transformPath + fold + "_TRAIN");

                //build the elastic Ensemble on our training data.
                WeightedEnsemble we = new WeightedEnsemble();
                we.setWeightType(WeightedEnsemble.WeightType.EQUAL);
                we.buildClassifier(train);
                double accuracy = utilities.ClassifierTools.accuracy(test, we);

                pw.printf("%d,%f\n",fold,accuracy);

                System.out.println(accuracy);
            }
            
            pw.close();
            
        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }

    //where does the train and test set come from, where do you want to save the resample versions.
    public static void createResmapleSets(String filePath, String savePath) {
        Instances test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        Instances train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
        Instances all = new Instances(train);
        all.addAll(test);

        int i = 0;

        //so sample split 0 is the train and test.
        LocalInfo.saveDataset(train, savePath + i + "_TRAIN");
        LocalInfo.saveDataset(test, savePath + i + "_TEST");

        Map<Double, Integer> trainDistribution = InstanceTools.createClassDistributions(train);
        Map<Double, Instances> classBins = InstanceTools.createClassInstancesMap(all);
       
        //generate 100 folds.
        //start from 1.
        for (i = 1; i < noSamples; i++) {
            Random r = new Random(i);
            
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
             
            LocalInfo.saveDataset(tr, savePath + i + "_TRAIN");
            LocalInfo.saveDataset(te, savePath + i + "_TEST");
        }
    }

    public static void createShapeletsOnResample(String filePath, String savePath, int fold, FullShapeletTransform transform) {
        Instances test, train;
        
        //Wheat/Wheat_TRAIN
        //Wheat/Wheat_TEST
        test = utilities.ClassifierTools.loadData(filePath + fold + "_TEST");
        train = utilities.ClassifierTools.loadData(filePath + fold + "_TRAIN");
        
        //construct shapelet classifiers.
        transform.setNumberOfShapelets(train.numInstances() * 10);
        transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        transform.setLogOutputFile(savePath+fold+"_shapelets.csv");

        //saveLocation/FullShapeletTransform/ItalyPowerDemand/ItalyPowerDemandi_TRAIN
        LocalInfo.saveDataset(transform.process(train), savePath + fold + "_TRAIN");
        LocalInfo.saveDataset(transform.process(test), savePath + fold + "_TEST");

    }
    
    public static void createLearnShapeleteAccuracies(String filePath, String savePath, int fold) {

        try {
            //get the train and test instances for each dataset.
            Instances test = utilities.ClassifierTools.loadData(filePath + fold + "_TEST");
            Instances train = utilities.ClassifierTools.loadData(filePath + fold + "_TRAIN");

            //create our accuracy file.
            File f = new File(savePath + fold + ".csv");
            f.getParentFile().mkdirs();
            f.createNewFile();
            PrintWriter pw = new PrintWriter(f);
            pw.printf("%s,%s,%s,%s\n", "percentageOfSeriesLength", "shapeletLengthScale","lambdaW", "accuracy");
            
            double[] lambdaW = {0.01, 0.1};
            double[] percentageOfSeriesLength = {0.1, 0.2};
            int[] shapeletLengthScale = {2, 3};
            
            int noFolds = 3;
            
            double accuracy = 0;
            LearnShapelets ls;
            for(int i=0; i< lambdaW.length; i++)
            {
                for(int j=0; j<percentageOfSeriesLength.length; j++)
                {
                    for(int k=0; k<shapeletLengthScale.length; k++)
                    {
                        double sumAccuracy=0;
                        //build our test and train sets. for cross-validation.
                        for (int l = 0; l < noFolds; l++) {
                            Instances trainCV = train.trainCV(noFolds, l);
                            Instances testCV = train.testCV(noFolds, l);
                            //build the elastic Ensemble on our training data.
                            ls = new LearnShapelets();
                            //{PercentageOfSeriesLength,shapeletLengthScale, weights}; 
                            ls.percentageOfSeriesLength = percentageOfSeriesLength[j];
                            ls.shapeletLengthScale = shapeletLengthScale[k];
                            ls.lambdaW = lambdaW[i];
                            ls.buildClassifier(trainCV);
                            accuracy = ClassifierTools.accuracy(testCV, ls);
                            sumAccuracy += accuracy;
                            
                            System.out.println(accuracy);
                            pw.printf("%f,%d,%f,%f\n", percentageOfSeriesLength[j],shapeletLengthScale[k],lambdaW[i],accuracy);
                        }
                        
                        pw.printf("%f,%d,%f,%f\n", percentageOfSeriesLength[j],shapeletLengthScale[k],lambdaW[i],sumAccuracy/noFolds);

                        //line space after each set of params.
                        pw.println();
                    }
                }
            }
            
            pw.close();
        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }

    public static void createWeightedEnsembleAccuracies(String filePath, String savePath, int fold) {

        try {
            //get the train and test instances for each dataset.
            Instances test = utilities.ClassifierTools.loadData(filePath + fold + "_TEST");
            Instances train = utilities.ClassifierTools.loadData(filePath + fold + "_TRAIN");

            //build the elastic Ensemble on our training data.
            WeightedEnsemble we = new WeightedEnsemble();
            we.setWeightType(WeightedEnsemble.WeightType.EQUAL);
            we.buildClassifier(train);
            double accuracy = utilities.ClassifierTools.accuracy(test, we);
            
            System.out.println(accuracy);
            
            //create our accuracy file.
            File f = new File(savePath + fold + ".csv");
            f.getParentFile().mkdirs();
            f.createNewFile();
            try (PrintWriter pw = new PrintWriter(f)) {
                pw.printf("%f", accuracy);
            }
        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }
    
    public static void createAccuracies(String filePath, String savePath)
    {
        System.out.println(filePath);
        System.out.println(savePath);
        
        File load = new File(filePath);

        Scanner sc;
        PrintWriter pw;
        File save;
        
        for(File dataSetDir : load.listFiles())
        {          

            try {
                save = new File(savePath+File.separator+dataSetDir.getName()+".csv");
                save.getParentFile().mkdirs();
                save.createNewFile();
                            
                pw = new PrintWriter(save);
                pw.printf("%s,%s\n", "sample","accuracy");
                for(File resampleCSV : dataSetDir.listFiles())
                {
                    sc = new Scanner(resampleCSV);
                    //Extract the resample number, 
                    pw.printf("%s,%s\n", resampleCSV.getName().substring(dataSetDir.getName().length(), resampleCSV.getName().length() - 4), sc.next());
                }
              
            pw.close();
            
            } catch (FileNotFoundException ex) {
                System.out.println("Exception : "+ ex);
            } catch (IOException ex) {
                System.out.println("Exception : "+ ex);
            }
            
        }
    }
}
