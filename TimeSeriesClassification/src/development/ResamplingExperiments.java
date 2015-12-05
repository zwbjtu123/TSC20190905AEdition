/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import development.Aaron.LocalInfo;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import tsc_algorithms.FastShapelets;
import tsc_algorithms.LearnShapelets;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.*;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.ShapeletSearch;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;

/**
 *
 * @author raj09hxu
 */
public class ResamplingExperiments {

    private static final String dotdotSlash = ".." + File.separator;
    private static final String ucrLocation = dotdotSlash + dotdotSlash + "Dropbox" + File.separator + "TSC Problems (1)";
    private static final String resampleLocation = dotdotSlash + dotdotSlash + "resampled data sets";
    private static final String transformLocation = dotdotSlash + dotdotSlash + "resampled transforms";
    private static final String accuraciesLocation = dotdotSlash + dotdotSlash + "resampled accuracies";
    private static final String resultsLocation = dotdotSlash + dotdotSlash + "resampled results";
    private static final String serialLocation = dotdotSlash + dotdotSlash + "resampled serial files";
    
    
    //private static final String[] subsample = {"Worms", "WormsTwoClass", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "yoga", "PhalangesOutlinesCorrect", "ScreenType", "SmallKitchenAppliances", "TwoPatterns", "ElectricDevices", "OSULeaf", "Strawberry", "wafer", "Computers", "Earthquakes"};
    //private static final String[] subsample = {"MALLAT","CinCECGtorso","FordA", "FordB", "RefrigerationDevices","StarLightCurves", "UWaveGestureLibraryAll"};
    //private static final String[] subsample = {"NonInvasiveFatalECGThorax1", "NonInvasiveFatalECGThorax2", "Phoneme","Haptics", "ShapesAll", "HandOutlines", "InlineSkate"};

    private static String[] datasets;

    private static String classifierName;
    private static int classifier;

    private static String currentDataSet;

    public static final int noSamples = 100;

    public static void main(String args[]) {
        //ive already 
        //datasets = removeSubArray(DataSets.ucrNames, DataSets.ucrSmall);
        //datasets = DataSets.ucrSmall;
        //we assume certain file structure for cross platform use.
        //arg0 is the dataset name. 
        //arg2 is the fold number. 0-99.
        //arg1 is the algorithm Full is 0 and Binary is 1.
        //We could opt to not have our arguments come from command line and instead run the folds and datasets and algorithms in 3 nested loops.
        
        /*int num = Integer.parseInt(args[0]) - 1;
        int index = num / 100;
        int fold = num % 100;*/

        currentDataSet = DataSets.fileNames[Integer.parseInt(args[0])];

        
        System.out.println(currentDataSet);
        classifier = Integer.parseInt(args[1]); //auto set this Balanced.
        setClassifierName(classifier);

        int fold =0;
        //1-100. we want 0-99. Cluster thing.
        if (args.length >= 3) {
            fold = Integer.parseInt(args[2]) - 1;
        }/*
        
        int currentSeries =0;
        if (args.length >= 4) {
            currentSeries = Integer.parseInt(args[3]) - 1;
        }
        
        boolean complete = false;
        if(args.length >=5){
            complete =  Boolean.parseBoolean(args[4]);
        }*/
        

        //createParameterShapelet(fold);
        
       
        
        //System.out.println(currentDataSet + " " + fold + " " + currentSeries);
        
        /*String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
        String classifierDir = File.separator + classifierName + fileExtension;
        String samplePath = resampleLocation + fileExtension + fold;
        String transformPath = transformLocation + classifierDir + fold;
        Instances train = utilities.ClassifierTools.loadData(samplePath + "_TRAIN");
        GraceFullShapeletTransform.buildGraceBSUB(currentDataSet, train.numInstances(), fold, "long-eth", 8000);
        */
        
        
        //createShapeletsGrace(fold, currentSeries, complete);
        
        
        /*File f = new File(transformLocation + File.separator + classifierName + File.separator + currentDataSet + File.separator+currentDataSet+fold+"_TRAIN.arff");
        if(!f.exists())
            createShapeletsSubsample(fold, 10);
        else
            System.out.println("already transformed");*/
                
                
        
        //createAllResamples();
        /*File f = new File(transformLocation + File.separator + classifierName + File.separator + currentDataSet + File.separator+currentDataSet+fold+"_TRAIN.arff");
        if(!f.exists())
            createShapelets(fold);
        else
            System.out.println("already transformed");*/
        
        createFastShapeletsAccuracies();
        
        
        //createLearnShapeleteAccuracies(fold);
        
        //System.out.println(currentDataSet);
        //createRandomForestOnTransform();
        
        //fileVerifier();
        
        //we need to do this for a specific dataset.
        //createWEOnTransform();
        
        //createAndWriteAccuracies();
        //createAccuracyTable();
        //createAccuracies();
        //collateData(resultsLocation, simpleName);
        //checkStatus();
        
    }

    public static void setClassifierName(int classifier) {
        switch (classifier) {
            case 0:
                classifierName = FullShapeletTransform.class.getSimpleName();
                break;

            case 1:
                classifierName = BalancedClassShapeletTransform.class.getSimpleName();
                break;
            case 2:
                classifierName = LearnShapelets.class.getSimpleName();
                break;
            case 3 :
                classifierName = FastShapelets.class.getSimpleName();
                break;
        }
    }

    public static void createAllResamples() {
        File fDir = new File(ucrLocation);
        final File[] ds = fDir.listFiles();

        for (File d : ds) {
            if(d.isHidden() || !d.isDirectory()) continue;
            
            System.out.println(d);
            
            String fileExtension = File.separator + d.getName() + File.separator + d.getName();
            createResmapleSets(ucrLocation + fileExtension, resampleLocation + fileExtension);
        }
    }

    public static void createShapelets(int fold) {
        FullShapeletTransform transform = null;

        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
        String classifierDir = File.separator + classifierName + fileExtension;
        String samplePath = resampleLocation + fileExtension + fold;
        String transformPath = transformLocation + classifierDir + fold;
        String serialiseName = classifierName+"_"+currentDataSet+fold+".ser";
        
        
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serialiseName));
            transform = (FullShapeletTransform) ois.readObject();            
            System.out.println("Loaded from file");
        } catch (IOException | ClassNotFoundException ex) {
            System.out.println(ex);

            //if we can't load our transform for whatever reason create a new one.
            if (classifier == 0) {
                transform = new FullShapeletTransform();
            } else {
                //transform = new BalancedClassShapeletTransform();
                //transform.setClassValue(new BinarisedClassValue());
            }
            
            transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
            System.out.println("Create new classifier");
        }
        
        Instances test, train;
        test = utilities.ClassifierTools.loadData(samplePath + "_TEST");
        train = utilities.ClassifierTools.loadData(samplePath + "_TRAIN");

        //construct shapelet classifiers.
        transform.useCandidatePruning();
        transform.setSerialName(serialiseName);
        transform.setNumberOfShapelets(train.numInstances() * 10);
        transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        transform.setLogOutputFile(transformPath + "_shapelets.csv");

        //saveLocation/FullShapeletTransform/ItalyPowerDemand/ItalyPowerDemandi_TRAIN
        LocalInfo.saveDataset(transform.process(train), transformPath + "_TRAIN");
        LocalInfo.saveDataset(transform.process(test), transformPath + "_TEST");
    }

    
        //rewrite
    public static void createShapeletsSubsample(int fold, int resamplePercentage) {
        
        FullShapeletTransform transform = null;

        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
        String classifierDir = File.separator + classifierName + fileExtension;
        String samplePath = resampleLocation + fileExtension + fold;
        String transformPath = transformLocation + classifierDir + fold;
        String serialiseName = classifierName+"_"+currentDataSet+fold+".ser";
        
        
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serialiseName));
            transform = (FullShapeletTransform) ois.readObject();            
            System.out.println("Loaded from file");
        } catch (IOException | ClassNotFoundException ex) {
            System.out.println(ex);

            //if we can't load our transform for whatever reason create a new one.
            if (classifier == 0) {
                transform = new FullShapeletTransform();
            } else {
                //transform = new BalancedClassShapeletTransform();
                //transform.setClassValue(new BinarisedClassValue());
            }
            
            transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
            System.out.println("Create new classifier");
        }
        
        Instances test, train;
        test = utilities.ClassifierTools.loadData(samplePath + "_TEST");
        train = utilities.ClassifierTools.loadData(samplePath + "_TRAIN");
        
        //subSample to 10%
        Instances subSample = InstanceTools.subSample(train, train.numInstances() / resamplePercentage, fold);

        //construct shapelet classifiers.
        transform.useCandidatePruning();
        transform.setSerialName(serialiseName);
        transform.setNumberOfShapelets(subSample.numInstances() * 10);
        transform.setShapeletMinAndMax(3, subSample.numAttributes() - 1);
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        transform.setLogOutputFile(transformPath + "_shapelets.csv");

        //saveLocation/FullShapeletTransform/ItalyPowerDemand/ItalyPowerDemandi_TRAIN
        //do cheap processing.
        transform.process(subSample);
        
        //then do proper datasets.
        LocalInfo.saveDataset(transform.process(train), transformPath + "_TRAIN");
        LocalInfo.saveDataset(transform.process(test), transformPath + "_TEST");
    }
    
    public static void createShapeletsGrace(int fold, int currentSeries, boolean complete){
        
        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
        String classifierDir = File.separator + classifierName + fileExtension;
        String samplePath = resampleLocation + fileExtension + fold;
        String transformPath = transformLocation + classifierDir + fold;
        String serialiseName = classifierName+"_"+currentDataSet+fold+".ser";
        
        Instances test,train;
        test = utilities.ClassifierTools.loadData(samplePath + "_TEST");
        train = utilities.ClassifierTools.loadData(samplePath + "_TRAIN");

        //if we can't load our transform for whatever reason create a new one.
        GraceFullShapeletTransform  transform  = new GraceFullShapeletTransform();
        transform.setClassValue(new BinarisedClassValue());
        transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
        transform.setSeries(currentSeries);
        transform.useCandidatePruning();
        transform.setSerialName(serialiseName);
        transform.setNumberOfShapelets(train.numInstances() * 10);
        transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        transform.setLogOutputFile(transformPath + "_shapelets.csv");

        GraceFullShapeletTransform.buildGraceBSUB(currentDataSet, train.numInstances(), fold, "long-eth", 8000);
        
        /*if(!complete){
            transform.process(train);
        }
        else{
            //saveLocation/FullShapeletTransform/ItalyPowerDemand/ItalyPowerDemandi_TRAIN
            LocalInfo.saveDataset(transform.processFromSubFile(train), transformPath + "_TRAIN");
            LocalInfo.saveDataset(transform.process(test), transformPath + "_TEST");
        }*/
    }
    
    public static String[] removeSubArray(String[] datasets, String[] subDatasets) {
        ArrayList<String> list = new ArrayList<>();

        for (String data : datasets) {
            boolean match = false;
            for (String data1 : subDatasets) {
                if (data.equalsIgnoreCase(data1)) {
                    match = true;
                    break;
                }
            }
            if (!match) {
                list.add(data);
            }
        }

        return list.toArray(new String[list.size()]);
    }

    public static void collateData(String resultsLocation, String simpleName) {
        try {
            String resultsPath;

            String classifierDir = File.separator + simpleName;

            //create the file.
            File output = new File(resultsLocation + classifierDir + "results.csv");
            output.createNewFile();

            PrintWriter pw = new PrintWriter(output);
            pw.printf("dataset,accuracy\n");

            for (String smallDataset : datasets) {
                String fileExtension = File.separator + smallDataset + File.separator + smallDataset;
                resultsPath = resultsLocation + classifierDir + fileExtension;

                File f = new File(resultsPath + "_RF500.csv");

                //if the file doesn't exist skip it.
                if (!f.exists()) {
                    continue;
                }

                System.out.println(f);

                Scanner sc = new Scanner(f);

                double avg = 0;
                int i = 0;

                //skip the header.
                if (!sc.hasNextLine()) {
                    continue;
                }

                sc.nextLine();

                while (sc.hasNextLine()) {
                    String line = sc.nextLine();
                    //should be size 2.
                    String[] split = line.split(",");

                    double value = Double.parseDouble(split[1]);
                    avg += value;
                    i++;
                }

                if (avg != 0 || i != 0) {
                    avg /= i;
                }

                //write the average and the file to the csv.
                pw.printf("%s,%f\n", smallDataset, avg);
            }
        } catch (IOException ex) {
            System.out.println("Exception " + ex);
        }
    }

    public static void fileVerifier() {
        String list = "";

        for (int i = 0; i < noSamples; i++) {
            String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
            String transformPath = transformLocation + File.separator + classifierName + fileExtension;

            File f = new File(transformPath + i + "_TRAIN.arff");
            System.out.println(f);
            if (!f.exists()) {
                list += (i + 1) + ",";
            }
        }

        System.out.println(list);
    }

    public static void createAndWriteAccuracies() {

        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;

        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + classifierName + fileExtension;
        String transformPath = transformLocation + classifierDir;
        String resultsPath = resultsLocation + classifierDir;

        try {
            //create our accuracy file.
            double[] accuracies = new double[noSamples];

            //create accuracies Array.
            for (int fold = 0; fold < noSamples; fold++) {
                //get the train and test instances for each dataset.
                Instances test = utilities.ClassifierTools.loadData(transformPath + fold + "_TEST");
                Instances train = utilities.ClassifierTools.loadData(transformPath + fold + "_TRAIN");

                //build the elastic Ensemble on our training data.
                WeightedEnsemble we = new WeightedEnsemble();
                we.setWeightType(WeightedEnsemble.WeightType.EQUAL);
                we.buildClassifier(train);
                accuracies[fold] = utilities.ClassifierTools.accuracy(test, we);

                System.out.printf("%d,%f\n", fold, accuracies[fold]);
            }
            
            //save Accuracies to File.
            File f = new File(resultsPath + "_WE.csv");
            f.getParentFile().mkdirs();
            f.createNewFile();
            try (PrintWriter pw = new PrintWriter(f)) {
                pw.printf("%s,%s\n", "fold", "accuracy");
                for (int fold = 0; fold < noSamples; fold++) {
                    pw.printf("%d,%f\n", fold, accuracies[fold]);
                }
            }

        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }

    public static void createResmapleSets(String filePath, String savePath) {

        Instances test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        Instances train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
        int i = 0;

        //so sample split 0 is the train and test.
        LocalInfo.saveDataset(train, savePath + i + "_TRAIN");
        LocalInfo.saveDataset(test, savePath + i + "_TEST");
        //generate 100 folds.
        //start from 
        Instances[] trainAndTest;

        for (i = 1; i < noSamples; i++) {
            trainAndTest = InstanceTools.resampleTrainAndTestInstances(train, test, i);
            LocalInfo.saveDataset(trainAndTest[0], savePath + i + "_TRAIN");
            LocalInfo.saveDataset(trainAndTest[1], savePath + i + "_TEST");
        }
    }

    public static void createLearnShapeleteAccuracies(int fold) {

        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;

        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + classifierName + fileExtension;
        String resultsPath = resultsLocation + classifierDir;
        String samplePath = resampleLocation + fileExtension + fold;
        
        try {
            //get the train and test instances for each dataset.
            Instances test = utilities.ClassifierTools.loadData(samplePath + "_TEST");
            Instances train = utilities.ClassifierTools.loadData(samplePath + "_TRAIN");
            
            LearnShapelets ls = new LearnShapelets();
            ls.setSeed(fold);
            ls.buildClassifier(train);
            double accuracy = ClassifierTools.accuracy(test, ls);
            
            System.out.println(accuracy);
            
            /*//create our accuracy file.
            File f = new File(resultsPath + fold + ".csv");
            f.getParentFile().mkdirs();
            f.createNewFile();
            PrintWriter pw = new PrintWriter(f);
            pw.printf("%s,%s,%s,%s\n", "percentageOfSeriesLength", "shapeletLengthScale", "lambdaW", "accuracy");
            for(String s : results)
                pw.print(s);
            pw.close();*/

        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }

    public static void createAccuracies(String filePath, String savePath) {
        System.out.println(filePath);
        System.out.println(savePath);

        File load = new File(filePath);

        Scanner sc;
        PrintWriter pw;
        File save;

        for (File dataSetDir : load.listFiles()) {

            try {
                save = new File(savePath + File.separator + dataSetDir.getName() + ".csv");
                save.getParentFile().mkdirs();
                save.createNewFile();

                pw = new PrintWriter(save);
                pw.printf("%s,%s\n", "sample", "accuracy");
                for (File resampleCSV : dataSetDir.listFiles()) {
                    sc = new Scanner(resampleCSV);
                    //Extract the resample number, 
                    pw.printf("%s,%s\n", resampleCSV.getName().substring(dataSetDir.getName().length(), resampleCSV.getName().length() - 4), sc.next());
                }

                pw.close();

            } catch (FileNotFoundException ex) {
                System.out.println("Exception : " + ex);
            } catch (IOException ex) {
                System.out.println("Exception : " + ex);
            }

        }
    }

    public static void createAccuracyTable() {
        try {
            String directory = resultsLocation + File.separator + classifierName;
            File dir = new File(directory);

            Scanner sc;
            
            PrintWriter pw;
            File save = new File(directory + "_RF500.csv");
            save.getParentFile().mkdirs();
            save.createNewFile();
            pw = new PrintWriter(save);
            
            File[] folders = dir.listFiles();

            for (File dataset : folders) {
                String name = dataset.getName();

                //get the accuracy file.
                File csv = new File(dataset.getAbsolutePath() + File.separator + name + "_RF500.csv");
                System.out.println(name + "------------------------------------------");
                
                sc = new Scanner(csv);
                
                pw.printf("%s", name);
                
                //ignore header info.
                if(sc.hasNextLine()){
                    sc.nextLine();
                }
                
                while(sc.hasNextLine()){
                    String line = sc.nextLine();
                    String[] seperate = line.split(",");
                    if(seperate.length >= 2){
                        System.out.println(Arrays.toString(seperate));
                        pw.printf(",%s", seperate[1]);
                    }
                }
                pw.print("\n");
            }
            
            pw.close();
        } catch (IOException x) {
            System.out.println("IOException:  "+ x);
        }
    }

    public static void createRandomForestOnTransform() {
        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;

        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + classifierName + fileExtension;
        String transformPath = transformLocation + classifierDir;
        String resultsPath = resultsLocation + classifierDir;

        File f = new File(resultsPath + "_RF500.csv");
        //if the file exists, then we've already created this data
        if(f.exists()) return;
        
        try {
            //create our accuracy file.
            double[] accuracies = new double[noSamples];

            //create accuracies Array.
            for (int fold = 0; fold < noSamples; fold++) {
                //get the train and test instances for each dataset.
                Instances test = utilities.ClassifierTools.loadData(transformPath + fold + "_TEST");
                Instances train = utilities.ClassifierTools.loadData(transformPath + fold + "_TRAIN");

                //build the elastic Ensemble on our training data.
                RandomForest rf = new RandomForest();
                rf.setNumTrees(500);
                rf.buildClassifier(train);
                accuracies[fold] = utilities.ClassifierTools.accuracy(test, rf);

                System.out.printf("%d,%f\n", fold, accuracies[fold]);
            }
            
            //save Accuracies to File.

            f.getParentFile().mkdirs();
            f.createNewFile();
            try (PrintWriter pw = new PrintWriter(f)) {
                pw.printf("%s,%s\n", "fold", "accuracy");
                for (int fold = 0; fold < noSamples; fold++) {
                    pw.printf("%d,%f\n", fold, accuracies[fold]);
                }
            }

        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }
       
    public static void checkCompletion(){
        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;

        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + classifierName + fileExtension;
        String transformPath = transformLocation + classifierDir;
    
        File f = null;
        System.out.print("[");
        
        for (int fold = 0; fold < noSamples; fold++) {
                //get the train and test instances for each dataset.
                f = new File(transformPath + fold + "_TRAIN.arff");
                //System.out.println(f.getAbsolutePath());

                if(!f.exists()){
                    System.out.print(fold + ",");
                }
        }
        System.out.print("]\n");
                
    }
      
    public static void createWEOnTransform() {
        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;

        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + classifierName + fileExtension;
        String transformPath = transformLocation + classifierDir;
        String resultsPath = resultsLocation + classifierDir;

        File f = new File(resultsPath + "_WE.csv");
        //if the file exists, then we've already created this data
        if(f.exists()) return;
        
        try {
            //create our accuracy file.
            double[] accuracies = new double[noSamples];

            //create accuracies Array.
            for (int fold = 0; fold < noSamples; fold++) {
                //get the train and test instances for each dataset.
                Instances test = utilities.ClassifierTools.loadData(transformPath + fold + "_TEST");
                Instances train = utilities.ClassifierTools.loadData(transformPath + fold + "_TRAIN");

                //build the elastic Ensemble on our training data.
                WeightedEnsemble we= new WeightedEnsemble();
                we.saveTestPreds(resultsPath+fold+"_TestPreds.csv");
                we.saveTrainCV(resultsPath+fold+"_TrainCV.csv");
                we.buildClassifier(train);
                accuracies[fold] = utilities.ClassifierTools.accuracy(test, we);

                System.out.printf("%d,%f\n", fold, accuracies[fold]);
            }
            
            //save Accuracies to File.

            f.getParentFile().mkdirs();
            f.createNewFile();
            try (PrintWriter pw = new PrintWriter(f)) {
                pw.printf("%s,%s\n", "fold", "accuracy");
                for (int fold = 0; fold < noSamples; fold++) {
                    pw.printf("%d,%f\n", fold, accuracies[fold]);
                }
            }

        } catch (Exception ex) {
            System.out.println("Classifier exception: " + ex);
        }
    }
    
    public static void checkStatus(){
        
        File dir = new File(resampleLocation);
        
        for (File listFile : dir.listFiles()) {
            
            String filePath = resampleLocation + File.separator + listFile.getName() + File.separator + listFile.getName();
            
            System.out.println(listFile.getName());

            BalancedClassShapeletTransform transform;

            System.out.println("Fold Percentage");
            //go through all the datasets and there folds, open up the train file.
            for (int fold = 0; fold < noSamples; fold++) {
                //get the train and test instances for each dataset.
                Instances train = utilities.ClassifierTools.loadData(filePath + fold + "_TRAIN");

                //see if a serial file exists with the same name.
                String serialiseName = serialLocation + File.separator + "BalancedClassShapeletTransform"+"_"+listFile.getName()+fold+".ser";
                try {
                    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serialiseName));
                    transform = (BalancedClassShapeletTransform) ois.readObject();
                    float completion = ((float)transform.dataSet/(float)train.numInstances())*100;
                    if(completion != 100)
                        System.out.println(fold + " " + completion);
                } catch (IOException | ClassNotFoundException ex) {
                }
            }
        }
        //resampleLocation
        //serialLocation
    }
    
    
    public static void createParameterShapelet(int fold){
        String sub = "_sub";
        
        String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
        String classifierDir = File.separator + classifierName + fileExtension;
        String samplePath = resampleLocation + fileExtension + fold;
        String transformPath = transformLocation + classifierDir + fold;
        String serialiseName = classifierName+"_"+currentDataSet+fold+sub+".ser";
        
        FullShapeletTransform transform;
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serialiseName));
            transform = (FullShapeletTransform) ois.readObject();            
            //System.out.println("Loaded from file");
        } catch (IOException | ClassNotFoundException ex) {
           // System.out.println(ex);

            transform = new BalancedClassShapeletTransform();
            transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
            transform.setClassValue(new BinarisedClassValue());
            //System.out.println("Create new classifier");
        }

        Instances test, train;
        test = utilities.ClassifierTools.loadData(samplePath + "_TEST");
        train = utilities.ClassifierTools.loadData(samplePath + "_TRAIN");
        
        System.out.print(currentDataSet);
        //reduce number of instances 
        int numInstances = train.numInstances();
        int numAttributes = train.numAttributes()-1;
   
        transform.setSearchFunction(createSearch(numAttributes));

        //construct shapelet classifiers.
        transform.useCandidatePruning();
        transform.setSerialName(serialiseName);
        transform.setNumberOfShapelets(numInstances);
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        transform.setLogOutputFile(transformPath + "_shapelets.csv");

        //if we end up with less classes than we have instances.
        Instances subSample = subSample(train, fold);
            
            //Instances subSample = InstanceTools.subSample(train, numInstances/sampling, fold);
            if(subSample.numInstances()- 1 < numInstances)
                transform.process(subSample);
            
        //then do proper datasets.
        LocalInfo.saveDataset(transform.process(train), transformPath + sub + "_TRAIN");
        LocalInfo.saveDataset(transform.process(test), transformPath  + sub + "_TEST");
    }
    
    private static ShapeletSearch createSearch(int m){
        int skipLength = 1;
        
        if (m>1500) 
            skipLength =32;
        else if (m>1000)
            skipLength =16;
        else if (m>500) 
            skipLength =8;
        else if (m>250)
            skipLength =4;

        int skipPos = 1;
        if(m>2000)
            skipPos= 8;
        else if(m>1000)
            skipPos= 4;
        else if(m> 500)
            skipPos= 2;

        return new ShapeletSearch(3, m, skipLength, skipPos);
    }
    
    private static Instances subSample(Instances train, int fold){
        int small_sf = InstanceTools.findSmallestClassAmount(train);           
        double proportion = 1;
        int min=25;
        if (small_sf>25){
            proportion = (double)min/(double)small_sf;

            if (proportion < 0.1)
                proportion = 0.1;
        }

        Instances subSample = InstanceTools.subSampleFixedProportion(train, proportion, fold);

        //System.out.print(","+subSample.numInstances()+"\n");
        
        return subSample;
    }

    private static void createFastShapeletsAccuracies() {
        try {
            String fileExtension = File.separator + currentDataSet + File.separator + currentDataSet;
            String classifierDir = File.separator + classifierName + fileExtension;
            String samplePath = resampleLocation + fileExtension;
            String accuracyPath = resultsLocation + classifierDir;
            
            
            double[] accuracies = new double[noSamples];
            for(int i=0; i< noSamples; i++){
                Instances train = utilities.ClassifierTools.loadData(samplePath + i + "_TRAIN");
                Instances test = utilities.ClassifierTools.loadData(samplePath + i +"_TEST");
                
                FastShapelets fs = new FastShapelets();
                
                try {
                    fs.buildClassifier(train);
                    
                    accuracies[i] = utilities.ClassifierTools.accuracy(test, fs);
                    System.out.println("fold " + i + " acc: " + accuracies[i]);
                } catch (Exception ex) {
                    System.out.println("Exception " + ex);
                }
            }

            
            File f  = new File(accuracyPath+".csv");
            f.getParentFile().mkdirs();
            f.createNewFile();
            try (PrintWriter pw = new PrintWriter(f)) {
                pw.printf("%s,%s\n", "fold", "accuracy");
                for (int fold = 0; fold < noSamples; fold++) {
                    pw.printf("%d,%f\n", fold, accuracies[fold]);
                }
            }

        } catch (IOException ex) {
            Logger.getLogger(ResamplingExperiments.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
    }
}
