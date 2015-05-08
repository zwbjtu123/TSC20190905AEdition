/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import AaronTest.LocalInfo;
import java.io.File;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import utilities.InstanceTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.*;

/**
 *
 * @author raj09hxu
 */
public class ResamplingExperiments {


    private static final String dotdotSlash = ".." + File.separator;


    public static final int noSamples = 100;

    public static void main(String args[]) {
        final String ucrLocation         = dotdotSlash + dotdotSlash + "75 Data sets for Elastic Ensemble DAMI Paper";
        final String resampleLocation    = dotdotSlash + dotdotSlash + "resampled data sets";
        final String transformLocation   = dotdotSlash + dotdotSlash + "resampled transforms";
        final String accuraciesLocation  = dotdotSlash + dotdotSlash + "resampled accuracies";
        
        /*File fDir = new File(ucrLocation);
        final File[] ds = fDir.listFiles();

        for (File d : ds) 
        {
            String fileExtension = File.separator + d.getName() + File.separator + d.getName();
            createResmapleSets(ucrLocation + fileExtension, resampleLocation + fileExtension, );
        }
       */

        //create all the shapelets sets for the small resamples.
        int inputVal = Integer.parseInt(args[0]) - 1;
        int sampleSize = Integer.parseInt(args[1]);

        //1565 / 100 = 15
        //1565 % 100 = 65
        int index = inputVal / sampleSize;
        int fold = inputVal % sampleSize;

        String[] smallDatasets = DataSets.ucrSmall;
        System.out.println("creating resample for " + smallDatasets[index] + " for fold: " + fold);
        
        String fileExtension = File.separator + smallDatasets[index] + File.separator + smallDatasets[index];

        FullShapeletTransform transform = new FullShapeletTransform();
        //BinarisedShapeletTransform transform = new BinarisedShapeletTransform();

        //get the loadLocation of the resampled files.
        String classifierDir = File.separator + transform.getClass().getSimpleName() + fileExtension;

        String samplePath       = resampleLocation + fileExtension;
        String transformPath    = transformLocation + classifierDir;
        String accuracyPath     = accuraciesLocation   + classifierDir;

        //createShapeletsOnResample(samplePath, transformPath, fold, transform);            
        
        //save path in this instance is where the transformed data is.
        createWeightedEnsembleAccuracies(transformPath, accuracyPath, fold);
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
        int[] minAndMax = null;

        test = utilities.ClassifierTools.loadData(filePath + fold + "_TEST");
        train = utilities.ClassifierTools.loadData(filePath + fold + "_TRAIN");

        try {
            //estimate min max of shapelet.
            minAndMax = ShapeletTransformFactory.estimateMinAndMax(train, transform.getClass().newInstance());

            //construct shapelet classifiers.
            //transform = new FullShapeletTransform(train.numInstances()*10,minAndMax[0], minAndMax[1], QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
            transform.setNumberOfShapelets(train.numInstances() * 10);
            transform.setShapeletMinAndMax(minAndMax[0], minAndMax[1]);
            transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
            transform.turnOffLog();

            //saveLocation/FullShapeletTransform/ItalyPowerDemand/ItalyPowerDemandi_TRAIN
            LocalInfo.saveDataset(transform.process(train), savePath + fold + "_TRAIN");
            LocalInfo.saveDataset(transform.process(test), savePath + fold + "_TEST");
        } catch (InstantiationException | IllegalAccessException ex) {
            System.out.println("Error: " + ex);
        }
    }

    public static void createWeightedEnsembleAccuracies(String filePath, String savePath, int fold) {

        try {
            //get the train and test instances for each dataset.
            Instances test = utilities.ClassifierTools.loadData(filePath + fold + "_TEST");
            Instances train = utilities.ClassifierTools.loadData(filePath + fold + "_TRAIN");

            //build the elastic Ensemble on our training data.
            WeightedEnsemble we = new WeightedEnsemble();
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

}
