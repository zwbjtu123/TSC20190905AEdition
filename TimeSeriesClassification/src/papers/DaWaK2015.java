/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package papers;

import AaronTest.LocalInfo;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;
import java.util.Scanner;
import static utilities.InstanceTools.createClassDistributions;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;

/**
 *
 * @author raj09hxu
 */
public class DaWaK2015
{
    
    //each dataSet has its own tuning parameters for length.

    public static void initializeShapelet(FullShapeletTransform s, int numberOfShapelets, QualityMeasures.ShapeletQualityChoice qm, int min, int max)
    {
        //transform from 3 - n, where n is the max length of the series.
        s.setNumberOfShapelets(numberOfShapelets); //10n
        int minLength = min;
        int maxLength = max;
        s.setShapeletMinAndMax(minLength, maxLength);
        s.setQualityMeasure(qm);
        s.turnOffLog();
    }

    public static void extractShapelet(dataParams dm, FullShapeletTransform transform)
    {
        Instances test = null;
        Instances train;
        
        
        QualityMeasures.ShapeletQualityChoice qm = QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN;

        Instances[] testAndTrain = new Instances[2];

        String filePath = "75 Data sets for Elastic Ensemble DAMI Paper"+File.separator+dm.fileName + File.separator + dm.fileName;

        System.out.println("FilePath: " + filePath);

        //get the train and test instances for each dataset.
        test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

        //get the save location from the static utility class for my local save.
        String outLogFileName = transform.getClass().getSimpleName() + "_oldDist" + File.separator + dm.fileName + File.separator + dm.fileName;
        
        transform.useCandidatePruning();
        transform.setLogFileName(outLogFileName);
        
        try
        {
            //init
            initializeShapelet(transform, (train.numInstances()*10), qm, dm.min, dm.max);
            testAndTrain[0] = transform.process(train);
            LocalInfo.saveDataset(testAndTrain[0], outLogFileName + "_TRAIN");

            
            //long opCount = ShapeletTransform.subseqDistOpCount;
            //System.out.println(transform.getClass().getSimpleName() + " train opCount\t" + ShapeletTransform.subseqDistOpCount);
            
            testAndTrain[1] = transform.process(test);
            LocalInfo.saveDataset(testAndTrain[1], outLogFileName + "_TEST");
            
            //System.out.println(transform.getClass().getSimpleName() + " test opCount\t" + (ShapeletTransform.subseqDistOpCount-opCount));
            
            
        }
        catch (IllegalArgumentException e)
        {
            System.out.println("error: " + e);
        }
    }

    public static void main(String args[])
    {
        int index = Integer.parseInt(args[0]) - 1;
        //buildDataSets(index);
        shapeletAccuracies(index);
    }
    
    public static void buildDataSets(int i)
    {
        String dir = "75 Data sets for Elastic Ensemble DAMI Paper";
        
        File fDir = new File(dir);
        final File[] ds = fDir.listFiles();
        
        ArrayList<dataParams> data = buildParamsArray();
        
        //create our classifier timing experiments are embedded
        FullShapeletTransform st = new FullShapeletTransform();
        st.setClassValue(new BinarisedClassValue());
        
        extractShapelet(data.get(i), st);  
        //extractShapelet(data.get(i), new ShapeletTransform());  
    }
    
    private static ArrayList<dataParams> buildParamsArray()
    {
        try
        {
            File f = new File("params.csv");

            Scanner sc = new Scanner(f);
            sc.useDelimiter(",");
            
            ArrayList<dataParams> dp = new ArrayList<>();
            
            //skip header
            sc.nextLine();
            
            while(sc.hasNextLine())
            {
                String fields[] = sc.nextLine().split(",");
                //System.out.println(sc.next()+ " " + sc.nextInt() + " " + sc.nextInt());
                dp.add(new dataParams(fields[0], Integer.parseInt(fields[1]), Integer.parseInt(fields[2])));
            }
        
            return dp;
        }
        catch(FileNotFoundException ex)
        {
            System.out.println("Exception: " + ex);
        }
        return null;
    }
    
    public static void shapeletAccuracies(int i)
    {
        String dir = "results1";
        
        File fDir = new File(dir);
        final File[] ds = fDir.listFiles();
        binarisedShapeletsAccuracies(ds[i]);  
    }
    
    public static void binarisedShapeletsAccuracies(File d)
    {
        try
        {
            File f = new File(d.getName()+"_accuracy.csv");
            PrintWriter outFile = new PrintWriter(new FileWriter(f));

            double accuracy = weightedEnsembleAccuracy(d);

            System.out.printf("%s,%f\n", d.getName(), accuracy);

            outFile.printf("%s,%f\n", d.getName(), accuracy);

            outFile.close();
        }
        catch (IOException ex)
        {
            System.out.println("IOException " + ex);
        }
    }


    public static double weightedEnsembleAccuracy(File dataSets)
    {
        try
        {
            Instances train, test;
            String filePath = dataSets.toString() + File.separator + dataSets.getName();

            //get the train and test instances for each dataset.
            test = utilities.ClassifierTools.loadData(filePath + "_TEST");
            train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

            System.out.println("loaded FilePath: " + filePath);
            
            //build the elastic Ensemble on our training data.
            WeightedEnsemble we = new WeightedEnsemble();
            we.buildClassifier(train);
            return utilities.ClassifierTools.accuracy(test, we);
        }
        catch (Exception ex)
        {
            System.out.println("Classifier exception: " + ex);
        }

        return 0;
    }
    
        //create a dataFile about representation of classes with certain values.
    public static void classDistributionsInfo()
    {
        //String dir = "75 Data sets for Elastic Ensemble DAMI Paper";
        String dir = "C:\\LocalData\\BinarisedShapelets";

        File fDir = new File(dir);
        File[] ds = fDir.listFiles();

        Instances[] testAndTrain = new Instances[2];
        Instances train, test;

        PrintWriter outFile = null;
        try
        {
            outFile = new PrintWriter(new FileWriter("binarisedDataDistributions.csv"));
            outFile.printf("fileName,train_dist,test_dist\n");

            for (File dataName : ds)
            {
                String filePath = dataName.toString() + File.separator + dataName.getName();

                System.out.println("FilePath: " + filePath);

                //get the train and test instances for each dataset.
                test = utilities.ClassifierTools.loadData(filePath + "_TEST");
                train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

                Map<Double, Integer> trainDist = createClassDistributions(train);
                Map<Double, Integer> testDist = createClassDistributions(test);
                outFile.printf("%s,%s,%s\n", dataName.getName(), trainDist.toString().replace('{', '\0').replace('}', '\0'), testDist.toString().replace('{', '\0').replace('}', '\0'));
            }
        }
        catch (IOException ex)
        {
            System.out.println("Exception: " + ex);
        }

        if (outFile != null)
        {
            outFile.close();
        }

    }
    
    public static class dataParams
    {
        public String fileName;
        public int min;
        public int max;
        
        public dataParams(String fn, int mn, int mx) 
        {
            fileName = fn;
            min = mn;
            max = mx;
        }
    }

}
