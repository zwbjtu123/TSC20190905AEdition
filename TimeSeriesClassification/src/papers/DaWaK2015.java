/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package papers;

import AaronTest.LocalInfo;
import static AaronTest.ShapeletTransformExperiments.testDataSet;
import development.DataSets;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.BinarisedShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;

/**
 *
 * @author raj09hxu
 */
public class DaWaK2015
{
    public static void initializeShapelet(FullShapeletTransform s, Instances data, QualityMeasures.ShapeletQualityChoice qm)
    {
        //transform from 3 - n, where n is the max length of the series.
        s.setNumberOfShapelets(data.numInstances() * 10); //10n
        int minLength = 3;
        int maxLength = data.numAttributes() - 1;
        s.setShapeletMinAndMax(minLength, maxLength);
        s.setQualityMeasure(qm);
        s.turnOffLog();
    }

    public static void extractShapelet(File dataName)
    {
        Instances test = null;
        Instances train;
        BinarisedShapeletTransform transform;
        QualityMeasures.ShapeletQualityChoice qm = QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN;
        
        Instances[] testAndTrain = new Instances[2];

        String filePath = dataName.toString() + File.separator + dataName.getName();
               
        System.out.println("FilePath: " + filePath);
        
        //get the train and test instances for each dataset.
        test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

        //create our classifier. 
        transform = new BinarisedShapeletTransform();
        //get the save location from the static utility class for my local save.
        String outLogFileName = "results" + File.separator + dataName.getName() + File.separator + dataName.getName();

        try{
            //init
            initializeShapelet(transform, train, qm);
            testAndTrain[0] = transform.process(train);
            LocalInfo.saveDataset(testAndTrain[0], outLogFileName + "_TRAIN");

            testAndTrain[1] = transform.process(test);
            LocalInfo.saveDataset(testAndTrain[1], outLogFileName + "_TEST");
        }
        catch(IllegalArgumentException e)        
        {
            System.out.println("error: " + e);
        }
    }
    
    
    
    public static void main(String args[])
    {
        /*String dir = "75 Data sets for Elastic Ensemble DAMI Paper";

        File fDir = new File(dir);
        File[] ds = fDir.listFiles();

        if (args.length >= 1 && args[0] != null)
        {
            int index = Integer.parseInt(args[0]) - 1;

            extractShapelet(ds[index]);
        }*/
        
        classDistributionsInfo();
    }
    
    
    
    
    //create a dataFile about representation of classes with certain values.
    public static void classDistributionsInfo()
    {
        String dir = "75 Data sets for Elastic Ensemble DAMI Paper";

        File fDir = new File(dir);
        File[] ds = fDir.listFiles();

        
        Instances[] testAndTrain = new Instances[2];
        Instances train,test;
        
        PrintWriter outFile =  null;
        try
        {
            outFile = new PrintWriter(new FileWriter("dataDistributions.csv"));
            outFile.printf("fileName,train_dist,test_dist\n");

            for(File dataName : ds)
            {   
                String filePath = dataName.toString() + File.separator + dataName.getName();

                System.out.println("FilePath: " + filePath);

                //get the train and test instances for each dataset.
                test = utilities.ClassifierTools.loadData(filePath + "_TEST");
                train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");


                TreeMap<Double, Integer> trainDist = FullShapeletTransform.getClassDistributions(train);
                TreeMap<Double, Integer> testDist = FullShapeletTransform.getClassDistributions(test);
                outFile.printf("%s,%s,%s\n", dataName.getName(), trainDist.toString().replace('{', '\0').replace('}', '\0'), testDist.toString().replace('{', '\0').replace('}', '\0'));
            }
        }
        catch (IOException ex)
        {
            System.out.println("Exception: " + ex);
        }
        
        if(outFile != null)
            outFile.close();
        
        
    }
}
