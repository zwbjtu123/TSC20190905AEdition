/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import AaronTest.LocalInfo;
import static development.TimeSeriesClassification.path;
import java.io.File;
import java.util.Arrays;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.*;

/**
 *
 * @author raj09hxu
 */
public class ResamplingExperiments
{
    
    private static final String saveLocation = "C:\\resampled data sets";
    private static final String loadLocation = "75 Data sets for Elastic Ensemble DAMI Paper";
    private static final String dotdotSlash = ".." + File.separator;
    private static final String shapeletLoadLocation = dotdotSlash + dotdotSlash + "resampled data sets";
    private static final String shapeletSaveLocation = dotdotSlash + dotdotSlash + "resampled transforms";
    
    public static final int noSamples = 100;

    public static void main(String args[])
    {
        File fDir = new File(loadLocation);
        final File[] ds = fDir.listFiles();
        
        //create the datasets.
        for (int i=0; i < ds.length; i++)
            createResampleSets(ds[i].getName());
        
        
        //create all the shapelets sets for the small resamples.
        //int index = Integer.parseInt(args[0]) - 1;
        //String[] smallDatasets = DataSets.ucrSmall;
        //createShapeletsOnResample(smallDatasets[index]);        
    }
    
    public static void createResampleSets(String fileName)
    {
        System.out.println("Creating resmaples for " +fileName);
        
        String fileExtension = File.separator+ fileName + File.separator + fileName;
        
        String filePath = loadLocation+ fileExtension;
        String savePath = saveLocation+ fileExtension;
        
        Instances test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        Instances train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
        Instances all = new Instances(train);
        all.addAll(test);
        
        int testSize = test.numInstances();

        int i=0;
        
        //so sample split 0 is the train and test.
        LocalInfo.saveDataset(train, savePath+i+"_TRAIN");
        LocalInfo.saveDataset(test, savePath+i+"_TEST");
        
        //generate 100 folds.
        //start from 1.
        for(i=1; i< noSamples; i++)
        {
            Random r = new Random(i);
            all.randomize(r);
                    
            //Form new Train/Test split
            Instances tr = new Instances(all);
            Instances te = new Instances(all, 0);
            for (int j = 0; j < testSize; j++)
            {
                te.add(tr.remove(0));
            }
            
            LocalInfo.saveDataset(tr, savePath+i+"_TRAIN");
            LocalInfo.saveDataset(te, savePath+i+"_TEST");
        }
    }

    
    public static void createShapeletsOnResample(String fileName)
    {
        System.out.println("creating resample for "+ fileName);
        String fileExtension = File.separator+ fileName + File.separator + fileName;
         
        //get the loadLocation of the resampled files.
        String filePath = shapeletLoadLocation + fileExtension;
        String savePath = shapeletSaveLocation + File.separator + "FullShapeletTransform" + fileExtension;
        
        FullShapeletTransform transform;
        Instances test, train;
        int[] minAndMax;
        for(int i=0; i<noSamples; i++)
        {
            test  = utilities.ClassifierTools.loadData(filePath + i + "_TEST");
            train = utilities.ClassifierTools.loadData(filePath + i + "_TRAIN");
            
            //estimate min max of shapelet. 
            minAndMax = ShapeletTransformFactory.estimateMinAndMax(train);
            
            System.out.println(Arrays.toString(minAndMax));
            
            //construct shapelet classifiers.
            transform = new FullShapeletTransform(train.numInstances()*10,minAndMax[0], minAndMax[1], QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
            transform.turnOffLog();

            //saveLocation/FullShapeletTransform/ItalyPowerDemand/ItalyPowerDemandi_TRAIN
            LocalInfo.saveDataset(transform.process(train), savePath + i + "_TRAIN");
            LocalInfo.saveDataset(transform.process(test) , savePath + i + "_TEST");
        }
         
    }
    
    
}
