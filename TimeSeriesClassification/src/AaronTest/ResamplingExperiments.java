/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AaronTest;

import static development.TimeSeriesClassification.path;
import java.io.File;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class ResamplingExperiments
{
    
    private static String saveLocation = "C:\\LocalData\\resampled data sets";
    private static String loadLocation = "75 Data sets for Elastic Ensemble DAMI Paper";

    public static void main(String args[])
    {
        File fDir = new File(loadLocation);
        final File[] ds = fDir.listFiles();
        createResmapleSets(ds[56].getName());
    }
    
    public static void createResmapleSets(String fileName)
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

        //generate 100 folds.
        for(int i=0; i<100; i++)
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
            
            LocalInfo.saveDataset(te, savePath+i+"_TRAIN");
            LocalInfo.saveDataset(te, savePath+i+"_TEST");
        }

        
        
        
        //save tr and te.
        
        
        
    }

}
