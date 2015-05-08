/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AaronTest;

import development.DataSets;
import java.io.File;
import java.util.HashMap;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.shapelet_trees.FStatShapeletTreeWithInfoGain;
import weka.classifiers.trees.shapelet_trees.KruskalWallisTree;
import weka.classifiers.trees.shapelet_trees.MoodsMedianTree;
import weka.classifiers.trees.shapelet_trees.ShapeletTreeClassifier;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import static weka.core.shapelet.QualityMeasures.ShapeletQualityChoice.*;
import weka.filters.timeseries.shapelet_transforms.BalancedClassShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.BinarisedShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ModularShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformDistCaching;

/**
 *
 * @author Aaron
 */
public class ShapeletTransformExperiments
{
    
    //creates the shapelet transoform datasets.
    static Class[] classList =
    {
      /* BalancedClassShapeletTransform.class,*/BinarisedShapeletTransform.class/*FullShapeletTransform.class /*, ShapeletTransformDistCaching.class, ShapeletTransformDistCaching2.class, ShapeletTransformDistCaching.class*/
    };

    static QualityMeasures.ShapeletQualityChoice[] qualityMeasures =
    { 
       INFORMATION_GAIN,/*F_STAT, KRUSKALL_WALLIS, MOODS_MEDIAN*/
    };
    

    public static void initializeShapelet(FullShapeletTransform s, Instances data, QualityMeasures.ShapeletQualityChoice qm)
    {
        //transform from 3 - n, where n is the max length of the series.
        s.setNumberOfShapelets(data.numInstances() * 10);
        int minLength = 3;
        int maxLength = data.numAttributes() - 1;
        s.setShapeletMinAndMax(minLength, maxLength);
        s.setQualityMeasure(qm);
        s.turnOffLog();
    }

    public static Instances[] extractShapelet(File dataName, Class shapeletClass, QualityMeasures.ShapeletQualityChoice qm)
    {
        Instances test = null;
        Instances train;
        FullShapeletTransform s;
        
        Instances[] testAndTrain = new Instances[2];

        String filePath = dataName.toString() + File.separator + dataName.getName();
        System.out.println("FilePath: " + filePath);
        
        //get the train and test instances for each dataset.
        test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

        //get the save location from the static utility class for my local save.
        String outLogFileName = LocalInfo.getSaveLocation(dataName.getName(), shapeletClass, qm);

        try{
            //create our classifier. 
            s = (FullShapeletTransform) shapeletClass.newInstance();
            //init
            initializeShapelet(s, train, qm);
            testAndTrain[0] = s.process(train);
            LocalInfo.saveDataset(testAndTrain[0], outLogFileName + "_TRAIN");

            System.out.println(s.getClass().getSimpleName() + " train opCount" + ShapeletTransform.subseqDistOpCount);

            testAndTrain[1] = s.process(test);
            LocalInfo.saveDataset(testAndTrain[1], outLogFileName + "_TEST");
            
            System.out.println(s.getClass().getSimpleName() + " test opCount" + ShapeletTransform.subseqDistOpCount);
        }
        catch(IllegalAccessException | IllegalArgumentException | InstantiationException e)        
        {
            System.out.println("error: " + e);
        }
        
        return testAndTrain;
    }
    
    public static AbstractClassifier shapeletTreeBuilder(QualityMeasures.ShapeletQualityChoice qm, int minLength, int maxLength) throws Exception
    {       
        switch(qm)
        {
            case INFORMATION_GAIN:
            {
                ShapeletTreeClassifier c = new ShapeletTreeClassifier("infoTree.txt");
                c.setShapeletMinMaxLength( minLength, maxLength);
                return c;
            }
            case KRUSKALL_WALLIS:
            {
                KruskalWallisTree c =  new KruskalWallisTree("kwTree.txt");
                c.setShapeletMinMaxLength( minLength, maxLength);
                return c;
            }
            case MOODS_MEDIAN:
            {
                MoodsMedianTree c =  new MoodsMedianTree("mmTree.txt");
                c.setShapeletMinMaxLength( minLength, maxLength);
                return c;
            }
            case F_STAT:
            {
                FStatShapeletTreeWithInfoGain c =  new FStatShapeletTreeWithInfoGain("fStatTree.txt");
                c.setShapeletMinMaxLength( minLength, maxLength);
                return c;
            }
        }
        return null;
    }

    //assume they're the same length.
    public static boolean AreInstancesEqual(Instances a, Instances b)
    {
        for (int i = 0; i < a.size(); i++)
        {
            double distance = a.get(i).value(0) - b.get(i).value(0);

            if (distance != 0)
            {
                return false;
            }
        }
        return true;
    }


    public static void CreateData(File dataName, Instances[][][] dataSets)
    {
        //for each classifier pass in the class name and construct it generically in the sub function.
        for (int i = 0; i < classList.length; i++)
        {
            for (int j = 0; j < qualityMeasures.length; j++)
            {
               dataSets[i][j] = extractShapelet(dataName, classList[i], qualityMeasures[j]);
            }
        }
    }

    public static void trainAndTest(String dataName, Instances[][][] dataSets)
    {
        HashMap<String, Double> results = new HashMap<>();
        
        //Use appropriate shapelet tree depending on distance measure used. so FStatShapeletTreeWithInfoGain for fstat etc.
        Classifier c = null;
        
        for(int i =0; i<dataSets.length; i++)
        {
            for(int j=0; j<dataSets[i].length; j++)
            {
                try
                {
                    //build the classifier based on the Quality Measure.
                    c = shapeletTreeBuilder(qualityMeasures[j], 3, dataSets[i][j][0].numAttributes() - 1 );
                    c.buildClassifier(dataSets[i][j][0]);
                    
                    double average = utilities.ClassifierTools.accuracy(dataSets[i][j][1], c);
                
                    String name = classList[i].getSimpleName()+"_"+qualityMeasures[j];

                    results.put(name, average);
                }
                catch (Exception ex)
                {
                    System.out.println("Failed to build classifier " + ex);
                }
            }
        }
        
        //save results
        LocalInfo.saveHashMap(results, dataName);
    }
    
    public static void testDataSet(File dataName, boolean create)
    {
        //[transformType][qualityMeasure][TRAIN/TEST]
        Instances[][][] dataSets = new Instances[classList.length][qualityMeasures.length][2];

        //either create or load it
        if(create)
        {
            CreateData(dataName, dataSets);
        }
        else
        {
            LocalInfo.LoadData(dataName.getName(),dataSets, classList, qualityMeasures);
        }
        
        //trainAndTest(dataName.getName(), dataSets);
    }

    public static void main(String[] args)
    {
                
        String folder = "75 Data sets for Elastic Ensemble DAMI Paper";
        
        //for (String dataSet : DataSets.ucrSmall)
        {
            File f = new File(folder+File.separator+"SonyAIBORobotSurface");
            testDataSet(f, true);
        }

        
        //File f = new File(folder+File.separator+"Adiac");
            //testDataSet(f, true);
        
        /*Instances[] shapelet1, shapelet2;
        String dataSet = LocalInfo.ucrTiny[1];
        QualityMeasures.ShapeletQualityChoice qm = F_STAT;
        
        System.out.println("New Code:");
        //shapelet = FullTransformTest(dataSet, new FullShapeletTransform2(), qm);
        //shapelet = FullTransformTest(dataSet, new ShapeletTransform2(), qm);
        
        FullShapeletTransform tf1 = new FullShapeletTransform();
        FullShapeletTransform tf2 = new ShapeletTransformDistCaching();
        
        shapelet1 = FullTransformTest(dataSet, tf1, qm);
        shapelet2 = FullTransformTest(dataSet, tf2, qm);

        for(int i=0; i< tf1.getShapelets().size(); i++)
        {
            int answer = tf1.getShapelets().get(i).compareTo(tf2.getShapelets().get(i));
            System.out.println("answer: " + answer);
        }*/
    }


    public static Instances[] FullTransformTest(String dataName, FullShapeletTransform s1, QualityMeasures.ShapeletQualityChoice qm)
    {
        Instances test;
        Instances train;

        Instances[] testAndTrain = new Instances[2];

        String filePath = DataSets.dropboxPath + dataName + File.separator + dataName;

        //get the train and test instances for each dataset.
        test = utilities.ClassifierTools.loadData(filePath + "_TEST");
        train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

        //get the save location from the localInfo.
        String outLogFileName = LocalInfo.getSaveLocation(dataName, s1.getClass(), qm);
        System.out.println("outLogFileName: " + outLogFileName);

        try
        {
            //create the shapelet filter. 
            initializeShapelet(s1, train, qm);
            long startTime = System.nanoTime();
            testAndTrain[0] = s1.process(train);
            long finishTime = System.nanoTime();
            System.out.println("Time taken:  " + (finishTime - startTime));

            startTime = System.nanoTime();
            testAndTrain[1] = s1.process(test);
            finishTime = System.nanoTime();
            System.out.println("Time taken:  " + (finishTime - startTime));
        }
        catch (IllegalArgumentException ex)
        {
            System.out.println("error: " + ex);
        }
        catch (Exception ex)
        {
            System.out.println("error: " + ex);
        }
        
        return testAndTrain;
    }
}
