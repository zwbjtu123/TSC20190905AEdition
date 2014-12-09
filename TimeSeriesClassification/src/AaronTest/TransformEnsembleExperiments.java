/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package AaronTest;

import development.TimeSeriesClassification;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformDistCaching;

/**
 *
 * @author raj09hxu
 */
public class TransformEnsembleExperiments {
    
    
    public static void main(String[] args) {
        
        //for each dataset. Run the test harness.
        for (String ucrTiny : LocalInfo.ucrTiny)
        {
            EnsembleTestHarness(ucrTiny);
        }
 
    }
    
    
    public static void EnsembleTestHarness(String dataName)
    {
        //names of the transforms available
        Class[] classList =
        {
            FullShapeletTransform.class, ShapeletTransform.class, ShapeletTransformDistCaching.class
        };

        //qualityMeasures available.
        QualityMeasures.ShapeletQualityChoice[] qualityMeasures =
        {
            QualityMeasures.ShapeletQualityChoice.F_STAT, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN
        };

        //[transformType][qualityMeasure][TRAIN/TEST]
        Instances[][][] dataSets = new Instances[classList.length][qualityMeasures.length][2];
        
        LocalInfo.LoadData(dataName, dataSets, classList, qualityMeasures);
        
        for (int i = 0; i < classList.length; i++)
        {
            for (int j = 0; j < qualityMeasures.length; j++)
            {
                ensembleTest(dataSets[i][j][0], dataSets[i][j][1]);
            }
        }
        
    }
    
    public static void ensembleTest(Instances train, Instances test)
    {
        //create a bagging ensemble and give it 1-NN as a base.
        Bagging bag = new Bagging();
        bag.setClassifier(new IBk(1));
        
        //train our bag
        try
        {
            bag.buildClassifier(train);
        }
        catch (Exception ex)
        {
            System.out.println("Ex: " + ex);
        }
        
        double average = utilities.ClassifierTools.accuracy(test, bag);
        
        System.out.println("Average correct = " + average);   
    }
}
