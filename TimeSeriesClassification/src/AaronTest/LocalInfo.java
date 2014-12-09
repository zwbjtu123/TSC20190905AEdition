/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AaronTest;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.shapelet.QualityMeasures;

/**
 *
 * @author raj09hxu
 */
public class LocalInfo
{


    public static final String[] ucrTiny ={"ItalyPowerDemand", "SonyAIBORobotSurface"};
    public static final String saveLocation = "";//C:/LocalData/Dropbox/PhD/Data/";

    /**
     *
     * @param dataName
     * @param transform
     * @param qm
     * @return
     * returns the save location, depending on the dataName, transform and quality measure. ADD TEST or TRAIN for each.
     * this ensures, if used, that we have consistent file naming for saving and loading.
     * EG. LocalArea\Dataset\ClassifierName_QualityMeasure
     */
    public static String getSaveLocation(String dataName, Class transform, QualityMeasures.ShapeletQualityChoice qm)
    {
        return saveLocation + dataName + File.separator + transform.getSimpleName() + "_" + qm;
    }

    /**
     *
     * @param fileName
     * @return
     */
    public static Instances[] loadTestAndTrain(String fileName)
    {
        Instances[] dataSet = new Instances[2];
        dataSet[0] = utilities.ClassifierTools.loadData(fileName + "_TRAIN");
        dataSet[0].setClassIndex(dataSet[0].numAttributes()-1);

        dataSet[1] = utilities.ClassifierTools.loadData(fileName + "_TEST");
        dataSet[1].setClassIndex(dataSet[1].numAttributes()-1);
        return dataSet;
    }

    public static void LoadData(String dataName, Instances[][][] dataSets, Class[] classList, QualityMeasures.ShapeletQualityChoice[] qualityMeasures)
    {
        //either extract them, or load them from the save file. 
        for (int i = 0; i < classList.length; i++)
        {
            for (int k = 0; k < qualityMeasures.length; k++)
            {
                //get the train and test instances for each dataset from the localInfo.
                String fileName = getSaveLocation(dataName, classList[i], qualityMeasures[k]);
                dataSets[i][k] = loadTestAndTrain(fileName);
            }
        }
    }
    
    /**
     *
     * @param dataSet
     * @param fileName
     */
    public static void saveDataset(Instances dataSet, String fileName)
    {
        try
        {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataSet);
            saver.setFile(new File(fileName + ".arff"));
            saver.writeBatch();
        }
        catch (IOException ex)
        {
            System.out.println("Error saving transformed dataset" + ex);
        }
    }
    
    
    //this will generate your filePath for you. Based on your dataName/Transform/QualityMeasure.
    public static void saveDataset(Instances dataSet, String dataName, Class transform, QualityMeasures.ShapeletQualityChoice qm)
    {
        saveDataset(dataSet, getSaveLocation(dataName, transform, qm));
    }
    
    public static void saveHashMap(HashMap map, String dataName)
    {
        FileWriter out = null;
        try
        {
            File file = new File(saveLocation+dataName+File.separator+ dataName+"_results.csv");
            file.getParentFile().mkdirs();
            out = new FileWriter(file);
            out.append("Classifier_QualityMeasure,Accuracy,\n");
            for (Iterator it = map.keySet().iterator(); it.hasNext();)
            {
                Object key = it.next();
                out.append(key +"," + map.get(key)+",\n");
            }   out.append("\n");
            out.close();
        }
        catch (IOException ex)
        {
            System.out.println("Saving Failed");
        }   
    }

}
