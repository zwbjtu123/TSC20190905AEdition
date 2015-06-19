/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.util.Arrays;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class InstanceTools {
       
    /**
     * Private method to calculate the class distributions of a dataset. Main
     * purpose is for computing shapelet qualities.
     *
     * @param data the input data set that the class distributions are to be
     * derived from
     * @return a TreeMap<Double, Integer> in the form of <Class Value,
     * Frequency>
     */
    public static Map<Double, Integer> createClassDistributions(Instances data)
    {
        Map<Double, Integer> classDistribution = new TreeMap<>();

        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            classValue = it.next().classValue();

            Integer val = classDistribution.get(classValue);

            val = (val != null) ? val + 1 : 1;
            classDistribution.put(classValue, val);
        }
        
        return classDistribution;
    }
    
    public static Map<Double, Instances> createClassInstancesMap(Instances data)
    {
        Map<Double, Instances> instancesMap = new TreeMap<>();
        
        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            Instance inst = it.next();
            classValue = inst.classValue();

            Instances val = instancesMap.get(classValue);

            if(val == null)
                val = new Instances(data, 0);
            
            val.add(inst);
            
            instancesMap.put(classValue, val);
        }
        
        return instancesMap;
        
    }
    
    /** 
     * Modified from Aaron's shapelet resampling code in development.ReasamplingExperiments. Used to resample
     * train and test instances while maintaining original train/test class distributions
     * 
     * @param train Input training instances
     * @param test Input test instances
     * @param seed Used to create reproducible folds by using a consistent seed value
     * @return Instances[] with two elements; [0] is the output training instances, [1] output test instances
     */
    public static Instances[] resampleTrainAndTestInstances(Instances train, Instances test, int seed){
        Instances all = new Instances(train);
        all.addAll(test);

        Map<Double, Integer> trainDistribution = createClassDistributions(train);
        Map<Double, Instances> classBins = createClassInstancesMap(all);
       
        Random r = new Random(seed);

        //empty instances.
        Instances outputTrain = new Instances(all, 0);
        Instances outputTest = new Instances(all, 0);

        Iterator<Double> keys = classBins.keySet().iterator();
        while(keys.hasNext()){
            double classVal = keys.next();
            int occurences = trainDistribution.get(classVal);
            Instances bin = classBins.get(classVal);
            bin.randomize(r); //randomise the bin.

            outputTrain.addAll(bin.subList(0,occurences));//copy the first portion of the bin into the train set
            outputTest.addAll(bin.subList(occurences, bin.size()));//copy the remaining portion of the bin into the test set.
        }

        return new Instances[]{outputTrain,outputTest};
    }
    
    
    //converts a 2d array into a weka Instance.
    public static Instances ToWekaInstances(double[][] data) {
        Instances wekaInstances = null;

        if (data.length <= 0) {
            return wekaInstances;
        }

        int dimRows = data.length;
        int dimColumns = data[0].length;

        //Logging.println("Converting " + dimRows + " and " + dimColumns + " features.", LogLevel.DEBUGGING_LOG);
        // create a list of attributes features + label
        FastVector attributes = new FastVector(dimColumns);
        for (int i = 0; i < dimColumns; i++) {
            attributes.addElement(new Attribute("attr" + String.valueOf(i + 1)));
        }

        // add the attributes 
        wekaInstances = new Instances("", attributes, dimRows);

        // add the values
        for (int i = 0; i < dimRows; i++) {
            double[] instanceValues = new double[dimColumns];

            for (int j = 0; j < dimColumns; j++) {
                instanceValues[j] = data[i][j];
            }

            wekaInstances.add(new DenseInstance(1.0, instanceValues));
        }

        return wekaInstances;
    }

    
    //converts a weka Instances into a 2d array.
    public static double[][] FromWekaInstances(Instances ds) {
        int numFeatures = ds.numAttributes();
        int numInstances = ds.numInstances();

        //Logging.println("Converting " + numInstances + " instances and " + numFeatures + " features.", LogLevel.DEBUGGING_LOG);
        double[][] data = new double[numInstances][numFeatures];

        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numFeatures; j++) {
                data[i][j] = ds.get(i).value(j);
            }
        }

        return data;
    }
    
    //this is for Grabockas train/test set combo matrix. removes the need for appending.
    public static double[][] create2DMatrixFromInstances(Instances train, Instances test) {
        double [][] data = new double[train.numInstances() + test.numInstances()][train.numAttributes()];
        
        for(int i=0; i<train.numInstances(); i++)
        {
            for(int j=0; j<train.numAttributes(); j++)
            {
                data[i][j] = train.get(i).value(j);
                System.out.print(data[i][j] + ",");
            }
            System.out.println();
        }
        
        int index=0;
        for(int i=train.numInstances(); i<train.numInstances()+test.numInstances(); i++)
        {
            for(int j=0; j<test.numAttributes(); j++)
            {
                data[i][j] = test.get(index).value(j);
            }
            ++index;
        }
        
        return data;
    }
}
