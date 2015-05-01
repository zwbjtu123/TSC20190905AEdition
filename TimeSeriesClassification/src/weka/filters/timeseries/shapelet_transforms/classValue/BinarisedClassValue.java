/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.classValue;

import java.util.Map;
import java.util.TreeMap;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class BinarisedClassValue extends NormalClassValue{

    Map<Double, Map<Double, Integer>> binaryClassDistribution;
    
    @Override
    public void init(Instances inst)
    {
        //this inits the classDistributions.
        super.init(inst);
        binaryClassDistribution = createBinaryDistributions(classDistributions);
    }
    
    @Override
    public Map<Double, Integer> getClassDistributions() {
        return binaryClassDistribution.get(shapeletValue);        
    }

    @Override
    public double getClassValue(Instance in) {
        return in.classValue() == shapeletValue ? 0.0 : 1.0;
    }
    
    
    private Map<Double, Map<Double, Integer>> createBinaryDistributions(Map<Double, Integer> classDistributions)
    {
        Map<Double, Map<Double, Integer>> binaryMapping = new TreeMap<>();
        
        //for each classVal build a binary distribution map.
        for(Double cVal : classDistributions.keySet())
        {
            binaryMapping.put(cVal, binariseDistributions(classDistributions, cVal));
        }
        return binaryMapping;
    }
    
    private static Map<Double, Integer> binariseDistributions(Map<Double, Integer> classDistributions, double shapeletClassVal)
    {
        Map<Double, Integer> binaryDistribution = new TreeMap<>();

        Integer shapeletClassCount = classDistributions.get(shapeletClassVal);
        binaryDistribution.put(0.0, shapeletClassCount);
        
        int sum = 0;
        for(Integer count : classDistributions.values())
        {
            sum += count;
        }
        
        //remove the shapeletsClass count. Rest should be all the other classes.
        sum -= shapeletClassCount; 
        binaryDistribution.put(1.0, sum);
        return binaryDistribution;
    }
}
