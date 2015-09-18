/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.class_distributions;

import java.util.ListIterator;
import java.util.Set;
import java.util.TreeMap;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class TreeSetClassDistribution extends ClassDistribution{

    TreeMap<Double,Integer> classDistribution;
    
    public TreeSetClassDistribution(Instances data) {
        
        classDistribution = new TreeMap<>();
        
        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            classValue = it.next().classValue();

            Integer val = classDistribution.get(classValue);

            val = (val != null) ? val + 1 : 1;
            classDistribution.put(classValue, val);
        }
    }
    
    public TreeSetClassDistribution(int size) {
        classDistribution = new TreeMap<>();
    }
    
    //clones the object - TODO: need to fix this using a proper keyset.
    public TreeSetClassDistribution(ClassDistribution in){
        
        /*//copy over the data.
        classDistribution = new TreeMap<>();
        for(int i=0; i<in.size(); i++)
        {
            classDistribution[i] = in.get(i);
        }*/
    }

    @Override
    public int get(double classValue) {
        return classDistribution.get(classValue);
    }

    @Override
    public void put(double classValue, int value) {
       classDistribution.put(classValue, value);
    }

    @Override
    public int size() {
        return classDistribution.size();
    }

    @Override
    public int get(int accessValue) {
        return classDistribution.get((double) accessValue);
    }
    
    public Set<Double> keySet()
    {
        return classDistribution.keySet();
    }
    
}
