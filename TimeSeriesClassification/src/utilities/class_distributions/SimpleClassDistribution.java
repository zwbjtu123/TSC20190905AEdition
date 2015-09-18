/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.class_distributions;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class SimpleClassDistribution extends ClassDistribution {
    private final int[] classDistribution;

    public SimpleClassDistribution(Instances data) {
        
        classDistribution = new int[data.numClasses()];
        
        for (Instance data1 : data) {
            int thisClassVal = (int) data1.classValue();
            classDistribution[thisClassVal]++;
        }
    }
    
    //clones the object 
    public SimpleClassDistribution(ClassDistribution in){
        
        //copy over the data.
        classDistribution = new int[in.size()];
        for(int i=0; i<in.size(); i++)
        {
            classDistribution[i] = in.get(i);
        }
    }
    
    //creates an empty distribution of specified size.
    public SimpleClassDistribution(int size)
    {
        classDistribution = new int[size];
    }

    @Override
    public int get(double classValue) {
        return classDistribution[(int)classValue];
    }

    //Use this with caution.
    @Override
    public void put(double classValue, int value) {
        classDistribution[(int)classValue] = value;
    }

    @Override
    public int size() {
        return classDistribution.length;
    }   

    @Override
    public int get(int accessValue) {
        return classDistribution[accessValue];
    }
    
    @Override
    public String toString(){
        String temp = "";
        for(int i=0; i<classDistribution.length; i++){
            temp+="["+i+" "+classDistribution[i]+"] ";
        }
        return temp;
    }
}
