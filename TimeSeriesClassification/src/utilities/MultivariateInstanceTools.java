/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class MultivariateInstanceTools {
    
    
        //given some univariate datastreams, we want to merge them to be interweaved.
        //so given dataset X, Y, Z.
        //X_0,Y_0,Z_0,X_1,.....,Z_m
    
        //Needs more testing.
        public static Instances mergeStreams(String dataset, Instances[] inst, String[] dimChars){
        FastVector atts = new FastVector();
        String name;
        
        Instances firstInst = inst[0];
        int dimensions = inst.length;
        int length = (firstInst.numAttributes()-1)*dimensions;

        
        for (int i = 0; i < length; i++) {
            name = dataset + "_" + dimChars[i%dimensions] + "_" + (i/dimensions);
            atts.addElement(new Attribute(name));
        }
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        FastVector vals = new FastVector(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.addElement(target.value(i));
        }
        atts.addElement(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        //same number of xInstances 
        Instances result = new Instances("AALTD" + dataset, atts, firstInst.numInstances());

        int size = result.numAttributes()-1;
        
        for(int i=0; i< firstInst.numInstances(); i++){
            result.add(new DenseInstance(size+1));
            
            for(int j=0; j<size;){
                for(int k=0; k< dimensions; k++){
                    result.instance(i).setValue(j,inst[k].get(i).value(j/dimensions)); j++;
                }
            }
        }
        
        for (int j = 0; j < result.numInstances(); j++) {
            result.instance(j).setValue(size, firstInst.get(j).classValue());
        }
        
        return result;
    }
    
    
    
    
    //function which returns the seperate channels of a multivariate problem as Instances[].
    public static Instances[] splitMultivariateInstances(Instances multiInstances){
        Instances[] output = new Instances[numChannels(multiInstances)];
        
        int length = channelLength(multiInstances); //all the values + a class value.

        //each channel we want to build an Instances object which contains the data, and the class attribute.
        for(int i=0; i< output.length; i++){
            //construct numeric attributes
            FastVector atts = new FastVector();
            for (int att = 0; att < length; att++) {
                atts.addElement(new Attribute("channel_"+i+"_"+att));
            }
            
            //construct the class values atttribute.
            Attribute target = multiInstances.attribute(multiInstances.classIndex());
            FastVector vals = new FastVector(target.numValues());
            for (int k = 0; k < target.numValues(); k++) {
                vals.addElement(target.value(k));
            }
            atts.addElement(new Attribute(multiInstances.attribute(multiInstances.classIndex()).name(), vals));
            
            output[i] = new Instances(multiInstances.relationName() + "_channel_" + i, atts, multiInstances.numInstances());
            output[i].setClassIndex(length);
            
            //for each Instance in 
            for(int j =0; j< multiInstances.numInstances(); j++){
                
                //add the denseinstance to write too.
                output[i].add(new DenseInstance(length+1));

                //System.out.println(index);
                double [] channel = multiInstances.get(j).relationalValue(0).get(i).toDoubleArray();
                int k=0;
                for(; k<channel.length; k++){
                    output[i].instance(j).setValue(k, channel[k]);
                }
                
                double classVal = multiInstances.get(j).classValue();
                output[i].instance(j).setValue(k, classVal);
            }
        }
        
        return output;
    }

    public static Instance[] splitMultivariateInstance(Instance instance){
        Instance[] output = new Instance[numChannels(instance)];
        for(int i=0; i< output.length; i++){
            output[i] = instance.relationalValue(0).get(i);
        }    
        return output;
    }
    
    
    //this won't include class value.    
    public static double[][] convertMultiInstanceToArrays(Instance[] data){
        double[][] output = new double[data.length][data[0].numAttributes()-1];
        
        for(int i=0; i<output.length; i++){
            //build muiltchannel column.
            output[i] = new double[data[0].numAttributes()-1];
            for(int j=0; j<output[i].length; j++){
                output[i][j] = data[i].value(j);
            }
        }
        return output;
    }
    
    //this won't include class value.
    public static double[][] convertMultiInstanceToTransposedArrays(Instance[] data){ 
        double[][] output = new double[data[0].numAttributes()-1][data.length];
        for(int i=0; i<output.length; i++){
            //build muiltchannel column.
            output[i] = new double[data.length];
            for(int j=0; j<output[i].length; j++){
                output[i][j] = data[j].value(i);
            }
        }
        
        return output;
    }
    
    public static int indexOfRelational(Instances inst, Instances findRelation){
        int index = -1;
        Attribute relationAtt = inst.get(0).attribute(0);
        for(int i=0; i< inst.numInstances(); i++){
            
            if(relationAtt.relation(i).equals(findRelation)){
                index  = i;
                break;
            }
        }
        return index;
    }
    
    public static int numChannels(Instance multiInstance){
        return multiInstance.relationalValue(0).numInstances();
    }
    
    public static int channelLength(Instance multiInstance){
        return multiInstance.relationalValue(0).numAttributes();
    }
    
    public static int numChannels(Instances multiInstances){
        //get the first attribute which we know is 
        return numChannels(multiInstances.firstInstance());
    }
    
    public static int channelLength(Instances multiInstances){
        return channelLength(multiInstances.firstInstance());
    }
    
}
