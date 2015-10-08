/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package bakeOffExperiments;

import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.DTW_1NN;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class ElasticDistanceMeasures {
    
    public static void debug(){
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        double[] foldAcc = new double[100];
        double act,pred;
        for(int i=0;i<10;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            DTW_1NN dtw=new DTW_1NN();
            dtw.setR(1.0);
            dtw.optimiseWindow(false);
            dtw.buildClassifier(data[0]);
            foldAcc[i]=0;
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=data[1].instance(j).classValue();
                pred=dtw.classifyInstance(data[1].instance(j));
                if(act==pred)
                    foldAcc[i]++;
            }
            foldAcc[i]/=data[1].numInstances();
            System.out.println(" Fold "+i+" acc ="+foldAcc[i]);
        }
//Test Splits        
        
    }
    
    public static void main(String[] args){
        System.out.println(" Code to check we are getting the same resuilts");
        debug();
    }
    
}
