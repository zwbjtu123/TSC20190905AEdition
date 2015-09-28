/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package bakeOffExperiments;

import bakeOffExperiments.Experiments;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.RemoveUseless;

/**
 *
 * @author ajb
 */
public class ThreadedClassifierExperiment extends Thread{
    Instances train;
    Instances test;
    Classifier c;
    double testAccuracy;
    SimpleBatchFilter filter;
    String name;
    int resamples=100;
    public static boolean removeUseless=false;
    
    public ThreadedClassifierExperiment(Instances tr, Instances te, Classifier cl,String n){
        train=tr;
        test=te;
        c=cl;
        filter=null;
        name=n;
    }
    public void setTransform(SimpleBatchFilter t){
        filter=t;
    }
    public double getTestAccuracy(){ 
        return testAccuracy;
    }
    public void singleExperiment(){
        testAccuracy=0;
        double act;
        double pred;
        try{
            if(filter!=null){
                train=filter.process(train);
                test=filter.process(test);
            }

            c.buildClassifier(train);
            for(int i=0;i<test.numInstances();i++)
            {
                    act=test.instance(i).classValue();
                    pred=c.classifyInstance(test.instance(i));
//				System.out.println(" Actual = "+act+" predicted = "+d[i]);
                    if(act==pred)
                            testAccuracy++;
            }
            testAccuracy/=test.numInstances();

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");
                
                System.exit(0);
        }
    }
    
    public void resampleExperiment(){
        double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            double act,pred;
            try{              
                if(filter!=null){
                    data[0]=filter.process(data[0]);
                    data[1]=filter.process(data[1]);
                }
//Filter out attributes with all the same values in the train data. These break BayesNet discretisation
                if(removeUseless){
                    InstanceTools.removeConstantTrainAttributes(train,test);
                }
                
                c.buildClassifier(data[0]);
                foldAcc[i]=0;
                for(int j=0;j<data[1].numInstances();j++)
                {
                    act=data[1].instance(j).classValue();
                    pred=c.classifyInstance(data[1].instance(j));
                    if(act==pred)
                        foldAcc[i]++;
                }
                foldAcc[i]/=data[1].numInstances();

            }catch(Exception e)
            {
                    System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                    e.printStackTrace();
                    System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                    System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                    System.exit(0);
            }
                
            }
        
            synchronized(Experiments.out){
                System.out.println(" finished ="+name);

                Experiments.out.writeString(name+",");
                for(int i=0;i<resamples;i++)
                    Experiments.out.writeString(foldAcc[i]+",");
                Experiments.out.writeString("\n");
            
        }
    }
    
    @Override
    public void run() {
		//Perform a simple experiment,
        if(resamples==1){
            singleExperiment();
        }
        else
            resampleExperiment();
    }
    
}
