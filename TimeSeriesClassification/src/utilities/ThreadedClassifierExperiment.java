/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package utilities;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author ajb
 */
public class ThreadedClassifierExperiment implements Runnable{
    Instances train;
    Instances test;
    Classifier c;
    double testAccuracy;
    SimpleBatchFilter filter;
    
    public ThreadedClassifierExperiment(Instances tr, Instances te, Classifier cl){
        train=tr;
        test=te;
        c=cl;
        filter=null;
    }
    public void setTransform(SimpleBatchFilter t){
        filter=t;
    }
    public double getTestAccuracy(){ 
        return testAccuracy;
    }
    @Override
    public void run() {
		//Perform a simple experiment,
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
                System.out.println("ACCURACY = "+testAccuracy);

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");
                
                System.exit(0);
        }
    }
    
}
