/*

 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.OutFile;
import tsc_algorithms.SaveableEnsemble;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * @author ajb
 * 
 * conducts 100 resample experiments sequentially
 * Results are written to the file passed in the form
 * 
 * 
 */
public class ClusterClassifierExperiment {
    public static String resultsPath=DataSets.clusterPath+"Results/";
    
//Store and save accuracies and predictions    
    public static double[] resampleExperiment(Instances train, Instances test, Classifier c, int resamples,OutFile of,String preds){

       double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
//Horrible hack here to save files
            if(c instanceof SaveableEnsemble){
                ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+i+".csv",preds+"/internalTestPreds_"+i+".csv");
            }
            foldAcc[i]=singleSampleExperiment(train,test,c,i,preds);

            
            of.writeString(foldAcc[i]+",");
        }            
         return foldAcc;
    }


    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){

        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, sample);
        double acc=0;
        double act,pred;
        OutFile p=new OutFile(preds+"/fold"+sample+".csv");
        
        
        
        try{              
            c.buildClassifier(data[0]);
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=data[1].instance(j).classValue();
                pred=c.classifyInstance(data[1].instance(j));
                if(act==pred)
                    acc++;
                p.writeLine(act+","+pred);
            }
            acc/=data[1].numInstances();
//            of.writeString(foldAcc[i]+",");

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }

}
