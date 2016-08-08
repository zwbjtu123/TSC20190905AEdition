/*
To rerun the WE algorithms by loading the CV accuracies and
rebuilding to get test predictions 
 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.OutFile;
import java.io.File;
import tsc_algorithms.PSACF_Ensemble;
import tsc_algorithms.PS_Ensemble;
import tsc_algorithms.ST_Ensemble;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.core.Instances;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class LoadParameterExperiments {
    
    
    public static void clusterRun(String[] args) throws Exception{
        String name=args[0];
        String problem=DataSets.fileNames[Integer.parseInt(args[1])-1];
        HESCA c=new HESCA();
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        if(name.equals("ACF")){
            train=ACF.formChangeCombo(train);
            int[] constantFeatures=InstanceTools.removeConstantTrainAttributes(train);
            test=ACF.formChangeCombo(test);
            InstanceTools.removeConstantAttributes(test, constantFeatures);
        }
        else if(name.equals("PS")){
            PowerSpectrum ps=new PowerSpectrum();
            train=ps.process(train);
            int[] constantFeatures=InstanceTools.removeConstantTrainAttributes(train);
            test=ps.process(test);
            InstanceTools.removeConstantAttributes(test, constantFeatures);
        }
//        else if(name.equals("ST")){
//            c=new ST_Ensemble();
//        }
        else{
            System.out.println("UNKNOWN CLASSIFIER "+name);
            System.exit(0);
        }
            String preds=DataSets.resultsPath+name+"/Predictions/"+problem+"/";
        for(int i=0;i<100;i++){
// Check existence
            File f= new File(preds+"internalCV_"+i+".csv");
            if(f.exists()){
                File f2= new File(preds+"internalTestPreds_"+i+".csv");
                File f3= new File(preds+"internalTestPreds_"+i+"OLD.csv");
                f2.renameTo(f3);
                f2= new File(preds+"fold"+i+".csv");
                f3= new File(preds+"fold"+i+".OLDcsv");
                f2.renameTo(f3);
                c.loadCVWeights(preds+"internalCV_"+i+".csv");
                c.saveTestPreds(preds+"internalTestPreds_"+i+".csv");
                Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
                c.buildClassifier(data[0]);
                OutFile p=new OutFile(preds+"/fold"+i+".csv");
                double act,pred,acc=0;
                for(int j=0;j<data[1].numInstances();j++)
                {
                    act=data[1].instance(j).classValue();
                    pred=c.classifyInstance(data[1].instance(j));
                    if(act==pred)
                        acc++;
                    p.writeLine(act+","+pred);                    
                }
                acc/=data[1].numInstances();
                System.out.println("Fold "+i+" Acc = "+acc);
            }
        }
            
    }
    public static void main(String[] args) throws Exception{
        if(args.length>1){   //Cluster run
            DataSets.resultsPath=DataSets.clusterPath+"Results/";
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            clusterRun(args);
        }
        else{
            String [] a={"ACF","38"};
            DataSets.resultsPath="C:/Users/ajb/Dropbox/Big TSC Bake Off/New Results/ensemble/";
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            clusterRun(a);
        }

    }
}
