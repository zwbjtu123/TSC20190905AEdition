package bakeOffExperiments;

import JamesStuff.TestTools;
import development.DataSets;
import fileIO.OutFile;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import tsc_algorithms.SAXVSM;
import tsc_algorithms.BOSSEnsemble;
import tsc_algorithms.BagOfPatterns;

/**
 * Runs resample experiments for the classifiers:
 *  BOSSEnsemble
 *  SAXVSM
 *  LinBagOfPatterns
 * 
 * With (currently) all using parameter search EACH FOLD 
 * 
 * @author JamesL
 */
public class DictionaryClassifierExperiments {
    public static double[] resampleExperiment(Instances train, Instances test, Classifier c, int resamples,OutFile of){

        double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            double act,pred;
            try{              
                c.buildClassifier(data[0]);
                foldAcc[i]=ClassifierTools.accuracy(test, c);
                of.writeString(foldAcc[i]+",");
            }catch(Exception e) {
                System.out.println(" Error in DictionaryClassifierExperiments.resampleExperiment(): "+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
            }
        }            
        return foldAcc;
    }
    
    public static void main(String[] args) throws Exception {
        //nonThreadedFull();
        threadedExperiments();
    }
    
    public static void threadedExperiments() throws Exception {
        
        //CHANGE DATAPATH AS FOUND in threadedSingleClassifier(...).str => "C:\\Temp\\TESTDATA\\TSC Problems\\"
        
        String dataPath="C:\\Temp\\TESTDATA\\TSC Problems\\";
        String outPath="C:\\Temp\\BAKEOFFRESULTS\\";
        
        BOSSEnsemble boss = new BOSSEnsemble(true);
        SAXVSM saxvsm = new SAXVSM();
        BagOfPatterns bop = new BagOfPatterns();

        Classifier[] classifiers = { boss, saxvsm, bop } ;
        String[] fileNames = { "BakeOff_BOSSEnsemble", "BakeOff_SAXVSM", "BakeOff_BOP" };

        for (int i = 0; i < classifiers.length; ++i) {
            System.out.println("**" + fileNames[i] + "**");
            OutFile out = new OutFile(outPath+fileNames[i] + ".csv");
            Experiments.threadedSingleClassifier(classifiers[i], out);
        }
    }
    
    public static void nonThreadedFull() {
        try {
            String dataPath="C:\\Temp\\TESTDATA\\TSC Problems\\";
            String outPath="C:\\Temp\\BAKEOFFRESULTS\\";
            
            String[] datasets = DataSets.fileNames;
            
            BOSSEnsemble boss = new BOSSEnsemble(true);
            SAXVSM saxvsm = new SAXVSM();
            BagOfPatterns bop = new BagOfPatterns();
            
            Classifier[] classifiers = { boss, saxvsm, bop } ;
            String[] fileNames = { "BakeOff_BOSSEnsemble", "BakeOff_SAXVSM", "BakeOff_BOP" };
            
            for (int i = 0; i < classifiers.length; ++i) {
                System.out.println("\n" + fileNames[i]);
                
                OutFile out = new OutFile(outPath+fileNames[i] + ".csv");
                
                for (String dataset : datasets) {
                    Instances[] data = TestTools.loadTestTrainsets(dataPath, dataset);
                    
                    
                    //outformat [classifier filename]: 
                    // dataset1, acc1, acc2....., acc100 \n
                    // dataset2, acc1, acc2....., acc100 \n
                    out.writeString(datasets[i] + ", ");
                    double[] res = resampleExperiment(data[0], data[1], classifiers[i], 100, out);
                    out.newLine();
                    
                    System.out.print(dataset + ": ");
                    for (double d : res)
                        System.out.print(d + ", ");
                    System.out.println("");
                }
                
                out.closeFile();
            }
 
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
}
