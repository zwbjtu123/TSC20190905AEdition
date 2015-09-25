package development.Jay;

import development.ElasticEnsembleCluster;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.timeseriesensembles.ElasticEnsemble;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.DerivativeFilter;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class ThreadedExperimentEE extends Thread{

    String datasetName;
    Instances originalTrain;
    Instances originalTest;
    ElasticEnsemble.ClassifierVariants measureType;
    int numResamples;
    String outputResultsDir;

    public ThreadedExperimentEE(String datasetName, Instances originalTrain, Instances originalTest, ElasticEnsemble.ClassifierVariants measureType, int numResamples, String outputResultsDir) {
        this.datasetName = datasetName;
        this.originalTrain = originalTrain;
        this.originalTest = originalTest;
        this.measureType = measureType;
        this.numResamples = numResamples;
        this.outputResultsDir = outputResultsDir;
    }
    
    
    public void resampleExperiment() throws Exception{
        
        Instances train;

        for(int resample = 1; resample <= this.numResamples; resample++){
            train = InstanceTools.resampleTrainAndTestInstances(originalTrain, originalTest, resample)[0];
            writeCvAllParams(train, datasetName, resample, measureType, outputResultsDir);
        }
        
    }
    
    @Override
    public void run() {
        try{
            resampleExperiment();
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    
    public static void writeCvAllParams(Instances instances, String datasetName, int resampleId, ElasticEnsemble.ClassifierVariants measureType, String outResultsDir) throws Exception{

        // don't bother if parsed output already exists for this dataset, resampleId and classifier
        if(new File(outResultsDir+"eeClusterOutput/"+datasetName+"/"+datasetName+"_"+resampleId+"/"+datasetName+"_"+resampleId+"_parsedOutput/"+datasetName+"_"+resampleId+"_"+measureType+".txt").exists()){
            return;
        }
       
        Instances fullTrain;
        double prediction;
        FileWriter out;
        File outDir;
        String outLoc;
        int correct;
        double acc;

        // set up the number of param options
        int maxParamId = 100;
        if(measureType.equals(ElasticEnsemble.ClassifierVariants.Euclidean_1NN)||measureType.equals(ElasticEnsemble.ClassifierVariants.DTW_R1_1NN)||measureType.equals(ElasticEnsemble.ClassifierVariants.DDTW_R1_1NN)){
            maxParamId = 1;
        }

        // transform the data (if necessary). Do this here so we only have to do it once, then copy for other folds
        if(measureType.equals(ElasticEnsemble.ClassifierVariants.DDTW_R1_1NN)||measureType.equals(ElasticEnsemble.ClassifierVariants.DDTW_Rn_1NN)||measureType.equals(ElasticEnsemble.ClassifierVariants.WDDTW_1NN)){
             DerivativeFilter d = new DerivativeFilter();
             fullTrain = d.process(instances);
        }else{
            fullTrain = new Instances(instances);
        }

        File paramFile;
        
        for(int paramId = 0; paramId< maxParamId; paramId++){
            
            outLoc = outResultsDir+"eeClusterOutput/"+datasetName+"/"+datasetName+"_"+resampleId+"/"+datasetName+"_"+resampleId+"_"+measureType+"/"+datasetName+"_"+resampleId+"_"+measureType+"_p_"+paramId+".txt";
            paramFile = new File(outLoc);
            if(paramFile.exists()){
                if(paramFile.length() > 0){ // weird bug on cluster made a few files with no content (maybe crashed while writing, or didn't close stream)
                    continue;
                }
            }

            outDir = new File(outResultsDir+"eeClusterOutput/"+datasetName+"/"+datasetName+"_"+resampleId+"/"+datasetName+"_"+resampleId+"_"+measureType+"/");
            outDir.mkdirs();

            StringBuilder st = new StringBuilder();
            Instances train;
            Instances test;
            
            correct = 0;
            
            Instance testIns;
            for(int i = 0; i < instances.numInstances(); i++){
                train = new Instances(fullTrain);
                testIns = train.remove(i);
            
                kNN classifier = ElasticEnsembleCluster.getInternalClassifier(measureType, paramId, train);
                
                prediction = classifier.classifyInstance(testIns);

                if(prediction==testIns.classValue()){
                    correct++;
                }
                st.append(i).append(",").append(prediction).append("/").append(testIns.classValue()).append("\n");
            }

            acc = (double)correct/fullTrain.numInstances();
            out = new FileWriter(outLoc);
            out.append(acc+"\n");
            out.append(st);
            out.close();
            // create a global information file as a contingency (if it doesn't exist). This can provide information such as number of instances and actual class values
            // without needing to load raw data, and also help inturpret old files if the origin is unclear by storing the relation name
            String infoLoc = outResultsDir+"eeClusterOutput/"+datasetName+"/"+datasetName+"_"+resampleId+"/"+datasetName+"_"+resampleId+".info";
            if(!new File(infoLoc).exists()){
                out = new FileWriter(infoLoc);
                String summary = instances.toSummaryString();
                Scanner scan = new Scanner(summary);
                scan.useDelimiter("\n");
                out.append(scan.next().split(":")[1].trim()+"\n"); // relationName
                out.append(scan.next().split(":")[1].trim()+"\n"); // numInstances
                scan.close();
                for(int i = 0; i < instances.numInstances(); i++){
                    out.append(instances.get(i).classValue()+"\n");
                }
                out.close();
            }
        }
    }
    
    public static void threadedSingleDataset(String tscProblemDir, String datasetName, String outputResultsDir, int numResamples) throws Exception{
        
        ElasticEnsemble.ClassifierVariants[] measureTypes = {
            ElasticEnsemble.ClassifierVariants.Euclidean_1NN,
            ElasticEnsemble.ClassifierVariants.DTW_R1_1NN,
            ElasticEnsemble.ClassifierVariants.DTW_Rn_1NN,
            ElasticEnsemble.ClassifierVariants.DDTW_R1_1NN,
            ElasticEnsemble.ClassifierVariants.DDTW_Rn_1NN,
            ElasticEnsemble.ClassifierVariants.WDTW_1NN,
            ElasticEnsemble.ClassifierVariants.WDDTW_1NN,
            ElasticEnsemble.ClassifierVariants.LCSS_1NN,
            ElasticEnsemble.ClassifierVariants.ERP_1NN,
            ElasticEnsemble.ClassifierVariants.MSM_1NN,
            ElasticEnsemble.ClassifierVariants.TWE_1NN
        };
        
        ThreadedExperimentEE[] threads = new ThreadedExperimentEE[measureTypes.length]; 

        NormalizeCase nc = new NormalizeCase();
        Instances train = nc.process(ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TRAIN"));
        Instances test = nc.process(ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TEST"));
        
        
        for(int i = 0; i < measureTypes.length; i++){    
            threads[i]=new ThreadedExperimentEE(datasetName, train, test, measureTypes[i], numResamples, outputResultsDir);
            threads[i].start();
            System.out.println(" started "+datasetName+" with "+measureTypes[i]);
        }
        
        for(int i=0;i<threads.length;i++){
            threads[i].join();
        }
    }
    
    public static void main(String[] args) throws Exception{
        threadedSingleDataset("C:/Users/sjx07ngu/Dropbox/TSC Problems/", "ItalyPowerDemand", "C:/Jay/TestOutputEE/", 10);
    }
    
}
