package development.Jay;

import development.DataSets;
import development.ElasticEnsembleCluster;
import static development.Jay.ElasticEnsembleClusterExperiments.writeBestParamFileFromClusterOutput;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import tsc_algorithms.ElasticEnsemble;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.DerivativeFilter;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class RemoteExperimentsEE {

    public static void scriptMaker_genericIndividualClassifiers(String jobType, String tscProblemDir, String[] datasets, int numResamples, String queueName) throws Exception{
        
        // make the dir for storing the scripts
        new File("eeClusterScripts/").mkdirs();
        
        // add classifiers (added like this to allow for the simple hack of commenting out classifiers)
        ArrayList<ElasticEnsemble.ClassifierVariants> classifiersToUse = new ArrayList<>();
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.Euclidean_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.DTW_R1_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.DTW_Rn_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.WDTW_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.DDTW_R1_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.DDTW_Rn_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.WDDTW_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.LCSS_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.ERP_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.MSM_1NN);
        classifiersToUse.add(ElasticEnsemble.ClassifierVariants.TWE_1NN);
        
        // New cluster version vs. Grace version
        String javaVer = "java/jdk1.8.0_51";  // New cluster (not on Grace yet)
//        String javaVer = "java/jdk/1.8.0_31";   // Current version on Grace (not on new cluster yet)
        
        String part1 = "#!/bin/csh\n\n#BSUB -q "+queueName+"\n#BSUB -J ";
        String part2 = "[1-"+numResamples+"]\n#BSUB -oo output/";
        String part3 = "%I.out\n#BSUB -eo error/";
        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add "+javaVer+"\n\njava -jar -Xmx2048m TimeSeriesClassification.jar "+ jobType +" ";
        String part5 = " $LSB_JOBINDEX \""+tscProblemDir+"\" ";
        
        // args output order: jobTypeArg datasetName $LSB_JOBINDEX tscProblemDir classifeirType
        // eg:  cvAllParamsInstanceResampling FordA $LSB_JOBINDEX "/gpfs/home/sjx07ngu/TSCProblems/" DDTW_Rn_1NN
        
         FileWriter out;
        String jobString, jobName;
        
        FileWriter instructionsOut = new FileWriter("instructions.txt");
        Scanner scan;
//        for(String dataset:datasets){
        for(int d = 0; d < datasets.length;d++){
            String dataset = datasets[d];
            for(ElasticEnsemble.ClassifierVariants classifier: classifiersToUse){
                new File("eeClusterScripts/"+dataset).mkdir();

                jobName = dataset+"_"+classifier;
                if(d%3==0){
//                    String part1Hack = "#!/bin/csh\n\n#BSUB -q long-ib\n#BSUB -J ";
                    // hack to use other queue for now!
                    jobString = "#!/bin/csh\n\n#BSUB -q long-ib\n#BSUB -J "+jobName+part2+jobName+part3+jobName+part4+dataset+part5+classifier;
                }else{
                    jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5+classifier;
                }
                out = new FileWriter("eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
                out.append(jobString);
                out.close();

                instructionsOut.append("bsub < eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub\n");
            }
        }
        
        instructionsOut.close();
    }
    
    public static void scriptMaker_parseCv(String jobType, String[] datasets, String queueName) throws Exception{
        
        // make the dir for storing the scripts
        new File("eeClusterParseScripts/").mkdirs();
        
        // New cluster version vs. Grace version
        String javaVer = "java/jdk1.8.0_51";  // New cluster (not on Grace yet)
//        String javaVer = "java/jdk/1.8.0_31";   // Current version on Grace (not on new cluster yet)
        
        String part1 = "#!/bin/csh\n\n#BSUB -q "+queueName+"\n#BSUB -J ";
        String part2 = "\n#BSUB -oo output/";
        String part3 = ".out\n#BSUB -eo error/";
//        String part4 = ".err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add "+javaVer+"\n\njava -jar -Xmx2048m TimeSeriesClassification.jar "+ jobType +" ";
        String part4 = ".err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add "+javaVer+"\n\njava -jar -Xmx2048m cvParser.jar "+ jobType +" ";
        
        
        // args output order: jobTypeArg datasetName $LSB_JOBINDEX tscProblemDir classifeirType
        // eg:  cvAllParamsInstanceResampling FordA $LSB_JOBINDEX "/gpfs/home/sjx07ngu/TSCProblems/" DDTW_Rn_1NN
        
        FileWriter out;
        String jobString, jobName;
        
        FileWriter instructionsOut = new FileWriter("parserInstructions.txt");
        Scanner scan;
//        for(String dataset:datasets){
        for(int d = 0; d < datasets.length;d++){
            String dataset = datasets[d];

                jobName = dataset;

                jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset;
                
                out = new FileWriter("eeClusterParseScripts/"+dataset+".bsub");
                out.append(jobString);
                out.close();

                instructionsOut.append("bsub < eeClusterParseScripts/"+dataset+".bsub\n");   
        }
        
        instructionsOut.close();
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    public static void trainTestClassification(Instances train, Instances test, String dataName, int resampleId, ElasticEnsemble.ClassifierVariants classifier, String existingCvResultsDir) throws Exception{
        new File("eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/").mkdirs();
        
        // check if test results already exist, skip if it does
        if(new File("eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_"+classifier+".txt").exists()){
            throw new Exception("Test results already exist for "+dataName+"_"+resampleId+" for "+classifier);
        }
        
        // check if training results exist, skip if not
        if(!new File(existingCvResultsDir+"eeClusterOutput/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_parsedOutput/"+dataName+"_"+resampleId+"_"+classifier+".txt").exists()){
            throw new Exception("Relevant training data doesn't exist for "+dataName+"_"+resampleId+" for "+classifier);
        }
        
        // transform the data (if necessary). 
        if(classifier.equals(ElasticEnsemble.ClassifierVariants.DDTW_R1_1NN)||classifier.equals(ElasticEnsemble.ClassifierVariants.DDTW_Rn_1NN)||classifier.equals(ElasticEnsemble.ClassifierVariants.WDDTW_1NN)){
             DerivativeFilter d = new DerivativeFilter();
             train = d.process(train);
             test = d.process(test);
        }
        
        ElasticEnsembleCluster ee = new ElasticEnsembleCluster(dataName, resampleId);
        ee.setPathToTrainingResults(existingCvResultsDir);
        // set only the relevant classifier, read in data using in-built functionality
        ee.removeAllClassifiersFromEnsemble();
        ee.addClassifierToEnsemble(classifier);
        ee.buildClassifier(train);
        
        // extract the classifier that we are interested in - because we want to access the distributionForInstance
        kNN knn = ee.getBuiltClassifier(classifier);
        
        // test classification
        int correct = 0;
        double[] distribution;
        double bsfDist;
        double bsfDistId;
        
        StringBuilder lineBuilder;
        StringBuilder output = new StringBuilder();
        
        for(int i = 0; i < test.numInstances(); i++){
            distribution = knn.distributionForInstance(test.instance(i));
            bsfDist = -1;
            bsfDistId = -1;
            lineBuilder = new StringBuilder();
            for(int d = 0; d < distribution.length; d++){
                lineBuilder.append(distribution[d]+",");
                if(distribution[d] > bsfDist){
                    bsfDist = distribution[d];
                    bsfDistId = d;
                }
            }
            output.append(bsfDistId+"/"+test.instance(i).classValue()+"["+lineBuilder.toString().substring(0, lineBuilder.length()-1)+"]\n");
            if(bsfDistId==test.instance(i).classValue()){
                correct++;
            }
        }
        
        
        FileWriter out = new FileWriter("eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_"+classifier+".txt");
        out.append(((double)correct/test.numInstances())+"\n"+output);
        out.close();
        
    }
    
    public static void main(String[] args) throws Exception{
        
        
        //local running
        if(args.length==0){

            System.out.println("Running locally!");
            
            String remoteTscProblemDir = "testDir";
                
            String[] datasets = DataSets.fileNames;
            String jobType = "cvAllParamsInstanceResampling_newData";
            int numResamples = 10;
            String queue = "long-eth";
            scriptMaker_genericIndividualClassifiers(jobType, remoteTscProblemDir, datasets, numResamples, queue);
            
            
            
        //remote running    
        }else{
            //<editor-fold defaultstate="collapsed" desc="script making">
            if(args[0].equalsIgnoreCase("writeLoocvScripts")){
                String remoteTscProblemDir = args[1];
                
                String[] datasets = {"ECG200","ECG5000","InsectWingbeatSound", "Phoneme"};
                String jobType = "cvAllParamsInstanceResampling";
                int numResamples = 100;
                String queue = "short";
                scriptMaker_genericIndividualClassifiers(jobType, remoteTscProblemDir, datasets, numResamples, queue);
                
            }else if(args[0].equalsIgnoreCase("writeLoocvScripts_redo")){
                String remoteTscProblemDir = args[1];
                
                String[] datasets = DataSets.fileNames;
                String jobType = "cvAllParamsInstanceResampling_newData";
                int numResamples = 10;
                String queue = "long-eth";
                scriptMaker_genericIndividualClassifiers(jobType, remoteTscProblemDir, datasets, numResamples, queue);
                
            }else if(args[0].equalsIgnoreCase("writeLoocvScripts_10Resamples")){
                String remoteTscProblemDir = args[1];
                String[] datasets = {
                    "FordA",
                    "FordB",
                    "HandOutlines",
                    "NonInvasiveFatalECG_Thorax1",
                    "NonInvasiveFatalECG_Thorax2",
                    "StarLightCurves",
                    "UWaveGestureLibraryAll",
                    "ElectricDevices"};
//                scriptMaker_loocvTraining(datasets, remoteTscProblemDir,10,"long-eth");
                String jobType = "cvAllParamsInstanceResampling";
                int numResamples = 100;
                String queueName = "long-eth";
                scriptMaker_genericIndividualClassifiers(jobType, remoteTscProblemDir, datasets, numResamples, queueName);
                
                
                
            }else if(args[0].equalsIgnoreCase("writeScripts_trainTest")){
                String remoteTscProblemDir = args[1];
                String[] datasets = {"ItalyPowerDemand","ECG200","ECG5000","InsectWingbeatSound", "Phoneme", "LargeKitchenAppliances", "RefrigerationDevices", "ScreenType", "ShapesAll", "SmallKitchenAppliances"};
                String jobType = "trainTest";
                int numResamples = 100;
                String queueName = "short";
                scriptMaker_genericIndividualClassifiers(jobType, remoteTscProblemDir, datasets, numResamples, queueName);
            
            }else if(args[0].equalsIgnoreCase("writeScripts_parseCv")){
//                String remoteTscProblemDir = args[1];
//                String remoteTscProblemDir = "dummyDir";
//                String[] datasets = {"ItalyPowerDemand","ECG200","ECG5000","InsectWingbeatSound", "Phoneme", "LargeKitchenAppliances", "RefrigerationDevices", "ScreenType", "ShapesAll", "SmallKitchenAppliances"};
                String[] datasets = DataSets.fileNames;//{"ItalyPowerDemand","ECG200","ECG5000","InsectWingbeatSound", "Phoneme", "LargeKitchenAppliances", "RefrigerationDevices", "ScreenType", "ShapesAll", "SmallKitchenAppliances"};
                String jobType = "parseCv";
//                int numResamples = 100;
                String queueName = "long-eth";
                scriptMaker_parseCv(jobType, datasets, queueName);
                
                
//</editor-fold>
             
            }else if(args[0].equalsIgnoreCase("cvAllParamsInstanceResampling_newData")){
                
                String dataName = args[1];
//                int resampleID = Integer.parseInt(args[2].trim()) -1; /////////NOTE now indexing from 0 to include
                //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                int resampleID = Integer.parseInt(args[2].trim()) -1+10; /////////NOTE PART 2!!!!! second batch, so adding 10 to the resample ID
                
                String tscProblemDir = args[3];
                ElasticEnsemble.ClassifierVariants measureType = ElasticEnsemble.ClassifierVariants.valueOf(args[4].trim());
                
                Instances train = ClassifierTools.loadData(tscProblemDir+"/"+dataName+"/"+dataName+"_TRAIN");
                Instances test = ClassifierTools.loadData(tscProblemDir+"/"+dataName+"/"+dataName+"_TEST");
                
                // note: ALWAYS normalised
                Instances resampledTrain = new NormalizeCase().process(InstanceTools.resampleTrainAndTestInstances(train, test, resampleID)[0]);
                
                ThreadedExperimentEE.writeCvAllParams(resampledTrain, dataName, resampleID, measureType,"");            
            
                    
                
            }else if(args[0].equalsIgnoreCase("trainTest")){   
                
                String dataName = args[1];
                int resampleID = Integer.parseInt(args[2].trim()) -1; /////////NOTE now indexing from 0 to include
                String tscProblemDir = args[3];
                ElasticEnsemble.ClassifierVariants measureType = ElasticEnsemble.ClassifierVariants.valueOf(args[4].trim());
                
                Instances train = ClassifierTools.loadData(tscProblemDir+"/"+dataName+"/"+dataName+"_TRAIN");
                Instances test = ClassifierTools.loadData(tscProblemDir+"/"+dataName+"/"+dataName+"_TEST");
                
                // note: ALWAYS normalised
                Instances[] resampled = InstanceTools.resampleTrainAndTestInstances(train, test, resampleID);
                Instances resampledTrain = new NormalizeCase().process(resampled[0]);  
                Instances resampledTest = new NormalizeCase().process(resampled[1]);  
                
                
                String existingCvDir = "/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/03_EEFinalResults/02_loocvTraining/";
            try{
//                clusterTestClassification(datasetName, classifier, resampleSeed);
                trainTestClassification(resampledTrain, resampledTest, dataName, resampleID, measureType, existingCvDir);
            }catch(Exception e){
                e.printStackTrace();
                // do nothing, as i don't want the output files being full of text!
            }
                
                
            }else if(args[0].equalsIgnoreCase("parseCv")){   
                
                ElasticEnsemble.ClassifierVariants[] classifiers = ElasticEnsemble.ClassifierVariants.values();
                
                String dataset = args[1];
                
                for(ElasticEnsemble.ClassifierVariants classifier:classifiers){
//                    for(int i = 1; i <=100; i++){
                    for(int i = 0; i <=100; i++){
                        try{
                            writeBestParamFileFromClusterOutput(dataset, dataset+"_"+i, classifier, true);
                        }catch(Exception e){
//                            e.printStackTrace();
                        }
                    }
                }
                
//            
//                
//            
//            }else if(args[0].equalsIgnoreCase("parseResultsClusterInTens")){
//
//                ElasticEnsemble.ClassifierVariants[] classifiers = ElasticEnsemble.ClassifierVariants.values();
//                int datasetOffset = Integer.parseInt(args[1])-1;
//                
//                String dataset;
//                String[] datasets = {"ECG200","ECG5000","InsectWingbeatSound", "Phoneme"};
//                for(int d = datasetOffset*10 ; d < datasetOffset*10+10; d++){
//                    if(d < datasets.length){
//                        dataset = datasets[d];
//                    }else{
//                        return;
//                    }
//                    for(ElasticEnsemble.ClassifierVariants classifier:classifiers){
//                        for(int i = 1; i <=100; i++){
//                            try{
//                                writeBestParamFileFromClusterOutput(dataset, dataset+"_"+i, classifier, true);
//                            }catch(Exception e){
//                                e.printStackTrace();
//                            }
//                        }
//                    }
//                }
            }else{
                throw new Exception("Error: undefined instructions: "+args[0]);
            }
            
            
        }
        
        
    }
    
    
    
}
