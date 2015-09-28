package development.Jay;

import development.DataSets;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.meta.timeseriesensembles.ElasticEnsemble;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class RemoteExperimentsEE {

    public static void scriptMaker_loocvTraining(String[] datasets, String tscProblemDir) throws Exception{
        new File("eeClusterScripts/").mkdirs();
        
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

        String part1 = "#!/bin/csh\n\n#BSUB -q short\n#BSUB -J ";
        String part2 = "[1-100]\n#BSUB -oo output/";
        String part3 = "%I.out\n#BSUB -eo error/";
        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar -Xmx2048m TimeSeriesClassification.jar cvAllParamsInstanceResampling ";
        String part5 = " $LSB_JOBINDEX \""+tscProblemDir+"\" ";

        FileWriter out;
        String jobString, jobName;
        
        FileWriter instructionsOut = new FileWriter("instructions.txt");
        Scanner scan;
        for(String dataset:datasets){
            
            for(ElasticEnsemble.ClassifierVariants classifier: classifiersToUse){
                new File("eeClusterScripts/"+dataset).mkdir();

                jobName = dataset+"_"+classifier;
                jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5+classifier;
                out = new FileWriter("eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
                out.append(jobString);
                out.close();

                instructionsOut.append("bsub < eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub\n");
            }
        }
        
        instructionsOut.close();
    }
     
    public static void main(String[] args) throws Exception{
        

        //local running
        if(args.length==0){

            System.out.println("Running locally!");
            
            
            
            
            
        //remote running    
        }else{
            if(args[0].equalsIgnoreCase("writeLoocvScripts")){
                String remoteTscProblemDir = args[1]; 
                String[] datasets = DataSets.fileNames;
                scriptMaker_loocvTraining(datasets, remoteTscProblemDir);
                
            }else if(args[0].equalsIgnoreCase("cvAllParamsInstanceResampling")){
                
                String dataName = args[1];
                int resampleID = Integer.parseInt(args[2].trim());
                String tscProblemDir = args[3];
                ElasticEnsemble.ClassifierVariants measureType = ElasticEnsemble.ClassifierVariants.valueOf(args[4].trim());
                
                Instances train = ClassifierTools.loadData(tscProblemDir+"/"+dataName+"/"+dataName+"_TRAIN");
                Instances test = ClassifierTools.loadData(tscProblemDir+"/"+dataName+"/"+dataName+"_TEST");
                
                Instances resampledTrain = new NormalizeCase().process(InstanceTools.resampleTrainAndTestInstances(train, test, resampleID)[0]);
                
                ThreadedExperimentEE.writeCvAllParams(resampledTrain, dataName, resampleID, measureType,"");                
            }else{
                throw new Exception("Error: undefined instructions: "+args[0]);
            }
            
            
        }
        
        
    }
    
    
    
}
