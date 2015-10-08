package development.Jay;

import development.DataSets;
import static development.Jay.GoreckiDerivativesDTW.DATA_DIR;
import static development.Jay.GoreckiDerivativesEuclideanDistance.getCorrect;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.core.EuclideanDistance;
import weka.core.Instances;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class GoreckiClusterExperiments {

    // not bothering with other classifiers (ED, DED, DTW, DDTW) since we've already got results for those with EE
    public enum GoreckiClassifierType{
        DD_ED,
        DD_DTW,
    };
    public static void cluster_goreckiTrain(String dataset, GoreckiClassifierType classifierType, int seed) throws Exception{
        
        String clusterDataDir = "../TSC Problems/";
//        String clusterDataDir = DATA_DIR;
        
        Instances train, test, dTrain, dTest;
        EuclideanDistance ed;
        
        train = ClassifierTools.loadData(clusterDataDir+dataset+"/"+dataset+"_TRAIN");
        test = ClassifierTools.loadData(clusterDataDir+dataset+"/"+dataset+"_TEST");

        // instance resampling happens here, seed of 0 means that the standard train/test split is used
        if(seed!=0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, seed);
            train = temp[0];
            test = temp[1];
        }

        test = null; // to make sure it can't creep into the LOOCV anywhere
        
        GoreckiDerivativesEuclideanDistance dist = null;
        
        switch(classifierType){
            case DD_ED:
                dist = new GoreckiDerivativesEuclideanDistance();
                break;
            case DD_DTW:
                dist = new GoreckiDerivativesDTW();
                break;
            default:
                throw new Exception("Error: undefined classifier type: "+classifierType);
        }
                       
        String filePath = "goreckiResampleLOOCV/"+dataset+"/"+dataset+"_"+seed+"/"+dataset+"_"+seed+"_parsedOutput/";
        new File(filePath).mkdirs();
        
        double[] trainingPredictions = dist.crossValidateForAandB(train);
        int correct = 0;
        
        StringBuilder resultsBuilder = new StringBuilder();
        for(int i = 0; i < trainingPredictions.length; i++){
            if(trainingPredictions[i]==train.instance(i).classValue()){
                correct++;
            }
            resultsBuilder.append(trainingPredictions[i]+"/"+train.instance(i).classValue()+"\n");
        }
        
        FileWriter out = new FileWriter(filePath+dataset+"_"+seed+"_gorecki"+classifierType+".txt");
        out.append(((double)correct/train.numInstances())+"\n");
        out.append(dist.getA()+","+dist.getB()+"\n");
        out.append(resultsBuilder);
        out.close();
        
        // make info file too, match EE output style
        String infoLoc = "goreckiResampleLOOCV/"+dataset+"/"+dataset+"_"+seed+"/"+dataset+"_"+seed+".info";
        if(!new File(infoLoc).exists()){
            out = new FileWriter(infoLoc);
            out.append(train.relationName()+"\n"); // relationName
            out.append(train.numInstances()+"\n"); // numInstances
            for(int i = 0; i < train.numInstances(); i++){
                out.append(train.get(i).classValue()+"\n");
            }
            out.close();
        }
        
    }
    
    public static void scriptMaker_loocvTraining() throws Exception{
        String[] datasets = DataSets.fileNames;
        
        new File("goreckiResampleScripts/").mkdirs();
        
        String part1 = "#!/bin/csh\n\n#BSUB -q short\n#BSUB -J ";
        String part2 = "[1-101]\n#BSUB -oo output/";
        String part3 = "%I.out\n#BSUB -eo error/";
        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar -Xmx2048m TimeSeriesClassification.jar goreckiResampleLOOCV ";
        String part5 = " $LSB_JOBINDEX";

        FileWriter out;
        String[] timeLineParts;
        String jobString, jobName;
        

        FileWriter instructionsOut = new FileWriter("instructions.txt");
        Scanner scan;
        for(String dataset:datasets){
                        
            for(GoreckiClassifierType classifier: GoreckiClassifierType.values()){
                new File("goreckiResampleScripts/"+dataset).mkdir();

                jobName = dataset+"_"+classifier;
                jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5+" "+classifier;
                out = new FileWriter("goreckiResampleScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
                out.append(jobString);
                out.close();

                
                instructionsOut.append("bsub < goreckiResampleScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub\n");
            }
        }
        
        instructionsOut.close();
    }
    
    
    public static void clusterMaster(String[] args) throws Exception{
        if(args[0].equalsIgnoreCase("gorecki_makeScripts")){
            scriptMaker_loocvTraining();
        }else if(args[0].equalsIgnoreCase("goreckiResampleLOOCV")){
            String dataset = args[1];
            GoreckiClassifierType classifier = GoreckiClassifierType.valueOf(args[3].trim());
            int resampleSeed = Integer.parseInt(args[2].trim());
            resampleSeed--; // indexing from 0
            cluster_goreckiTrain(dataset, classifier, resampleSeed);
        }
    }
    
    public static void main(String[] args) throws Exception{
        if(args.length>0){
            clusterMaster(args);
            return;
        }
        // rest is for local 
//        
//        scriptMaker_loocvTraining();
        
//        GoreckiDerivativesEuclideanDistance ged = new GoreckiDerivativesEuclideanDistance();
//        Instances train = ClassifierTools.loadData(DATA_DIR+"Beef/Beef_TRAIN");
//        Instances train = ClassifierTools.loadData(DATA_DIR+"GunPoint/GunPoint_TRAIN");
//        ged.crossValidateForAandB(train);
        
//        double[] preds = ged.crossValidateForAandB(train);
//   for(int i = 0; i < train.numInstances(); i++){
//       System.out.println(preds[i]+"/"+train.instance(i).classValue());
//   }
        
        cluster_goreckiTrain("Beef", GoreckiClassifierType.DD_ED, 1);
    }

}
