
package vector_classifiers.weightedvoters;

import development.CollateResults;
import development.DataSets;
import static development.Experiments.singleClassifierAndFoldTrainTestSplit;
import development.MultipleClassifierEvaluation;
import java.io.File;
import java.util.ArrayList;
import timeseriesweka.classifiers.ensembles.voting.MajorityVote;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import vector_classifiers.HESCA;    
import vector_classifiers.stackers.SMLRE;
import weka.core.Instances;

/**
 * Implemented as separate classifier for explicit comparison, from Kuncheva and Rodríguez (2014)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class HESCA_MajorityVote extends HESCA {
    public HESCA_MajorityVote() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "HESCA_MajorityVote"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new MajorityVote();
    }   
    
    
        public static void main(String[] args) throws Exception {
        exps();
//        ana();
    }
    
    public static void ana() throws Exception {
        String dsetGroup = "UCI";
        
        String basePath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/";
//        String[] datasets = (new File(basePath + "Results/DTWCV/Predictions/")).list();
        
        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"hesca_tuningAlphaOnDset", 30).
            setTestResultsOnly(true).
//            setBuildMatlabDiagrams(true).
//            setDatasets(dsetGroup.equals("UCI") ? development.DataSets.UCIContinuousFileNames : development.DataSets.fileNames).
            setDatasets(basePath + dsetGroup + ".txt").
            readInClassifiers(new String[] { "HESCA", "HESCA_TunedAlpha"}, basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA", "HESCA_MajorityVote", "HESCA_NaiveBayesCombiner", "HESCA_RecallCombiner", "HESCA_WeightedMajorityVote"}, basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    

    public static void exps() {
        String resPath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
        int numfolds = 30;
        
        String[] dsets = DataSets.UCIContinuousFileNames;
        
//        String classifier = "HESCA_MajorityVote";
//        String classifier = "HESCA_NaiveBayesCombiner";
//        String classifier = "HESCA_RecallCombiner";
//        String classifier = "HESCA_WeightedMajorityVote";
//        String classifier = "HESCA_TunedAlpha";
        String classifier = "HESCA_PickBest";
        
        System.out.println("\t" + classifier);

        for (String dset : dsets) {
            
            System.out.println(dset);
            
            Instances all = ClassifierTools.loadData("C:/UCI Problems/" + dset + "/" + dset + ".arff");
            
            for (int fold = 0; fold < numfolds; fold++) {
                String predictions = resPath+classifier+"/Predictions/"+dset;
                File f=new File(predictions);
                if(!f.exists())
                    f.mkdirs();
        
                //Check whether fold already exists, if so, dont do it, just quit
                if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
                    Instances[] data = InstanceTools.resampleInstances(all, fold, .5);
                    
                    HESCA c = null;
                    try {
                        c = (HESCA) Class.forName("vector_classifiers.weightedvoters." + classifier).newInstance();
    //                    HESCA c = new HESCA_NaiveBayesCombiner();
    //                    HESCA c = new HESCA_RecallCombiner();
    //                    HESCA c = new HESCA_WeightedMajorityVote();
                    } catch (Exception e) {
                        System.out.println("balls: " + e);
                        System.exit(1);
                    }

                    c.setBuildIndividualsFromResultsFiles(true);
                    c.setResultsFileLocationParameters(resPath, dset, fold);
                    c.setRandSeed(fold);
                    c.setPerformCV(true);
                    c.setResultsFileWritingLocation(resPath);
                    
                    singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
                }
            }
        }        
    }
}
