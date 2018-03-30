
package vector_classifiers.weightedvoters;

import development.CollateResults;
import development.DataSets;
import static development.Experiments.singleClassifierAndFoldTrainTestSplit;
import development.MultipleClassifierEvaluation;
import java.io.File;
import java.util.ArrayList;
import timeseriesweka.classifiers.ensembles.voting.BestIndividualTrain;
import timeseriesweka.classifiers.ensembles.voting.MajorityConfidence;
import timeseriesweka.classifiers.ensembles.voting.MajorityVote;
import timeseriesweka.classifiers.ensembles.weightings.AUROC;
import timeseriesweka.classifiers.ensembles.weightings.BalancedAccuracy;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import timeseriesweka.classifiers.ensembles.weightings.FScore;
import timeseriesweka.classifiers.ensembles.weightings.MCCWeighting;
import timeseriesweka.classifiers.ensembles.weightings.ModuleWeightingScheme;
import timeseriesweka.classifiers.ensembles.weightings.NLL;
import timeseriesweka.classifiers.ensembles.weightings.RecallByClass;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import utilities.ClassifierTools;
import utilities.GenericTools;
import utilities.InstanceTools;
import vector_classifiers.EnsembleSelection;
import vector_classifiers.HESCA;    
import vector_classifiers.stackers.SMLR;
import vector_classifiers.stackers.SMLRE;
import vector_classifiers.stackers.SMM5;
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
//        weightingSchemeExps();
//        ana();
//        ana_testDeleteAfter();
    }
    
    public static void ana_testDeleteAfter() throws Exception {
        System.out.println("ana_testDeleteAfter");
        String dsetGroup = "UCI";
        
        String basePath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/";
//        String[] datasets = (new File(basePath + "Results/DTWCV/Predictions/")).list();
        
        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"hescavses_differentlibraries_test7", 30). 
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true).
            setDatasets(basePath + dsetGroup + ".txt").
//            setDatasets(new String[] { "ionosphere" }).
//            readInClassifiers(new String[] { "HESCA", "HESCA+", "HESCAks", "EnsembleSelectionHESCAClassifiers", "EnsembleSelectionHESCA+Classifiers", "EnsembleSelectionAll22Classifiers" },
//                    new String[] {           "CAWPE", "CAWPE+", "CAWPEks", "ES", "ES+", "ESks" },
//                    basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] {  "HESCA+", "EnsembleSelectionHESCA+Classifiers_Preds", },
//                    new String[] {            "CAWPE+", "ES+",},
//                    basePath+dsetGroup+"Results/").
            readInClassifiers(new String[] { "HESCA", "HESCA+", "HESCAks", "EnsembleSelectionHESCAClassifiers_Preds", "EnsembleSelectionHESCA+Classifiers_Preds", "EnsembleSelectionAll22Classifiers_Preds"},
                    new String[] {           "CAWPE", "CAWPE+", "CAWPEks","ESpreds", "ES+preds", "ESks_Preds" },
                    basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    public static void ana() throws Exception {
        String dsetGroup = "UCI";
        
        String basePath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/";
//        String[] datasets = (new File(basePath + "Results/DTWCV/Predictions/")).list();
        
//        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"hesca+v4VSheteroensembles", 30). //_tunedAlpha
//        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"hesca+VSheteroensembles_tunedAlpha", 30).
//        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"hesca+v2VSheteroensembles_tunedAlpha", 30).
//        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"hesca+v3VSheteroensembles", 30). //_tunedAlpha
        new MultipleClassifierEvaluation(basePath+"ResubmissionAna/", dsetGroup+"probsVspredsAnalysis_noClassifierWeighting", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true).
//            setDatasets(dsetGroup.equals("UCI") ? development.DataSets.UCIContinuousFileNames : development.DataSets.fileNames).
            setDatasets(basePath + dsetGroup + ".txt").
//            readInClassifiers(new String[] { "HESCA", "HESCA_TunedAlpha"}, basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA", "HESCA_MajorityVote", "HESCA_NaiveBayesCombiner", "HESCA_RecallCombiner", "HESCA_WeightedMajorityVote", "HESCA_PickBest", "EnsembleSelectionHESCAClassifiers", "SMLRE", "SMLR", "SMM5", },//"HESCA_TunedAlpha"}, 
//                    new String[] { "HESCA", "MV", "NBC", "RC", "WMV", "PB", "ES", "SMLRE", "SMLR", "SMM5", },//"THESCA"},
//                    basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA+", "HESCA+_MajorityVote", "HESCA+_NaiveBayesCombiner", "HESCA+_RecallCombiner", "HESCA+_WeightedMajorityVote", "HESCA+_PickBest", "EnsembleSelectionHESCA+Classifiers", "SMLRE+", "SMLR+", "SMM5+", "HESCA+_TunedAlpha"}, 
//                    new String[] { "HESCA+", "MV+", "NBC+", "RC+", "WMV+", "PB+", "ES+", "SMLRE+", "SMLR+", "SMM5+", "THESCA+"},
//                    basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA+v2", "HESCA+v2_MajorityVote", "HESCA+v2_NaiveBayesCombiner", "HESCA+v2_RecallCombiner", "HESCA+v2_WeightedMajorityVote", "HESCA+v2_PickBest", "EnsembleSelectionHESCA+v2Classifiers", "HESCA+v2_TunedAlpha" },// "SMLRE+", "SMLR+", "SMM5+"}, 
//                    new String[] { "HESCA+v2", "MV+v2", "NBC+v2", "RC+v2", "WMV+v2", "PB+v2", "ES+v2", "THESCA+v2", }, //"SMLRE+", "SMLR+", "SMM5+"},
//                    basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA+v3", "HESCA+v3_MajorityVote", "HESCA+v3_NaiveBayesCombiner", "HESCA+v3_RecallCombiner", "HESCA+v3_WeightedMajorityVote", "HESCA+v3_PickBest", "EnsembleSelectionHESCA+v3Classifiers", }, //"HESCA+v3_TunedAlpha" },// "SMLRE+", "SMLR+", "SMM5+"}, 
//                    new String[] { "HESCA+v3", "MV+v3", "NBC+v3", "RC+v3", "WMV+v3", "PB+v3", "ES+v3", }, //"THESCA+v3", }, //"SMLRE+", "SMLR+", "SMM5+"},
//                    basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA+v4", "HESCA+v4_MajorityVote", "HESCA+v4_NaiveBayesCombiner", "HESCA+v4_RecallCombiner", "HESCA+v4_WeightedMajorityVote", "HESCA+v4_PickBest", "EnsembleSelectionHESCA+v4Classifiers", }, //"HESCA+v4_TunedAlpha" },// "SMLRE+", "SMLR+", "SMM5+"}, 
//                    new String[] { "HESCA+v4", "MV+v4", "NBC+v4", "RC+v4", "WMV+v4", "PB+v4", "ES+v4", }, //"THESCA+v4", }, //"SMLRE+", "SMLR+", "SMM5+"},
//                    basePath+dsetGroup+"Results/").
            readInClassifiers(new String[] { "HESCA_MajorityVote", "HESCA_MajorityConfidence", },
                    new String[] { "Preds", "Probs", },
                    basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    

    public static void exps() throws Exception {
        String resPath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
//        String resPath = "C:/JamesLPHD/HESCA/UCI/UCIResults/TempOverTunedClassifiers/";
//        String resPath = "C:/JamesLPHD/HESCA/UCR/UCRResults/";
//        String resWritePath = "C:/JamesLPHD/HESCA/UCR/UCRResults/HESCA+TunedAlpha/";
//        String resPath = "Z:/Results/JayMovingInProgress/";
//        int numfolds = 5;
        int numfolds = 30;
//        int numfolds = 100;
        
//        String[] dsets = DataSets.UCIContinuousFileNames;
//        String[] dsets = { "letter", "statlog-shuttle" };
//        String[] dsets = GenericTools.readFileLineByLineAsArray("C:/JamesLPHD/HESCA/UCI/UCI_4biggunsRemoved.txt");
        String[] dsets = GenericTools.readFileLineByLineAsArray("C:/JamesLPHD/HESCA/UCI/UCI.txt");
//        String[] dsets = GenericTools.readFileLineByLineAsArray("C:/JamesLPHD/HESCA/UCR/UCR.txt");
        
        

        
//        String classifier = "HESCA_MajorityVote";
//        String classifier = "HESCA_NaiveBayesCombiner";
//        String classifier = "HESCA_RecallCombiner";
//        String classifier = "HESCA_WeightedMajorityVote";
//        String classifier = "HESCA_TunedAlpha";
//        String classifier = "HESCA_PickBest";
        
        String[] classifiers = { 
//            "HESCA", 
//            "HESCA_MajorityVote", 
//            "EnsembleSelectionOverTunedClassifiers",
//            "HESCA_NaiveBayesCombiner", "HESCA_RecallCombiner", 
//            "HESCA_WeightedMajorityVote", "HESCA_TunedAlpha", "HESCA_PickBest"
            
//            "HESCA_MajorityConfidence"
//            "HIVE-COTE_hescaWeightings"
//            "HESCA+"
//            "HESCA_WeightedMajorityConfidence(Jay)",
//            "HESCA_ExponentiallyWeightedMajorityVote(Jay)",
            
//            "HESCA_OverTunedClassifiers"
//            "HESCA_TunedAlpha"
            "HESCA_ExponentiallyWeightedVote"
        };
        
        for (String classifier : classifiers) {
            System.out.println("\t" + classifier);

            for (String dset : dsets) {

                System.out.println(dset);

//                Instances train = ClassifierTools.loadData("C:/TSC Problems/" + dset + "/" + dset + "_TRAIN.arff");
//                Instances test = ClassifierTools.loadData("C:/TSC Problems/" + dset + "/" + dset + "_TEST.arff");
                Instances all = ClassifierTools.loadData("C:/UCI Problems/" + dset + "/" + dset + ".arff");

                for (int fold = 0; fold < numfolds; fold++) {
                    String predictions = resPath+classifier+"/Predictions/"+dset;
                    File f=new File(predictions);
                    if(!f.exists())
                        f.mkdirs();

                    //Check whether fold already exists, if so, dont do it, just quit
                    if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
                        Instances[] data = InstanceTools.resampleInstances(all, fold, .5);
//                        Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, fold);

                        HESCA c = new HESCA();
                        c.setWeightingScheme(new TrainAcc(4.0));
                        c.setVotingScheme(new MajorityVote());
                        
//                        HESCA c = null;
//                        try {
//                            if (classifier.equals("HESCA"))
//                                c = new HESCA();
//                            else if (classifier.contains("EnsembleSelection")) {
//                                c = new EnsembleSelection();
//                                ((EnsembleSelection)c).setNumOfTopModelsToInitialiseBagWith(1);
//                            }
//                            else if (classifier.contains("SMLRE"))
//                                c = new SMLR();
//                            else if (classifier.contains("SMLR") && !classifier.contains("SMLRE"))
//                                c = new SMLRE();
//                            else if (classifier.contains("SMM5"))
//                                c = new SMM5();        
//                            else 
//                                c = (HESCA) Class.forName("vector_classifiers.weightedvoters." + classifier).newInstance();
//                        } catch (Exception e) {
//                            System.out.println("balls: " + e);
//                            System.exit(1);
//                        }

//                        c.setClassifiers(null, HESCAUCR_Classifiers, null);
//                        c.setResultsFileLocationParameters(tunedSingleClassifiersLocations, dset, fold);
                        
                        c.setBuildIndividualsFromResultsFiles(true);
                        c.setResultsFileLocationParameters(resPath, dset, fold);
                        c.setRandSeed(fold);
                        
                        //hive cote results dont have trainprobabilities
                        c.setPerformCV(true);
                        c.setResultsFileWritingLocation(resPath);

                        singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
                    }
                }
            }        
        }
    }
    
    
    public static void weightingSchemeExps() throws Exception {
        String dsetGroup = "UCR";
        
        String resReadPath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/"+dsetGroup+"Results/";
        String resWritePath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/"+dsetGroup+"Results/WeightingSchemes/";
        int numfolds = 30;
        String[] dsets = GenericTools.readFileLineByLineAsArray("C:/JamesLPHD/HESCA/"+dsetGroup+"/"+dsetGroup+".txt");
        

        Instances all=null, train=null, test=null;
        Instances[] data=null;

        String[] cbases = { 
            "HESCA", 
//            "HESCA+", 
//            "HESCAks" 
        };
        
        ModuleWeightingScheme[] weights = {
//            new TrainAcc(4.0),
//            new BalancedAccuracy(4.0),
//            new AUROC(4.0),
//            new NLL(4.0),
            new FScore(4.0),
            new MCCWeighting(4.0),
//            new TrainAcc(1.0),
//            new BalancedAccuracy(1.0),
//            new AUROC(1.0),
//            new NLL(1.0),
//            new FScore(1.0),
//            new MCCWeighting(1.0),
//            new TrainAcc(8.0),
//            new BalancedAccuracy(8.0),
//            new AUROC(8.0),
//            new NLL(8.0),
//            new FScore(8.0),
//            new MCCWeighting(8.0),
        };

        for (String cbase : cbases) {
            for (ModuleWeightingScheme weight : weights) {

                String classifier = cbase + "_" + weight.toString();

                System.out.println("\t" + classifier);

                for (String dset : dsets) {

                    System.out.println(dset);

                    if (dsetGroup.equals("UCR")) {
                        train = ClassifierTools.loadData("C:/TSC Problems/" + dset + "/" + dset + "_TRAIN.arff");
                        test = ClassifierTools.loadData("C:/TSC Problems/" + dset + "/" + dset + "_TEST.arff");
                    }
                    else 
                        all = ClassifierTools.loadData("C:/UCI Problems/" + dset + "/" + dset + ".arff");

                    for (int fold = 0; fold < numfolds; fold++) {
                        String predictions = resWritePath+classifier+"/Predictions/"+dset;
                        File f=new File(predictions);
                        if(!f.exists())
                            f.mkdirs();

                        //Check whether fold already exists, if so, dont do it, just quit
                        if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){

                            if (dsetGroup.equals("UCR"))
                                data = InstanceTools.resampleTrainAndTestInstances(train, test, fold);
                            else 
                                data = InstanceTools.resampleInstances(all, fold, .5);

                            HESCA c = new HESCA();
                            c.setWeightingScheme(weight); 

                            if (cbase.equals("HESCA")) {
                                if (dsetGroup.equals("UCR"))
                                    c.setClassifiers(null, HESCAUCR_Classifiers, null);
                                //else default
                            } else if (cbase.equals("HESCA+")) {
                                c.setClassifiers(null, HESCAplus_Classifiers, null);
                            }

                            c.setBuildIndividualsFromResultsFiles(true);
                            c.setResultsFileLocationParameters(resReadPath, dset, fold);
                            c.setRandSeed(fold);
                            c.setPerformCV(true);
                            c.setResultsFileWritingLocation(resWritePath);

                            singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
                        }
                    }
                }        
            }
        }
    }
    
    public static String[] HESCAplus_Classifiers = { 
        "XGBoost500Iterations",
        "RotFDefault",
        "RandF",
        "SVMQ",
        "DNN",
    };
    public static String[] HESCAUCR_Classifiers = { 
        "NN",
        "SVML",
        "C4.5",
//        "Logistic",
        "MLP"
    };
    public static String[] HIVECOTE_Classifiers = { 
        "BOSS",
        "EE_proto",
        "RISE",
        "ST_HiveProto",
        "TSF",
    };
    public static String[] tunedSingleClassifiers = { 
        "TunedXGBoost", 
        "TunedTwoLayerMLP", 
        "TunedRandF", 
        "TunedSVMRBF"
    };
    public static String[] tunedSingleClassifiersLocations = { 
        "Z:/Results/UCIContinuous/", 
        "Z:/Results/UCIContinuous/", 
        "Z:/Results/FinalisedUCIContinuous/", 
        "Z:/Results/FinalisedUCIContinuous/"
    };
}
