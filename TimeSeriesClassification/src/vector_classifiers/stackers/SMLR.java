
package vector_classifiers.stackers;

import development.CollateResults;
import development.DataSets;
import static development.Experiments.singleClassifierAndFoldTrainTestSplit;
import development.MultipleClassifierEvaluation;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import timeseriesweka.classifiers.ensembles.voting.stacking.StackingOnDists;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import utilities.ClassifierTools;
import utilities.GenericTools;
import utilities.InstanceTools;
import vector_classifiers.HESCA;
import vector_classifiers.MultiLinearRegression;
import vector_classifiers.weightedvoters.HESCA_MajorityVote;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Stacking with multi-response linear regression (MLR), Ting and Witten (1999) 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SMLR extends HESCA {
    public SMLR() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "SMLR"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new StackingOnDists(new MultiLinearRegression());
    }     
    
    public static void main(String[] args) throws Exception {
        exps();
//        ana();
//        cluster(args);
    }
    
    public static void ana() throws Exception {
        String dsetGroup = "UCI";
        
        String basePath = "C:/JamesLPHD/HESCA/"+dsetGroup+"/";
//        String[] datasets = (new File(basePath + "Results/DTWCV/Predictions/")).list();
        
        new MultipleClassifierEvaluation(basePath+"XGBoostAnalysis/", dsetGroup+"smlrvshesca", 30).
            setTestResultsOnly(true).
//            setBuildMatlabDiagrams(true).
//            setDatasets(dsetGroup.equals("UCI") ? development.DataSets.UCIContinuousFileNames : development.DataSets.fileNames).
            setDatasets(basePath + dsetGroup + ".txt").
//            readInClassifiers(new String[] { "MLR", "MRMT", "1NN", "C4.5", }, basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA", "SMM5", "SMLRE", "SMLR"}, basePath+dsetGroup+"Results/").
            readInClassifiers(new String[] { "HESCA", "SMLR"}, basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    public static void cluster(String[] args) {
        String classifier = args[0];
        String dset = args[1];
        int fold = Integer.parseInt(args[2])-1;
        
        String resPath = "Results/UCIContinuous/";
        
////        String classifier = "SMLR";
//        String classifier = "SMLRE";
////        String classifier = "SMM5";
        

        Instances all = ClassifierTools.loadData("UCIContinuous/" + dset + "/" + dset + ".arff");

        String predictions = resPath+classifier+"/Predictions/"+dset;
        File f=new File(predictions);
        if(!f.exists())
            f.mkdirs();

        //Check whether fold already exists, if so, dont do it, just quit
        if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
            Instances[] data = InstanceTools.resampleInstances(all, fold, .5);

////                    SMLR c = new SMLR();
//            SMLRE c = new SMLRE();
////                    SMM5 c = new SMM5();

            
            HESCA c = null;
            try {
                c = (HESCA) Class.forName("vector_classifiers.stackers." + classifier).newInstance();
            } catch (Exception e) {
                System.out.println("balls: " + e);
                System.exit(1);
            }

            c.setClassifiers(null, HESCA_MajorityVote.HESCAplus_Classifiers, null);

            
            c.setBuildIndividualsFromResultsFiles(true);
            c.setResultsFileLocationParameters(resPath, dset, fold);
            c.setRandSeed(fold);
            c.setPerformCV(true);
            c.setResultsFileWritingLocation(resPath);

            singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
        }

    }
    
    
    public static void exps() throws Exception {
        String resPath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
        int numfolds = 5;
        
        ArrayList<String> plantdsets = new ArrayList<>();
        plantdsets.add("plant-margin");
        plantdsets.add("plant-shape");
        plantdsets.add("plant-texture"); //have lot of classes, makes smlr and smlre super slow, LOTS of regression
        
//        String[] dsets = DataSets.UCIContinuousFileNames;
        String[] dsets = GenericTools.readFileLineByLineAsArray("C:/JamesLPHD/HESCA/UCI/UCI_4biggunsRemoved.txt");
//        String[] dsets = { "zoo" };
        
//        String classifier = "SMLR_OverTunedClassifiers";
        String classifier = "SMLRE_OverTunedClassifiers";
//        String classifier = "SMM5_OverTunedClassifiers";
        
        System.out.println("\t" + classifier);
        
//        for (String dset : dsets) {
//            if (plantdsets.contains(dset))
//                continue;
//            
//            System.out.println(dset);
//            
//            Instances all = ClassifierTools.loadData("C:/UCI Problems/" + dset + "/" + dset + ".arff");
//            
//            for (int fold = 0; fold < numfolds; fold++) {
//                String predictions = resPath+classifier+"/Predictions/"+dset;
//                File f=new File(predictions);
//                if(!f.exists())
//                    f.mkdirs();
//        
//                //Check whether fold already exists, if so, dont do it, just quit
//                if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
//                    Instances[] data = InstanceTools.resampleInstances(all, fold, .5);
//                    
//                    
////                    SMLR c = new SMLR();
////                    SMLRE c = new SMLRE();
//                    SMM5 c = new SMM5();
//
//                    c.setClassifiers(null, HESCA_MajorityVote.tunedSingleClassifiers, null);
//                    c.setResultsFileLocationParameters(HESCA_MajorityVote.tunedSingleClassifiersLocations, dset, fold);
//                        
//                    c.setBuildIndividualsFromResultsFiles(true);
////                    c.setResultsFileLocationParameters(resPath, dset, fold);
//                    c.setRandSeed(fold);
//                    c.setPerformCV(true);
//                    c.setResultsFileWritingLocation(resPath);
//                    
//                    singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
//                }
//            }
//        }
        for (String dset : plantdsets) {
            
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
                    
                    
//                    SMLR c = new SMLR();
                    SMLRE c = new SMLRE();
//                    SMM5 c = new SMM5();

                    c.setClassifiers(null, HESCA_MajorityVote.tunedSingleClassifiers, null);
                    c.setResultsFileLocationParameters(HESCA_MajorityVote.tunedSingleClassifiersLocations, dset, fold);
                    
                    c.setBuildIndividualsFromResultsFiles(true);
//                    c.setResultsFileLocationParameters(resPath, dset, fold);
                    c.setRandSeed(fold);
                    c.setPerformCV(true);
                    c.setResultsFileWritingLocation(resPath);
                    
                    singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
                }
            }
        }
        
    }
}