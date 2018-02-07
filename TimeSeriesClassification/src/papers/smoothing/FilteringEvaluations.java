
package papers.smoothing;

import development.DataSets;
import development.Experiments;
import development.MultipleClassifierEvaluation;
import java.io.File;
import java.io.FilenameFilter;
import java.nio.file.Files;
import java.util.Arrays;
import statistics.tests.TwoSampleTests;
import utilities.MultipleClassifierResultsCollection;
import utilities.StatisticalUtilities;
import vector_classifiers.ChooseClassifierFromFile;
import vector_classifiers.ChooseDatasetFromFile;

/**
 * Some functions to compute generic evaluations on completed filtering results
 * 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class FilteringEvaluations {
    public static String[] UCRDsetsNoPigs = {	
        //Train Size, Test Size, Series Length, Nos Classes
        "Adiac",        // 390,391,176,37
        "ArrowHead",    // 36,175,251,3
        "Beef",         // 30,30,470,5
        "BeetleFly",    // 20,20,512,2
        "BirdChicken",  // 20,20,512,2
        "Car",          // 60,60,577,4
        "CBF",                      // 30,900,128,3
        "ChlorineConcentration",    // 467,3840,166,3
        "CinCECGtorso", // 40,1380,1639,4
        "Coffee", // 28,28,286,2
        "Computers", // 250,250,720,2
        "CricketX", // 390,390,300,12
        "CricketY", // 390,390,300,12
        "CricketZ", // 390,390,300,12
        "DiatomSizeReduction", // 16,306,345,4
        "DistalPhalanxOutlineCorrect", // 600,276,80,2
        "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
        "DistalPhalanxTW", // 400,139,80,6
        "Earthquakes", // 322,139,512,2
        "ECG200",   //100, 100, 96
        "ECG5000",  //4500, 500,140
        "ECGFiveDays", // 23,861,136,2
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "FacesUCR", // 200,2050,131,14
        "FiftyWords", // 450,455,270,50
        "Fish", // 175,175,463,7
        "GunPoint", // 50,150,150,2
        "Ham",      //105,109,431
        "Haptics", // 155,308,1092,5
        "Herring", // 64,64,512,2
        "InlineSkate", // 100,550,1882,7
        "InsectWingbeatSound",//1980,220,256
        "ItalyPowerDemand", // 67,1029,24,2
        "LargeKitchenAppliances", // 375,375,720,3
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "Mallat", // 55,2345,1024,8
        "Meat",//60,60,448
        "MedicalImages", // 381,760,99,10
        "MiddlePhalanxOutlineCorrect", // 600,291,80,2
        "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
        "MiddlePhalanxTW", // 399,154,80,6
        "MoteStrain", // 20,1252,84,2
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "PhalangesOutlinesCorrect", // 1800,858,80,2
        "Phoneme",//1896,214, 1024
        "Plane", // 105,105,144,7
        "ProximalPhalanxOutlineCorrect", // 600,291,80,2
        "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
        "ProximalPhalanxTW", // 400,205,80,6
        "RefrigerationDevices", // 375,375,720,3
        "ScreenType", // 375,375,720,3
        "ShapeletSim", // 20,180,500,2
        "ShapesAll", // 600,600,512,60
        "SmallKitchenAppliances", // 375,375,720,3
        "SonyAIBORobotSurface1", // 20,601,70,2
        "SonyAIBORobotSurface2", // 27,953,65,2
        "Strawberry",//370,613,235
        "SwedishLeaf", // 500,625,128,15
        "Symbols", // 25,995,398,6
        "SyntheticControl", // 300,300,60,6
        "ToeSegmentation1", // 40,228,277,2
        "ToeSegmentation2", // 36,130,343,2
        "Trace", // 100,100,275,4
        "TwoLeadECG", // 23,1139,82,2
        "TwoPatterns", // 1000,4000,128,4
        "UWaveGestureLibraryX", // 896,3582,315,8
        "UWaveGestureLibraryY", // 896,3582,315,8
        "UWaveGestureLibraryZ", // 896,3582,315,8
        "Wafer", // 1000,6164,152,2
        "Wine",//54	57	234
        "WordSynonyms", // 267,638,270,25
        "Worms", //77, 181,900,5
        "WormsTwoClass",//77, 181,900,5
        "Yoga" // 300,3000,426,2
    };   
    
    public static void main(String[] args) throws Exception {
        performStandardClassifierComparisonWithFilteredAndUnfilteredDatasets();
//        selectFilterAndWriteResults();
//        selectFilterParametersAndWriteResults();
//        performSomeSimpleStatsOnWhetherFilteringIsAtAllReasonableWithTestData();


//        //just wanted to double check none where missing
//        MultipleClassifierResultsCollection mcrc = new MultipleClassifierResultsCollection(
//                new String [] { "ED", "DTWCV", "RotF" }, 
//                UCRDsetsNoPigs, 
//                30, 
//                "C:/JamesLPHD/TSC_Smoothing/Results/TSC_Exponential/", 
//                true, true, false);
    }
    
    /**
     * Assumes folder structure: 
     * some base path
     *      analysis
     *      results
     *          filtertype1
     *              classifier1
     *                  predictions
     *                      dataset1
     *                          foldfiles
     *                      dataset2
     *                      ...
     *              classifier2
     *                  ...
     *              ...
     *          filtertype2
     *              ...
     *          ...
     */
    public static  void performStandardClassifierComparisonWithFilteredAndUnfilteredDatasets() throws Exception {
        String expName = "RotFvsPCA_DFT_SG_EXP";
//        String expName = "RotFvsRotFwith(DFT_EXP_SG)vsRotFFilterANDParaSelected";
        
        String basePath = "C:/JamesLPHD/TSC_Smoothing/";
        String analysisPath = basePath + "Analysis/";
        String baseResultsPath = basePath + "Results/";
        String unfilteredResultsPath = baseResultsPath + "TSC_Unfiltered/";


        String baseClassifier = "RotF";        
        String[] filters = { "_PCAFiltered", "_DFTFiltered", "_EXPFiltered", "_SGFiltered" }; //
        String[] filterResultsFolders = { "TSCProblems_PCA_smoothed", "TSC_FFT_zeroed", "TSC_Exponential", "TSC_SavitzkyGolay" }; //
        
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation( analysisPath, expName, 30);
        mce.setBuildMatlabDiagrams(false);
//            setUseAllStatistics().
        mce.setDatasets(UCRDsetsNoPigs);

        mce.readInClassifier(baseClassifier, unfilteredResultsPath);
//        mce.readInClassifier(baseClassifier+"_FILTERED", baseResultsPath + "FilterSelected");
        for (int i = 0; i < filters.length; i++)
            mce.readInClassifier(baseClassifier + filters[i], baseResultsPath + filterResultsFolders[i]);
        
        mce.runComparison(); 
    }
    
    /**
     * this makes the '_FILTERED'-suffixed results
     * given the base classifier results (with no filter), and results for each filtered version,
     * where the parameters of the filter have already been selected (the '_XXFiltered' versions),
     * selects the best from these. i.e from the train data picks the best filter to use (given optimal parameters on train set), 
     * or no filtering at all, and copies across the top trainfold files and corresponding test files 
     */
    public static void selectFilterAndWriteResults() throws Exception {
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String[] baseDatasets = UCRDsetsNoPigs;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        
        String unfilteredResultsPath = "TSC_Unfiltered";
        String unfilteredReadPath = baseReadPath + unfilteredResultsPath + "/";
        
        String[] filterSuffixes = { "_DFTFiltered", "_EXPFiltered", "_SGFiltered" }; //"_PCAFiltered",    //"_MAFiltered",
        String[] filterResultsPaths = { unfilteredReadPath, baseReadPath+"TSC_FFT_zeroed", baseReadPath+"TSC_Exponential", baseReadPath+"TSC_SavitzkyGolay" }; //"TSCProblems_PCA_smoothed", //"TSCProblems_MovingAverage", 
        
        for (String classifier : new String [] { "RotF" }) { //"ED", "DTWCV"

            String[] classifierNames = new String[filterResultsPaths.length];
            classifierNames[0] = classifier;
            for (int i = 0; i < filterSuffixes.length; i++)
                classifierNames[i+1] = classifier + filterSuffixes[i];
            
            for (int dset = 0; dset < numBaseDatasets; dset++) {
                String baseDset = baseDatasets[dset];

                for (int fold = 0; fold < numFolds; fold++) {
                    ChooseClassifierFromFile ccff = new ChooseClassifierFromFile();
                    ccff.setName(classifier + "_FILTERED");
                    ccff.setClassifiers(classifierNames);
                    ccff.setRelationName(baseDset);
                    ccff.setResultsPath(filterResultsPaths);
                    ccff.setResultsWritePath(baseReadPath + "FilterSelected/");
                    ccff.setFold(fold);

                    ccff.buildClassifier(null);
                }
            }
        }
        
    }
    
    /**
     * this makes the '_XXFiltered'-suffixed results
     * given the classifier results for a filter with all it's different parameter settings, 
     * chooses the best 'dataset' (filter parameter) and organises the corresponding train/test fold files 
     * to mirror a single classifier on single dataset (the chosen filtered-version)
     */
    public static  void selectFilterParametersAndWriteResults() throws Exception {
//        String classifier = "ED";
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
//        String[] baseDatasets = DataSets.fileNames;
        String[] baseDatasets = UCRDsetsNoPigs;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        
        String unfilteredReadPath = baseReadPath + "TSC_Unfiltered/";
        
        String[] filterSuffixes = { "_PCAFiltered", }; //"_DFTFiltered", "_EXPFiltered", "_SGFiltered"
        String[] filterResultsPaths = { "TSCProblems_PCA_smoothed", }; //"TSC_FFT_zeroed", "TSC_Exponential", "TSC_SavitzkyGolay" 
        
        for (int i = 0; i < filterResultsPaths.length; i++) {
            String filterReadPath = baseReadPath + filterResultsPaths[i] + "/";
            String filterSuffix = filterSuffixes[i];
            
            for (String classifier : new String [] { "ED", "DTWCV", "RotF" }) { // 
                for (int dset = 0; dset < numBaseDatasets; dset++) {
                    String baseDset = baseDatasets[dset];
                    
                    
                    //COPYING BASE DATASET OVER TO FILTERED RESULTS DIRECTORY
                    File sourceLocation = new File(unfilteredReadPath + classifier + "/Predictions/" + baseDset + "/");
                    File targetLocation = new File(filterReadPath + classifier + "/Predictions/" + baseDset + "/");
                    targetLocation.mkdirs();
                    for (File foldFile : sourceLocation.listFiles())
                        Files.copy(foldFile.toPath(), (new File(targetLocation.getAbsolutePath() + "/" + foldFile.getName())).toPath());
                    //END


                    for (int fold = 0; fold < numFolds; fold++) {
                        ChooseDatasetFromFile cdff = new ChooseDatasetFromFile();
                        cdff.setName(classifier + filterSuffix);
                        cdff.setClassifier(classifier);
                        cdff.setFinalRelationName(baseDset);
                        cdff.setResultsPath(filterReadPath);
                        cdff.setFold(fold);

                        String[] datasets = (new File(filterReadPath + classifier + "/Predictions/")).list(new FilenameFilter() {
                            @Override
                            public boolean accept(File dir, String name) {
                                return name.contains(baseDset);
                            }
                        });
                        Arrays.sort(datasets);
                        if (!datasets[0].equals(baseDset))
                            throw new Exception("hwut" + baseDset  +"/n" + Arrays.toString(datasets));
                        cdff.setRelationNames(datasets);

                        cdff.buildClassifier(null);
                    }
                    
                    
                    //DELETING COPIED BASE DATASET FROM FILERED RESULTS DIRECTORY        
                        for (File foldFile : targetLocation.listFiles())
                            foldFile.delete();
                        targetLocation.delete();
                    //END


                }
            }
        }
    }
    
    public static void performSomeSimpleStatsOnWhetherFilteringIsAtAllReasonableWithTestData() throws Exception { 
        final double P_VAL = 0.05;
        
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String[] classifiers = { "ED" };
        String[] baseDatasets = DataSets.fileNames;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        boolean testResultsOnly = false;
        boolean cleanResults = true;
        boolean allowMissing = false;
        
        MultipleClassifierResultsCollection[] mcrcs = new MultipleClassifierResultsCollection[numBaseDatasets];
        boolean [] aFilteredVersionIsSigBetter = new boolean[numBaseDatasets];
        boolean [] aFilteredVersionIsBetter = new boolean[numBaseDatasets];
        boolean [] unFilteredVersionIsSigBetterThanAllFiltered = new boolean[numBaseDatasets];
//        boolean [] unFilteredVersionIsBetterThanAllFiltered = new boolean[numBaseDatasets];
        
        for (int i = 0; i < numBaseDatasets; i++) {
            String datasetBase = baseDatasets[i];
            String[] datasets = (new File(baseReadPath + classifiers[0] + "/Predictions/")).list(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.contains(datasetBase);
                }
            });
            Arrays.sort(datasets);
            if (!datasets[0].equals(datasetBase))
                throw new Exception("hwut" + datasetBase  +"/n" + Arrays.toString(datasets));
            
            MultipleClassifierResultsCollection mcrc = new MultipleClassifierResultsCollection(classifiers, datasets, numFolds, baseReadPath, testResultsOnly, cleanResults, allowMissing);
            mcrcs[i] = mcrc;
            
            double[][] resFolds = mcrc.getAccuracies()[1][0]; // [test][firstclassifier]
            double[] resDsets = StatisticalUtilities.averageFinalDimension(resFolds); 
            
            double unfilteredAcc = resDsets[0];
            
            boolean allFilteredAreSigWorse = true;
            for (int j = 1; j < resDsets.length; j++) {
                double p = TwoSampleTests.studentT_PValue(resFolds[0], resFolds[j]);
                if (resDsets[j] > unfilteredAcc) {
                    aFilteredVersionIsBetter[i] = true;
                    if (p < P_VAL) 
                        aFilteredVersionIsSigBetter[i] = true;
                }
                else {
                    if (p > P_VAL)
                        allFilteredAreSigWorse = false;
                }
            }
            unFilteredVersionIsSigBetterThanAllFiltered[i] = allFilteredAreSigWorse;
        }    
        
        System.out.println("aFilteredVersionIsSigBetter: " + countNumTrue(aFilteredVersionIsSigBetter) );
        System.out.println("aFilteredVersionIsBetter: " + countNumTrue(aFilteredVersionIsBetter) );
        System.out.println("unFilteredVersionIsSigBetterThanAllFiltered: " + countNumTrue(unFilteredVersionIsSigBetterThanAllFiltered) );
    }
    
    public static int countNumTrue(boolean[] arr) { 
        int counter = 0;
        for (boolean b : arr)
            if (b) counter++;
        return counter;
    }
}
