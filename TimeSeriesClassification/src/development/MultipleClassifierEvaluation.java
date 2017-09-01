package development;

import utilities.ClassifierResultsAnalysis;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import utilities.ClassifierResults;
import utilities.DebugPrinting;
import utilities.ErrorReport;
import utilities.generic_storage.Pair;

/**
 * This essentially just wraps ClassifierResultsAnalysis.writeAllEvaluationFiles(...) in a nicer to use way. Will be updated over time
 * 
 * Builds summary stats, sig tests, and optionally matlab dias for the ClassifierResults objects provided/files pointed to on disk. Can optionally use
 * just the test results, if that's all that is available, or both train and test (will also compute the train test diff)
 * 
 * USAGE: Construct object, set any non-default bool options, set any non-default statistics to use, set datasets to compare on, and (rule of thumb) LASTLY add 
 * classifiers/results located in memory or on disk and call runComparison(). 
 * 
 * Least-code one-off use case that's good enough for most problems is: 
 *       new MultipleClassifierEvaluation("writePath/", "expName", numFolds).
 *          setDatasets(development.DataSets.UCIContinuousFileNames).
 *          readInClassifiers(new String[] {"NN", "C4.5"}, basePath).
 *          runComparison();  
 * 
 * Will call findAllStatsOnce on each of the ClassifierResults (i.e. will do nothing if findAllStats has already been called elsewhere before), 
 * and there's a bool (default true) to set whether to null the instance prediction info after stats are found to save memory. 
 * If some custom analysis method not defined natively in classifierresults uses the individual prediction info, 
 * (defined using addEvaluationStatistic(String statName, Function<ClassifierResults, Double> classifierResultsManipulatorFunction))
 * will need to keep the info, but that can get problematic depending on how many classifiers/datasets/folds there are
 * 
 * For some reason, the first excel workbook writer library i found/used makes xls files (instead of xlsx) and doesn't 
 * support recent excel default fonts. Just open it and saveas if you want to switch it over. There's a way to globally change font in a workbook 
 * if you want to change it back
 *
 * Future work (here and in ClassifierResultsAnalysis.writeAllEvaluationFiles(...)) when wanted/needed/motivation is available could be to 
 * handle incomplete results (e.g random folds missing), more matlab figures over time, and more refactoring of the crap parts of the code
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultipleClassifierEvaluation implements DebugPrinting { 
    private String writePath; 
    private String experimentName;
    private List<String> datasets;
    private Map<String, ClassifierResults[/* train/test */][/* dataset */][/* fold */]> classifiersResults; 
    private int numFolds;
    private ArrayList<Pair<String, Function<ClassifierResults, Double>>> statistics;
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    private boolean buildMatlabDiagrams;
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found 
     */
    private boolean cleanResults;

    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     */
    private boolean testResultsOnly;
    
    /**
     * @param experimentName forms the analysis directory name, and the prefix to most files
     */
    public MultipleClassifierEvaluation(String writePath, String experimentName, int numFolds) {
        this.writePath = writePath;
        this.experimentName = experimentName;
        this.numFolds = numFolds;
        
        this.buildMatlabDiagrams = false;
        this.cleanResults = true;
        this.testResultsOnly = false;

        this.datasets = new ArrayList<>();
        this.classifiersResults = new HashMap<>();
        
        this.statistics = ClassifierResultsAnalysis.getDefaultStatistics();
    }

    public MultipleClassifierEvaluation setTestResultsOnly(boolean b) {
        testResultsOnly = b;
        return this;
    }
    public MultipleClassifierEvaluation setBuildMatlabDiagrams(boolean b) {
        buildMatlabDiagrams = b;
        return this;
    }
    public MultipleClassifierEvaluation setCleanResults(boolean b) {
        cleanResults = b;
        return this;
    }
    
    public MultipleClassifierEvaluation setDatasets(List<String> datasets) {
        this.datasets = datasets;
        return this;
    }
    public MultipleClassifierEvaluation setDatasets(String[] datasets) {
        this.datasets = Arrays.asList(datasets);
        return this;
    }
    public MultipleClassifierEvaluation addDataset(String dataset) {
        this.datasets.add(dataset);
        return this;
    }
    public MultipleClassifierEvaluation removeDataset(String dataset) {
        this.datasets.remove(dataset);
        return this;
    }
    public MultipleClassifierEvaluation clearDatasets() {
        this.datasets.clear();
        return this;
    }

    /**
     * 4 stats: acc, balanced acc, auroc, nll
     */
    public MultipleClassifierEvaluation setUseDefaultEvaluationStatistics() {
        statistics = ClassifierResultsAnalysis.getDefaultStatistics();
        return this;
    }

    public MultipleClassifierEvaluation setUseAccuracyOnly() {
        statistics = ClassifierResultsAnalysis.getAccuracyStatisticOnly();
        return this;
    }
    
    public MultipleClassifierEvaluation setUseAllStatistics() {
        statistics = ClassifierResultsAnalysis.getAllStatistics();
        return this;
    }
    
    public MultipleClassifierEvaluation addEvaluationStatistic(String statName, Function<ClassifierResults, Double> classifierResultsManipulatorFunction) {
        statistics.add(new Pair<>(statName, classifierResultsManipulatorFunction));
        return this;
    }
    
    public MultipleClassifierEvaluation removeEvaluationStatistic(String statName) {
        for (Pair<String, Function<ClassifierResults, Double>> statistic : statistics)
            if (statistic.var1.equalsIgnoreCase(statName))
                statistics.remove(statistic);
        return this;
    }
    
    public MultipleClassifierEvaluation clearEvaluationStatistics(String statName) {
        statistics.clear();
        return this;
    }
    
    /**
     * @param trainDatasetFoldResults [dataset][fold], e.g [121][30]
     */
    public MultipleClassifierEvaluation addClassifier(String classifierName, ClassifierResults[][] trainDatasetFoldResults, ClassifierResults[][] testDatasetFoldResults) throws Exception {
        if (datasets.size() == 0) 
            throw new Exception("No datasets set for evaluation");

        for (int d = 0; d < testDatasetFoldResults.length; d++) {
            for (int f = 0; f < testDatasetFoldResults[d].length; f++) {
                if (!testResultsOnly && trainDatasetFoldResults != null) {
                    trainDatasetFoldResults[d][f].findAllStatsOnce();
                    if (cleanResults)
                        trainDatasetFoldResults[d][f].cleanPredictionInfo();
                }
                testDatasetFoldResults[d][f].findAllStatsOnce();
                if (cleanResults)
                    testDatasetFoldResults[d][f].cleanPredictionInfo();
            }
        }

        classifiersResults.put(classifierName, new ClassifierResults[][][] { trainDatasetFoldResults, testDatasetFoldResults } );
        return this;
    }
    /**
     * @param trainClassifierDatasetFoldResults [classifier][dataset][fold], e.g [5][121][30]
     */
    public MultipleClassifierEvaluation addClassifiers(String[] classifierNames, ClassifierResults[][][] trainClassifierDatasetFoldResults, ClassifierResults[][][] testClassifierDatasetFoldResults) throws Exception {
        for (int i = 0; i < classifierNames.length; i++)
            addClassifier(classifierNames[i], trainClassifierDatasetFoldResults[i], trainClassifierDatasetFoldResults[i]);            
        return this;
    }

    /**
     * Read in the results from file classifier by classifier, can be used if results are in different locations 
     * (e.g beast vs local)
     * 
     * @param classifierName Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing a subdirectory named [classifierName]
     * @return 
     */
    public MultipleClassifierEvaluation readInClassifier(String classifierName, String baseReadPath) throws Exception { 
        if (datasets.size() == 0) 
            throw new Exception("No datasets set for evaluation");

        if (baseReadPath.charAt(baseReadPath.length()-1) != '/')
            baseReadPath += "/";

        printlnDebug(classifierName + " reading");

        int totalFnfs = 0;
        ErrorReport er = new ErrorReport("FileNotFoundExceptions thrown (### total):\n");

        ClassifierResults[][][] results = new ClassifierResults[2][datasets.size()][numFolds];
        if (testResultsOnly)
            results[0]=null; //crappy but w/e
        
        for (int d = 0; d < datasets.size(); d++) {
            for (int f = 0; f < numFolds; f++) {
                
                if (!testResultsOnly) {
                    String trainFile = baseReadPath + classifierName + "/Predictions/" + datasets.get(d) + "/trainFold" + f + ".csv";
                    try {
                        results[0][d][f] = new ClassifierResults(trainFile);
                        results[0][d][f].findAllStatsOnce();
                        if (cleanResults)
                            results[0][d][f].cleanPredictionInfo();
                    } catch (FileNotFoundException ex) {
                        er.log(trainFile + "\n");
                        totalFnfs++;
                    }
                }
                
                String testFile = baseReadPath + classifierName + "/Predictions/" + datasets.get(d) + "/testFold" + f + ".csv";
                try {
                    results[1][d][f] = new ClassifierResults(testFile);
                    results[1][d][f].findAllStatsOnce();
                    if (cleanResults)
                        results[1][d][f].cleanPredictionInfo();
                } catch (FileNotFoundException ex) {
                    er.log(testFile + "\n");
                    totalFnfs++;
                }
            }
        }

        er.getLog().replace("###", totalFnfs+"");
        er.throwIfErrors();

        printlnDebug(classifierName + " successfully read in");

        classifiersResults.put(classifierName, results);
        return this;
    }
    /**
     * Read in the results from file from a common base path
     * 
     * @param classifierName Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing subdirectories with the names in classifierNames 
     * @return 
     */
    public MultipleClassifierEvaluation readInClassifiers(String[] classifierNames, String baseReadPath) throws Exception { 
        ErrorReport er = new ErrorReport("Results files not found:\n");
        for (int i = 0; i < classifierNames.length; i++) {
            try {
                readInClassifier(classifierNames[i], baseReadPath);
            } catch (Exception e) {
                er.log("Classifier Errors: " + classifierNames[i] + "\n" + e);
            }
        }
        er.throwIfErrors();
        return this;
    }
    
    public MultipleClassifierEvaluation removeClassifier(String classifierName) {
        classifiersResults.remove(classifierName);
        return this;
    }
    
    public MultipleClassifierEvaluation clearClassifiers() {
        classifiersResults.clear();
        return this;
    }
    
    public void runComparison() {
        ArrayList<ClassifierResultsAnalysis.ClassifierEvaluation> results = new ArrayList<>(classifiersResults.size());
        for (Map.Entry<String, ClassifierResults[][][]> classifier : classifiersResults.entrySet())
            results.add(new ClassifierResultsAnalysis.ClassifierEvaluation(classifier.getKey(), classifier.getValue()[1], classifier.getValue()[0], null));
        
        ClassifierResultsAnalysis.buildMatlabDiagrams = buildMatlabDiagrams;
        ClassifierResultsAnalysis.testResultsOnly = testResultsOnly;
        
        
        printlnDebug("Writing started");
        ClassifierResultsAnalysis.writeAllEvaluationFiles(writePath, experimentName, statistics, results, datasets.toArray(new String[] { }));
        printlnDebug("Writing finished");
    }

    public static void main(String[] args) throws Exception {
        String basePath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
//            String basePath = "Z:/Results/FinalisedUCIContinuous/";

        MultipleClassifierEvaluation mcc = 
            new MultipleClassifierEvaluation("C:/JamesLPHD/analysisTest/", "testrunAll7", 30);
        
        mcc.setTestResultsOnly(false); //as is default
        mcc.setBuildMatlabDiagrams(false); //as is default
        mcc.setCleanResults(true); //as is default
        mcc.setDebugPrinting(true);
        
        mcc.setUseDefaultEvaluationStatistics(); //as is default, acc,balacc,auroc,nll
//        mcc.setUseAccuracyOnly();
//        mcc.addEvaluationStatistic("F1", (ClassifierResults cr) -> {return cr.f1;}); //add on the f1 stat too
//        mcc.setUseAllStatistics();
        
        mcc.setDatasets(development.DataSets.UCIContinuousFileNames);
        
        //general rule of thumb: set/add/read the classifiers as the last thing before running
        mcc.readInClassifiers(new String[] {"NN", "C4.5"}, basePath); 
//        mcc.readInClassifier("RandF", basePath); //

        mcc.runComparison();  
    }
}
