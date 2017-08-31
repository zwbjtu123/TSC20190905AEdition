package development;

import utilities.ClassifierResultsAnalysis;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import utilities.ClassifierResults;
import utilities.DebugPrinting;
import utilities.ErrorReport;

/**
 * This essentially just wraps writeALLEvaluationFiles(...) in a nicer to use object. Will be updated over time
 * 
 * Set options, set datasets, add results located in memory or on disk and call runComparison(). Using these methods
 * will call findAllStatsOnce on each of the classifier results (i.e,, will do nothing if findAllStats has already been called), 
 * and there's a bool (default true) to set whether to null the instance prediction info after stats are found to save memory. 
 * If some custom or future analysis method not defined natively in classifierresults uses the individual prediction info, 
 * will need to keep it, but that can get problematic depending on how many classifiers/datasets/folds there are
 * 
 * For some reason, the excel workbook writer library i found/used makes xls files (instead of xlsx) and doens't 
 * support recent excel default fonts. Just open it and saveas if you want to
 
 * Future work when wanted/needed/motivation could be to handle incomplete results (e.g random folds missing), to be able to 
 * better customise what parts of the analysis is performed (e.g to only require test results), to define extra analysis through
 * function args that manipulate classifierResults objects in some way, more matlab figures over time, 
 * and a MASSIVE refactor to remove the crap code
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultipleClassifierEvaluation implements DebugPrinting { 
    private String writePath; 
    private String experimentName;
    private List<String> datasets;
    private Map<String, ClassifierResults[/* train/test */][/* dataset */][/* fold */]> classifiersResults; 
    private int numFolds;
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    private boolean buildMatlabDiagrams;
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found 
     */
    private boolean cleanResults;

    /**
     * @param experimentName forms the analysis directory name, and the prefix to most files
     */
    public MultipleClassifierEvaluation(String writePath, String experimentName, int numFolds) {
        this.writePath = writePath;
        this.experimentName = experimentName;
        this.numFolds = numFolds;
        
        this.buildMatlabDiagrams = false;
        this.cleanResults = true;

        this.datasets = new ArrayList<>();
        this.classifiersResults = new HashMap<>();
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

    /**
     * @param allDatasetFoldResults [train,test][dataset][fold], e.g [2][121][30]
     */
    public MultipleClassifierEvaluation addClassifier(String classifierName, ClassifierResults[][][] allDatasetFoldResults) throws Exception {
        if (datasets.size() == 0) 
            throw new Exception("No datasets set for evaluation");

        for (int t = 0; t < allDatasetFoldResults.length; t++) {
            for (int d = 0; d < allDatasetFoldResults[t].length; d++) {
                for (int f = 0; f < allDatasetFoldResults[t][d].length; f++) {
                    allDatasetFoldResults[t][d][f].findAllStatsOnce();
                    if (cleanResults)
                        allDatasetFoldResults[t][d][f].cleanPredictionInfo();
                }
            }
        }

        classifiersResults.put(classifierName, allDatasetFoldResults);
        return this;
    }
    /**
     * @param allDatasetFoldResults [classifier][train,test][dataset][fold], e.g [5][2][121][30]
     */
    public MultipleClassifierEvaluation addClassifiers(String[] classifierNames, ClassifierResults[][][][] classifierSetDatasetFoldResults) throws Exception {
        for (int i = 0; i < classifierNames.length; i++)
            addClassifier(classifierNames[i], classifierSetDatasetFoldResults[i]);            
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
        for (int d = 0; d < datasets.size(); d++) {
            for (int f = 0; f < numFolds; f++) {
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

    public void runComparison() {
        ArrayList<ClassifierResultsAnalysis.ClassifierEvaluation> results = new ArrayList<>(classifiersResults.size());
        for (Map.Entry<String, ClassifierResults[][][]> classifier : classifiersResults.entrySet())
            results.add(new ClassifierResultsAnalysis.ClassifierEvaluation(classifier.getKey(), classifier.getValue()[1], classifier.getValue()[0], null));

        ClassifierResultsAnalysis.writeAllEvaluationFiles(writePath, experimentName, results, datasets.toArray(new String[] { }), buildMatlabDiagrams);
    }

    public static void main(String[] args) throws Exception {
        String basePath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
//            String basePath = "Z:/Results/FinalisedUCIContinuous/";

        MultipleClassifierEvaluation mcc = 
            new MultipleClassifierEvaluation("C:/JamesLPHD/analysisTest/", "testrun6", 30).
                setDatasets(development.DataSets.UCIContinuousFileNames).
                setBuildMatlabDiagrams(true);

        mcc.setDebugPrinting(true);
        mcc.readInClassifiers(new String[] {"NN", "C4.5"}, basePath);
//        mcc.readInClassifier("RandF", basePath);

        mcc.printlnDebug("Writing started");
        mcc.runComparison();
        mcc.printlnDebug("Writing finished");
    }
}
