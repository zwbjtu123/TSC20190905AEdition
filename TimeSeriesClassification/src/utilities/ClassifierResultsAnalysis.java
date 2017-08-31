
package utilities;

import ResultsProcessing.MatlabController;
import development.MultipleClassifiersPairwiseTest;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;
import java.util.Scanner;
import jxl.Workbook;
import jxl.WorkbookSettings;
import jxl.write.WritableCellFormat;
import jxl.write.WritableFont;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import statistics.tests.TwoSampleTests;
import utilities.ClassifierResults;
import utilities.StatisticalUtilities;

/**
 *
 * Basically functions to analyse/handle COMPLETED test AND train results in ClassifierResults format
 * 
 * For some reason, the excel workbook writer library i found/used makes xls files (instead of xlsx) and doens't 
 * support recent excel default fonts. Just open it and saveas if you want to
 
 Future work when wanted/needed/motivation could be to handle incomplete results (e.g random folds missing), to be able to 
 better customise what parts of the analysis is performed (e.g to only require test results), to define extra analysis through
 function args that manipulate classifierResults objects in some way, more matlab figures over time, 
 and a MASSIVE refactor to remove the crap code
 
 Two ways to use, the big one-off static method writeALLEvaluationFiles(...) if you have everything in memory
 in the correct format already (it is assumed that findAllStats on each classifierResults has already been called, and 
 if desired any results cleaning/nulling for space has already occurred)
 
 or you'd normally use development.MultipleClassifierEvaluation to set up results located in memory or
 on disk and call runComparison(), which essentially just wraps writeALLEvaluationFiles(...). Using this method 
 will call findAllStats on each of the classifier results, and there's a bool (default true) to set whether 
 to null the instance prediction info after stats are found to save memory. if some custom or future analysis 
 method not defined natively in classifierresults uses the individual prediction info, will need to keep it 
 * 
 * @author James Large james.large@uea.ac.uk
 */


public class ClassifierResultsAnalysis {
    
    public static double FREIDMANCDDIA_PVAL = 0.05;
        
    public static class ClassifierEvaluation  {
        public String classifierName;
        public ClassifierResults[][] testResults; //[dataset][fold]
        public ClassifierResults[][] trainResults; //[dataset][fold]
        public BVOnDset[] bvs; //bias variance decomposition, each element for one dataset
        
        
        public ClassifierEvaluation(String name, ClassifierResults[][] testResults, ClassifierResults[][] trainResults, BVOnDset[] bvs) {
            this.classifierName = name;
            this.testResults = testResults;
            this.trainResults = trainResults;
            this.bvs = bvs;
        }
    }
    
    public static class BVOnDset {
        //bias variance decomposition for a single classifier on a single dataset
        
        public int numClasses;
        public int[] trueClassVals; //[dataset][inst]
        public ArrayList<Integer>[] allPreds; //[dataset][inst][fold]
                
        public BVOnDset(int[] trueClassVals, int numClasses) {
            this.trueClassVals = trueClassVals;
            this.numClasses = numClasses;
            
            allPreds = new ArrayList[trueClassVals.length];
            for (int i = 0; i < allPreds.length; i++)
                allPreds[i] = new ArrayList<>();
        }
        
        /**
         * stores the test predictions on a single fold of this dataset
         */
        public void storePreds(int[] testInds, int[] testPreds) {
            for (int i = 0; i < testInds.length; i++)
                allPreds[testInds[i]].add(testPreds[i]);
        }
        
        public int[][] getFinalPreds() {
            int[][] res = new int[allPreds.length][] ;
            for (int i = 0; i < res.length; i++) {
                res[i] = new int[allPreds[i].size()];
                for (int j = 0; j < allPreds[i].size(); j++) 
                    res[i][j] = allPreds[i].get(j);
            }
            return res;
        }
    }
    
    protected static void writeTableFile(String filename, String tableName, double[][] accs, String[] cnames, String[] dsets) {
        OutFile out=new OutFile(filename);
        out.writeLine(tableName + ":" + tabulate(accs, cnames, dsets));
//        out.writeLine("\navg:" + average(accs));
        out.closeFile();
    }
    
    protected static void writeTableFileRaw(String filename, double[][] accs, String[] cnames) {
        OutFile out=new OutFile(filename);
        out.writeLine(tabulateRaw(accs, cnames));
        out.closeFile();
    }
    
    /**
     * also writes separate win/draw/loss files now
     */
    protected static String[] writeSummaryFile(String outPath, String filename, String statistic, double[][][] statPerFold, double[][] statPerDset, double[][] ranks, double[][] stddevsFoldAccs, String[] cnames, String[] dsets) {   
        StringBuilder suppressedSummaryStats = new StringBuilder();
        suppressedSummaryStats.append(header(cnames)).append("\n");
        suppressedSummaryStats.append("Avg"+statistic+":").append(average(statPerDset)).append("\n");
        suppressedSummaryStats.append("Avg"+statistic+"_RANK:").append(average(ranks)).append("\n");
        
        StringBuilder summaryStats = new StringBuilder();
        
        summaryStats.append(header(cnames)).append("\n");
        summaryStats.append("avgOverDsets:").append(average(statPerDset)).append("\n");
        summaryStats.append("stddevOverDsets:").append(stddev(statPerDset)).append("\n");
        summaryStats.append("avgStddevsOverFolds:").append(average(stddevsFoldAccs)).append("\n");
        summaryStats.append("avgRankOverDsets:").append(average(ranks)).append("\n");
        summaryStats.append("stddevsRankOverDsets:").append(stddev(ranks)).append("\n");

        String[] wdl = winsDrawsLosses(statPerDset, cnames, dsets);
        String[] sig01wdl = sigWinsDrawsLosses(0.01, statPerDset, statPerFold, cnames, dsets);
        String[] sig05wdl = sigWinsDrawsLosses(0.05, statPerDset, statPerFold, cnames, dsets);
        
        (new File(outPath+"/WinsDrawsLosses/")).mkdir();
        OutFile outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLFLAT_"+statistic+".csv");
        outwdl.writeLine(wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLSig01_"+statistic+".csv");
        outwdl.writeLine(sig01wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLSig05_"+statistic+".csv");
        outwdl.writeLine(sig05wdl[1]);
        outwdl.closeFile();
        
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLFLAT_"+statistic+".csv");
        outwdl.writeLine(wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLSig01_"+statistic+".csv");
        outwdl.writeLine(sig01wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLSig05_"+statistic+".csv");
        outwdl.writeLine(sig05wdl[2]);
        outwdl.closeFile();
        
        
        OutFile out=new OutFile(outPath+filename+"_"+statistic+"_SUMMARY.csv");
        
        out.writeLine(summaryStats.toString());
        
        out.writeLine(wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig01wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig05wdl[0]);
        out.writeLine("\n");
        
        String cliques = "";
        try {
            System.out.println(filename+"_"+statistic+".csv");
            out.writeLine(MultipleClassifiersPairwiseTest.runTests(outPath+filename+"_"+statistic+".csv").toString());       
            cliques = MultipleClassifiersPairwiseTest.printCliques();
            out.writeLine("\n\n" + cliques);
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        out.closeFile();
        
        return new String[] { summaryStats.toString(), suppressedSummaryStats.toString(), cliques };
    }
    
    protected static String cdFileName(String filename, String statistic) {
        return "cd_"+filename+"_"+statistic+"S";
    }
    
    protected static String[] writeEvaluationFiles(String outPath, String filename, String statistic, double[][][] trainFolds, double[][][] testFolds, String[] cnames, String[] dsets) {
        new File(outPath).mkdirs();
        
        //dataprep
        double[][][] trainTestDiffsFolds = findTrainTestDiffs(trainFolds, testFolds);
        double[][] trainTestDiffsDsets = findAvgsOverFolds(trainTestDiffsFolds); //method is now a bit misnamed but w/e, it jsut avgs the 3rd dimension
        double[][] stddevsTrainTestDiffsOverFolds = findStddevsOverFolds(trainTestDiffsFolds); //and same
        
        double[][] trainDsets = findAvgsOverFolds(trainFolds);
        double[][] testDsets = findAvgsOverFolds(testFolds);
        double[][] ranks = findRanks(testDsets);
        double[][] stddevsTrainFoldAccs = findStddevsOverFolds(trainFolds);
        double[][] stddevsTestFoldAccs = findStddevsOverFolds(testFolds);

        //BEFORE ordering, write the individual folds files
        writePerFoldFiles(outPath+"TRAINFOLD"+statistic+"S/", trainFolds, cnames, dsets, "train");
        writePerFoldFiles(outPath+"TESTFOLD"+statistic+"S/", testFolds, cnames, dsets, "test");
        writePerFoldFiles(outPath+"TRAINTESTDIFFFOLD"+statistic+"S/", trainTestDiffsFolds, cnames, dsets, "trainTestDiff");
                
        //ordering
        int[] ordering = findOrdering(ranks);
        ranks = order(ranks, ordering);
        cnames = order(cnames, ordering);
        
        trainFolds = order(trainFolds, ordering);
        testFolds = order(testFolds, ordering);
        
        trainDsets = order(trainDsets, ordering);
        testDsets = order(testDsets, ordering);
        trainTestDiffsDsets = order(trainTestDiffsDsets, ordering);
        
        stddevsTrainFoldAccs = order(stddevsTrainFoldAccs, ordering);
        stddevsTestFoldAccs = order(stddevsTestFoldAccs, ordering);
        stddevsTrainTestDiffsOverFolds = order(stddevsTrainTestDiffsOverFolds, ordering);
        
        //writing
        writeTableFile(outPath+filename+"_TRAIN"+statistic+"S.csv", "accs", trainDsets, cnames, dsets);
        writeTableFileRaw(outPath+filename+"_TRAIN"+statistic+"SRAW.csv", trainDsets, cnames); //for matlab stuff
        writeTableFile(outPath+filename+"_TRAIN"+statistic+"STDDEVS.csv", "accs", stddevsTrainFoldAccs, cnames, dsets);
        
        writeTableFile(outPath+filename+"_TEST"+statistic+"S.csv", "accs", testDsets, cnames, dsets);
        writeTableFileRaw(outPath+filename+"_TEST"+statistic+"SRAW.csv", testDsets, cnames); //for matlab stuff
        writeTableFile(outPath+filename+"_TEST"+statistic+"STDDEVS.csv", "accs", stddevsTestFoldAccs, cnames, dsets);
        
        //qol for cd dia creation, make a copy of all the raw test stat files in a common folder, one for pairwise, one for freidman
        File f = new File(outPath);
        String cdFolder = f.getParent() + "/cddias/";
        (new File(cdFolder)).mkdirs();
        OutFile out = new OutFile(cdFolder+"readme.txt");
        out.writeLine("remember that nlls are auto-negated now\n");
        out.writeLine("and that basic notepad wont show the line breaks properly, view in notepad++");
        out.closeFile();
        for (String subFolder : new String[] { "pairwise", "freidman" }) {
            (new File(cdFolder+subFolder+"/")).mkdirs();
            String cdName = cdFolder+subFolder+"/"+cdFileName(filename,statistic)+".csv";
            //meta hack for qol, negate the nll (sigh...) for correct ordering on dia
            if (statistic.contains("NLL")) {
                double[][] negatedTestDsets = new double[testDsets.length][testDsets[0].length];
                for (int i = 0; i < testDsets.length; i++) {
                    for (int j = 0; j < testDsets[i].length; j++) {
                        negatedTestDsets[i][j] = testDsets[i][j] * -1;
                    }
                }
                writeTableFileRaw(cdName, negatedTestDsets, cnames);
            } else {
                writeTableFileRaw(cdName, testDsets, cnames);
            } 
        } //end qol
        
        //back to normal
        writeTableFile(outPath+filename+"_TRAINTEST"+statistic+"DIFFS.csv", "traintestdiffs", trainTestDiffsDsets, cnames, dsets);
        writeTableFileRaw(outPath+filename+"_TRAINTEST"+statistic+"DIFFSRAW.csv", trainTestDiffsDsets, cnames); //for matlab stuff
        writeTableFile(outPath+filename+"_TRAINTEST"+statistic+"DIFFSSTDDEVS.csv", "traintestdiffs", stddevsTrainTestDiffsOverFolds, cnames, dsets);
        writeSummaryFile(outPath, filename, "TRAINTEST"+statistic+"DIFFS", trainTestDiffsFolds, trainTestDiffsDsets, ranks, stddevsTrainTestDiffsOverFolds, cnames, dsets); //ranks are still based on the primary statistic
        
        writeTableFile(outPath+filename+"_"+statistic+"RANKS.csv", "ranks", ranks, cnames, dsets);
        
        return writeSummaryFile(outPath, filename, "TEST"+statistic+"S", testFolds, testDsets, ranks, stddevsTestFoldAccs, cnames, dsets);
    }
    
    public static void writeAllEvaluationFiles(String outPath, String expname, ArrayList<ClassifierEvaluation> results, String[] dsets, boolean makeCDDias) {
        new File(outPath).mkdirs();
        
        String[] accSummaries = writeAccuracyEvaluationFiles(outPath, expname, results, dsets);
        String[] baccSummaries = writeBalancedAccuracyEvaluationFiles(outPath, expname, results, dsets);
        String[] aurocSummaries = writeAUROCEvaluationFiles(outPath, expname, results, dsets);
        String[] nllSummaries = writeNLLEvaluationFiles(outPath, expname, results, dsets);
          
        OutFile bigSummary = new OutFile(outPath + expname + "_BIGglobalSummary.csv");
        
        bigSummary.writeLine("ACCURACY:");
        bigSummary.writeLine(accSummaries[0]);
        bigSummary.writeLine("BALANCED ACCURACY:");
        bigSummary.writeLine(baccSummaries[0]);
        bigSummary.writeLine("AUROC:");
        bigSummary.writeLine(aurocSummaries[0]);
        bigSummary.writeLine("NLL:");
        bigSummary.writeLine(nllSummaries[0]);
        
        bigSummary.closeFile();
        
        OutFile smallSummary = new OutFile(outPath + expname + "_SMALLglobalSummary.csv");
        
        smallSummary.writeString("ACCURACY:");
        smallSummary.writeLine(accSummaries[1]);
        smallSummary.writeString("BALANCED ACCURACY:");
        smallSummary.writeLine(baccSummaries[1]);
        smallSummary.writeString("AUROC:");
        smallSummary.writeLine(aurocSummaries[1]);
        smallSummary.writeString("NLL:");
        smallSummary.writeLine(nllSummaries[1]);
        
        smallSummary.closeFile();
        
        buildResultsSpreadsheet(outPath, expname);
        
        if(makeCDDias) {
            String [] cliques = { accSummaries[2], baccSummaries[2], aurocSummaries[2], nllSummaries[2] };
            String [] stats = { "ACC", "BALACC", "AUROC", "NLL" };
            buildPairwiseCDDias(outPath + "/cdDias/pairwise/", expname, stats, cliques);
            buildFreidmanCDDias(outPath + "/cdDias/friedman/");
        }
    }
        
    protected static void buildFreidmanCDDias(String cdCSVpath) {
        MatlabController proxy = MatlabController.getInstance();
        proxy.eval("buildDiasInDirectory('"+cdCSVpath+"', 0, "+FREIDMANCDDIA_PVAL+")");
    }
    protected static void buildPairwiseCDDias(String cdCSVpath, String expname, String[] stats, String[] cliques) {
        //temp workaround, just write the cliques and readin again from matlab for ease of checking.editing for pairwise edge cases
        for (int i = 0; i < stats.length; i++) {
            OutFile out = new OutFile (cdCSVpath + cdFileName(expname, stats[i]) + "_cliques.txt");
            out.writeString(cliques[i]);
            out.closeFile();
        }
        
        MatlabController proxy = MatlabController.getInstance();
        proxy.eval("buildDiasInDirectory('"+cdCSVpath+"', 1)"); 
    }
        
    protected static String[] writeAccuracyEvaluationFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, String[] dsets) {
        outPath += "Accuracy/";
        
        String[] cnames = getNames(results);
        
        double[][][] trainFolds = getAccs(results, "train");
        double[][][] testFolds = getAccs(results, "test");
        
        return writeEvaluationFiles(outPath, filename, "ACC", trainFolds, testFolds, cnames, dsets);
    }
    
    protected static String[] writeBalancedAccuracyEvaluationFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, String[] dsets) {
        outPath += "BalancedAccuracy/";
        
        String[] cnames = getNames(results);
        
        double[][][] trainFolds = getBalAccs(results, "train");
        double[][][] testFolds = getBalAccs(results, "test");
        
        return writeEvaluationFiles(outPath, filename, "BALACC", trainFolds, testFolds, cnames, dsets);
    }
    
    protected static String[] writeAUROCEvaluationFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, String[] dsets) {
        outPath += "AUROC/";
        
        String[] cnames = getNames(results);
        
        double[][][] trainFolds = getAUROCs(results, "train");
        double[][][] testFolds = getAUROCs(results, "test");
        
        return writeEvaluationFiles(outPath, filename, "AUROC", trainFolds, testFolds, cnames, dsets);
    }
    
    protected static String[] writeNLLEvaluationFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, String[] dsets) {
        outPath += "NLL/";
        
        String[] cnames = getNames(results);
        
        double[][][] trainFolds = getNLLs(results, "train");
        double[][][] testFolds = getNLLs(results, "test");
        
        return writeEvaluationFiles(outPath, filename, "NLL", trainFolds, testFolds, cnames, dsets);
    }
    
    protected static void writePerFoldFiles(String outPath, double[][][] folds, String[] cnames, String[] dsets, String trainTest) {
        new File(outPath).mkdirs();
        
        StringBuilder headers = new StringBuilder("folds:");
        for (int f = 0; f < folds[0][0].length; f++)
            headers.append(","+f);
        
        for (int c = 0; c < folds.length; c++) {
            OutFile out=new OutFile(outPath + cnames[c]+"_"+trainTest.toUpperCase()+"FOLDS.csv");
            out.writeLine(headers.toString());
            
            for (int d = 0; d < folds[c].length; d++) {
                out.writeString(dsets[d]);
                for (int f = 0; f < folds[c][d].length; f++)
                    out.writeString("," + folds[c][d][f]);
                out.writeLine("");
            }
            
            out.closeFile();
        }
        
        
        OutFile out = new OutFile(outPath + "TEXASPLOT_"+trainTest.toUpperCase()+".csv");
        out.writeString(cnames[0]);
        for (int c = 1; c < cnames.length; c++)
            out.writeString("," + cnames[c]);
        out.writeLine("");
        
        for (int d = 0; d < dsets.length; d++) {
            for (int f = 0; f < folds[0][0].length; f++) {
                out.writeDouble(folds[0][d][f]);
                for (int c = 1; c < cnames.length; c++)
                    out.writeString("," + folds[c][d][f]);
                out.writeLine("");
            }
        }
        out.closeFile();
    }
    
    protected static String tabulate(double[][] res, String[] cnames, String[] dsets) {
        StringBuilder sb = new StringBuilder();
        sb.append(header(cnames));
        
        for (int i = 0; i < res[0].length; ++i) {
            sb.append("\n").append(dsets[i]);

            for (int j = 0; j < res.length; j++)
                sb.append("," + res[j][i]);
        }      
        return sb.toString();
    }
    
    protected static String tabulateRaw(double[][] res, String[] cnames) {
        StringBuilder sb = new StringBuilder();
        sb.append(header(cnames).substring(1));
        
        for (int i = 0; i < res[0].length; ++i) {
            sb.append("\n").append(res[0][i]);
            for (int j = 1; j < res.length; j++)
                sb.append("," + res[j][i]);
        }      
        return sb.toString();
    }
    
    protected static String header(String[] names) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < names.length; i++)
            sb.append(",").append(names[i]);
        return sb.toString();
    }
    
    protected static String average(double[][] res) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) 
            sb.append(",").append(StatisticalUtilities.mean(res[i], false));
        
        return sb.toString();
    }
    
    protected static String stddev(double[][] res) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) 
            sb.append(",").append(StatisticalUtilities.standardDeviation(res[i], false, StatisticalUtilities.mean(res[i], false)));
        
        return sb.toString();
    }
    
    protected static double[][][] findTrainTestDiffs(double[][][] trainFoldAccs, double[][][] testFoldAccs) {
        double[][][] diffs = new double[trainFoldAccs.length][trainFoldAccs[0].length][trainFoldAccs[0][0].length];
        
        for (int c = 0; c < diffs.length; c++) 
            for (int d = 0; d < diffs[c].length; d++) 
                for (int f = 0; f < diffs[c][d].length; f++) 
                    diffs[c][d][f] =  trainFoldAccs[c][d][f] - testFoldAccs[c][d][f];
        
        return diffs;
    }
        
    protected static double[][] findAvgsOverFolds(double[][][] foldaccs) {
        double[][] accs = new double[foldaccs.length][foldaccs[0].length];
        for (int i = 0; i < accs.length; i++)
            for (int j = 0; j < accs[i].length; j++)
                accs[i][j] = StatisticalUtilities.mean(foldaccs[i][j], false);
        
        return accs;
    }
    
    protected static double[][] findStddevsOverFolds(double[][][] foldaccs) {
        double[][] devs = new double[foldaccs.length][foldaccs[0].length];
        for (int i = 0; i < devs.length; i++)
            for (int j = 0; j < devs[i].length; j++)
                devs[i][j] = StatisticalUtilities.standardDeviation(foldaccs[i][j], false, StatisticalUtilities.mean(foldaccs[i][j], false));
        
        return devs;
    }
    
    public static int[] findOrdering(double[][] r) {
        double[] avgranks = new double[r.length];
        for (int i = 0; i < r.length; i++) 
            avgranks[i] = StatisticalUtilities.mean(r[i], false);
        
        double[] cpy = Arrays.copyOf(avgranks, avgranks.length);
        
        int[] res = new int[avgranks.length];
        
        int i = 0;
        while (i < res.length) {
            ArrayList<Integer> mins = min(avgranks);
            
            for (int j = 0; j < mins.size(); j++) {
                res[mins.get(j)] = i++;
                avgranks[mins.get(j)] = Double.MAX_VALUE;
            }
        }
        
        return res;
    }
    
    protected static int[] findReverseOrdering(double[][] r) {
        double[] avgranks = new double[r.length];
        for (int i = 0; i < r.length; i++) 
            avgranks[i] = StatisticalUtilities.mean(r[i], false);
        
        double[] cpy = Arrays.copyOf(avgranks, avgranks.length);
        
        int[] res = new int[avgranks.length];
        
        int i = 0;
        while (i < res.length) {
            ArrayList<Integer> maxs = max(avgranks);
            
            for (int j = 0; j < maxs.size(); j++) {
                res[maxs.get(j)] = i++;
                avgranks[maxs.get(j)] = -Double.MAX_VALUE;
            }
        }
        
        return res;
    }
    
    protected static ArrayList<Integer> min(double[] d) {
        double min = d.length+1;
        ArrayList<Integer> minIndices = null;
        
        for (int c = 0; c < d.length; c++) {
            if(d[c] < min){
                min = d[c];
                minIndices = new ArrayList<>();
                minIndices.add(c);
            }else if(d[c] == min){
                minIndices.add(c);
            }
        }
        
        return minIndices;
    }
    
    protected static ArrayList<Integer> max(double[] d) {
        double max = -1;
        ArrayList<Integer> maxIndices = null;
        
        for (int c = 0; c < d.length; c++) {
            if(d[c] > max){
                max = d[c];
                maxIndices = new ArrayList<>();
                maxIndices.add(c);
            }else if(d[c] == max){
                maxIndices.add(c);
            }
        }
        
        return maxIndices;
    }
    
    public static String[] order(String[] s, int[] ordering) {
        String[] res = new String[s.length];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    public static double[][] order(double[][] s, int[] ordering) {
        double[][] res = new double[s.length][];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    public static double[][][] order(double[][][] s, int[] ordering) {
        double[][][] res = new double[s.length][][];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    /**
     * @param accs [classifiers][acc on datasets]
     * @return [classifiers][rank on dataset]
     */
    public static double[][] findRanks(double[][] accs) {
        double[][] ranks = new double[accs.length][accs[0].length];
        
        for (int d = 0; d < accs[0].length; d++) {
            Double[] a = new Double[accs.length];
            for (int c = 0; c < accs.length; c++) 
                a[c] = accs[c][d];
            
            Arrays.sort(a, Collections.reverseOrder());
            
            int numTies = 0;
            double lastAcc = -1;
            for (int c1 = 0; c1 < accs.length; c1++) {
                for (int c2 = 0; c2 < accs.length; c2++) {
                    if (a[c1] == accs[c2][d]) {
                        ranks[c2][d] = c1; //count from one
                    }
                }
            }
            
            //correcting ties
            int[] hist = new int[accs.length];
            for (int c = 0; c < accs.length; c++)
                ++hist[(int)ranks[c][d]];
            
            for (int r = 0; r < hist.length; r++) {
                if (hist[r] > 1) {//ties
                    double newRank = 0;
                    for (int i = 0; i < hist[r]; i++)
                        newRank += r-i;
                    newRank/=hist[r];
                    for (int c = 0; c < ranks.length; c++)
                        if (ranks[c][d] == r) 
                            ranks[c][d] = newRank;
                }
            }
            
            //correcting for index from 1
            for (int c = 0; c < accs.length; c++)
                ++ranks[c][d];
        }
        
        return ranks;
    }
    
    protected static String[] winsDrawsLosses(double[][] accs, String[] cnames, String[] dsets) {
        StringBuilder table = new StringBuilder();
        ArrayList<ArrayList<ArrayList<String>>> wdlList = new ArrayList<>(); //[classifierPairing][win/draw/loss][dsetNames]
        ArrayList<String> wdlListNames = new ArrayList<>();
        
        String[][] wdlPlusMinus = new String[cnames.length*cnames.length][dsets.length];
        
        table.append("flat" + header(cnames)).append("\n");
        
        int count = 0;
        for (int c1 = 0; c1 < accs.length; c1++) {
            table.append(cnames[c1]);
            for (int c2 = 0; c2 < accs.length; c2++) {
                wdlListNames.add(cnames[c1] + "_VS_" + cnames[c2]);
                wdlList.add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());                
                
                int wins=0, draws=0, losses=0;
                for (int d = 0; d < dsets.length; d++) {
                    if (accs[c1][d] > accs[c2][d]) {
                        wins++;
                        wdlList.get(count).get(0).add(dsets[d]);
                        wdlPlusMinus[count][d] = "1";
                    }
                    else if ((accs[c1][d] == accs[c2][d])) {
                        draws++;
                        wdlList.get(count).get(1).add(dsets[d]);
                        wdlPlusMinus[count][d] = "0";
                    }
                    else { 
                        losses++;
                        wdlList.get(count).get(2).add(dsets[d]);
                        wdlPlusMinus[count][d] = "-1";
                    }
                }
                table.append(","+wins+"|"+draws+"|"+losses);
                count++;
            }
            table.append("\n");
        }
        
        StringBuilder list = new StringBuilder();
        for (int i = 0; i < wdlListNames.size(); ++i) {
            list.append(wdlListNames.get(i));
            list.append("\n");
            list.append("Wins("+wdlList.get(i).get(0).size()+"):");
            for (String dset : wdlList.get(i).get(0)) 
                list.append(",").append(dset);
            list.append("\n");
            list.append("Draws("+wdlList.get(i).get(1).size()+"):");
            for (String dset : wdlList.get(i).get(1)) 
                list.append(",").append(dset);
            list.append("\n");
            list.append("Losses("+wdlList.get(i).get(2).size()+"):");
            for (String dset : wdlList.get(i).get(2)) 
                list.append(",").append(dset);
            list.append("\n\n");
        }
        
        StringBuilder plusMinuses = new StringBuilder();
        for (int j = 0; j < wdlPlusMinus.length; j++) 
            plusMinuses.append(",").append(wdlListNames.get(j));
        
        for (int i = 0; i < dsets.length; i++) {
            plusMinuses.append("\n").append(dsets[i]);
            for (int j = 0; j < wdlPlusMinus.length; j++) 
                plusMinuses.append(",").append(wdlPlusMinus[j][i]);
        }
        
        return new String[] { table.toString(), list.toString(), plusMinuses.toString() };
    }
    
    protected static String[] sigWinsDrawsLosses(double pval, double[][] accs, double[][][] foldAccs, String[] cnames, String[] dsets) {
        StringBuilder table = new StringBuilder();
        ArrayList<ArrayList<ArrayList<String>>> wdlList = new ArrayList<>(); //[classifierPairing][win/draw/loss][dsetNames]
        ArrayList<String> wdlListNames = new ArrayList<>();
        
        String[][] wdlPlusMinus = new String[cnames.length*cnames.length][dsets.length];
        
        table.append("p=" + pval + header(cnames)).append("\n");
        
        int count = 0;
        for (int c1 = 0; c1 < foldAccs.length; c1++) {
            table.append(cnames[c1]);
            for (int c2 = 0; c2 < foldAccs.length; c2++) {
                wdlListNames.add(cnames[c1] + "_VS_" + cnames[c2]);
                wdlList.add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());    
                
                int wins=0, draws=0, losses=0;
                for (int d = 0; d < dsets.length; d++) {
                    if (accs[c1][d] == accs[c2][d]) {
                        //when the accuracies are identical, p == NaN. 
                        //because NaN < 0.05 apparently it wont be counted as a draw, but a loss
                        //so handle it here                        
                        draws++;
                        wdlList.get(count).get(1).add(dsets[d]);
                        wdlPlusMinus[count][d] = "0";
                        continue;
                    }
                    
                    double p = TwoSampleTests.studentT_PValue(foldAccs[c1][d], foldAccs[c2][d]);
                    
                    if (p > pval) {
                        draws++;
                        wdlList.get(count).get(1).add(dsets[d]);
                        wdlPlusMinus[count][d] = "0";
                    }
                    else { //is sig
                        if (accs[c1][d] > accs[c2][d]) {
                            wins++;
                            wdlList.get(count).get(0).add(dsets[d]);
                            wdlPlusMinus[count][d] = "1";
                        }
                        else  {
                            losses++;
                            wdlList.get(count).get(2).add(dsets[d]);
                            wdlPlusMinus[count][d] = "-1";
                        }
                    }
                }
                table.append(","+wins+"|"+draws+"|"+losses);
                count++;
            }
            table.append("\n");
        }
        
        StringBuilder list = new StringBuilder();
        for (int i = 0; i < wdlListNames.size(); ++i) {
            list.append(wdlListNames.get(i));
            list.append("\n");
            list.append("Wins("+wdlList.get(i).get(0).size()+"):");
            for (String dset : wdlList.get(i).get(0)) 
                list.append(",").append(dset);
            list.append("\n");
            list.append("Draws("+wdlList.get(i).get(1).size()+"):");
            for (String dset : wdlList.get(i).get(1)) 
                list.append(",").append(dset);
            list.append("\n");
            list.append("Losses("+wdlList.get(i).get(2).size()+"):");
            for (String dset : wdlList.get(i).get(2)) 
                list.append(",").append(dset);
            list.append("\n\n");
        }
        
        StringBuilder plusMinuses = new StringBuilder();
        for (int j = 0; j < wdlPlusMinus.length; j++) 
            plusMinuses.append(",").append(wdlListNames.get(j));
        
        for (int i = 0; i < dsets.length; i++) {
            plusMinuses.append("\n").append(dsets[i]);
            for (int j = 0; j < wdlPlusMinus.length; j++) 
                plusMinuses.append(",").append(wdlPlusMinus[j][i]);
        }
        
        return new String[] { table.toString(), list.toString(), plusMinuses.toString() };
    }
    
    protected static double[][][] getAccs(ArrayList<ClassifierEvaluation> res, String trainortest) {
        double[][][] accs = new double[res.size()][res.get(0).trainResults.length][res.get(0).trainResults[0].length];
        for (int i = 0; i < res.size(); i++)
            if (trainortest.equals("train")) {
                for (int j = 0; j < res.get(i).trainResults.length; j++) {
                    for (int k = 0; k < res.get(i).trainResults[j].length; k++) {
                        accs[i][j][k] = res.get(i).trainResults[j][k].acc;
                    }
                }
            }   
            else {
                for (int j = 0; j < res.get(i).testResults.length; j++) {
                    for (int k = 0; k < res.get(i).testResults[j].length; k++) {
                        accs[i][j][k] = res.get(i).testResults[j][k].acc;
                    }
                }
            }
        return accs;
    }
    
    protected static double[][][] getBalAccs(ArrayList<ClassifierEvaluation> res, String trainortest) {
        double[][][] balaccs = new double[res.size()][res.get(0).trainResults.length][res.get(0).trainResults[0].length];
        for (int i = 0; i < res.size(); i++)
            if (trainortest.equals("train")) {
                for (int j = 0; j < res.get(i).trainResults.length; j++) {
                    for (int k = 0; k < res.get(i).trainResults[j].length; k++) {
                        balaccs[i][j][k] = res.get(i).trainResults[j][k].balancedAcc;
                    }
                }
            }   
            else {
                for (int j = 0; j < res.get(i).testResults.length; j++) {
                    for (int k = 0; k < res.get(i).testResults[j].length; k++) {
                        balaccs[i][j][k] = res.get(i).testResults[j][k].balancedAcc;
                    }
                }
            }
        return balaccs;
    }
    
    protected static double[][][] getAUROCs(ArrayList<ClassifierEvaluation> res, String trainortest) {
        double[][][] aurocs = new double[res.size()][res.get(0).trainResults.length][res.get(0).trainResults[0].length];
        for (int i = 0; i < res.size(); i++)
            if (trainortest.equals("train")) {
                for (int j = 0; j < res.get(i).trainResults.length; j++) {
                    for (int k = 0; k < res.get(i).trainResults[j].length; k++) {
                        aurocs[i][j][k] = res.get(i).trainResults[j][k].meanAUROC;
                    }
                }
            }   
            else {
                for (int j = 0; j < res.get(i).testResults.length; j++) {
                    for (int k = 0; k < res.get(i).testResults[j].length; k++) {
                        aurocs[i][j][k] = res.get(i).testResults[j][k].meanAUROC;
                    }
                }
            }
        return aurocs;
    }
    
    protected static double[][][] getNLLs(ArrayList<ClassifierEvaluation> res, String trainortest) {
        double[][][] nlls = new double[res.size()][res.get(0).trainResults.length][res.get(0).trainResults[0].length];
        for (int i = 0; i < res.size(); i++)
            if (trainortest.equals("train")) {
                for (int j = 0; j < res.get(i).trainResults.length; j++) {
                    for (int k = 0; k < res.get(i).trainResults[j].length; k++) {
                        nlls[i][j][k] = res.get(i).trainResults[j][k].nll;
                    }
                }
            }   
            else {
                for (int j = 0; j < res.get(i).testResults.length; j++) {
                    for (int k = 0; k < res.get(i).testResults[j].length; k++) {
                        nlls[i][j][k] = res.get(i).testResults[j][k].nll;
                    }
                }
            }
        return nlls;
    }
    
    public static String[] getNames(ArrayList<ClassifierEvaluation> res) {
        String[] names = new String[res.size()];
        for (int i = 0; i < res.size(); i++)
            names[i] = res.get(i).classifierName;
        return names;
    }
    
    protected static void buildResultsSpreadsheet(String basePath, String expName) {        
        WritableWorkbook wb = null;
        WorkbookSettings wbs = new WorkbookSettings();
        wbs.setLocale(new Locale("en", "EN"));
        
        try {
            wb = Workbook.createWorkbook(new File(basePath + expName + "ResultsSheet.xls"), wbs);        
        } catch (Exception e) { 
            System.out.println("ERROR CREATING RESULTS SPREADSHEET");
            System.out.println(e);
            System.exit(0);
        }
        
        WritableSheet summarySheet = wb.createSheet("GlobalSummary", 0);
        String summaryCSV = basePath + expName + "_SMALLglobalSummary.csv";
        copyCSVIntoSheet(summarySheet, summaryCSV);
        
        String filenameprefix = basePath + "Accuracy/" + expName;
        buildStatSheets(wb, filenameprefix, "ACC", 0);
        
        filenameprefix = basePath + "BalancedAccuracy/" + expName;
        buildStatSheets(wb, filenameprefix, "BALACC", 1);
        
        filenameprefix = basePath + "AUROC/" + expName;
        buildStatSheets(wb, filenameprefix, "AUROC", 2);
        
        filenameprefix = basePath + "NLL/" + expName;
        buildStatSheets(wb, filenameprefix, "NLL", 3);
        
        try {
            wb.write();
            wb.close();      
        } catch (Exception e) { 
            System.out.println("ERROR WRITING AND CLOSING RESULTS SPREADSHEET");
            System.out.println(e);
            System.exit(0);
        }
    }
    
    protected static void buildStatSheets(WritableWorkbook wb, String filenameprefix, String statName, int statInd) {
        int numSubStats = 5;        
        
        WritableSheet accTrainSheet = wb.createSheet(statName+"Train", 1+statInd*numSubStats+0);
        String accTrainCSV = filenameprefix + "_TRAIN"+statName+"S.csv";
        copyCSVIntoSheet(accTrainSheet, accTrainCSV);
        
        WritableSheet accTestSheet = wb.createSheet(statName+"Test", 1+statInd*numSubStats+1);
        String accTestCSV = filenameprefix + "_TEST"+statName+"S.csv";
        copyCSVIntoSheet(accTestSheet, accTestCSV);
        
        WritableSheet accRankSheet = wb.createSheet(statName+"TestRanks", 1+statInd*numSubStats+2);
        String accRankCSV = filenameprefix + "_"+statName+"RANKS.csv";
        copyCSVIntoSheet(accRankSheet, accRankCSV);
        
        WritableSheet accTrainTestDiffSheet = wb.createSheet(statName+"TrainTestDiffs", 1+statInd*numSubStats+3);
        String accTrainTestDiffCSV = filenameprefix + "_TRAINTEST"+statName+"DIFFS.csv";
        copyCSVIntoSheet(accTrainTestDiffSheet, accTrainTestDiffCSV);
        
        WritableSheet accSummarySheet = wb.createSheet(statName+"SigDiffs", 1+statInd*numSubStats+4);
        String accSummaryCSV = filenameprefix + "_TEST"+statName+"S_SUMMARY.csv";
        copyCSVIntoSheet(accSummarySheet, accSummaryCSV);
    }
    
    protected static void copyCSVIntoSheet(WritableSheet sheet, String csvFile) {
        try { 
            Scanner fileIn = new Scanner(new File(csvFile));

            int rowInd = 0;
            while (fileIn.hasNextLine()) {
                Scanner lineIn = new Scanner(fileIn.nextLine());
                lineIn.useDelimiter(",");

                int colInd = -1;
                while (lineIn.hasNext()) {
                    colInd++; //may not reach end of block, so incing first and initialising at -1
                    
                    String cellContents = lineIn.next();
                    WritableFont font = new WritableFont(WritableFont.ARIAL, 10); 	
                    WritableCellFormat format = new WritableCellFormat(font);
                    
                    try {
                        int iCellContents = Integer.parseInt(cellContents);
                        sheet.addCell(new jxl.write.Number(colInd, rowInd, iCellContents, format));
                        continue; //if successful, val was int, has been written, move on
                    } catch (NumberFormatException nfm) { }
                        
                    try {
                        double dCellContents = Double.parseDouble(cellContents);
                        sheet.addCell(new jxl.write.Number(colInd, rowInd, dCellContents, format));
                        continue; //if successful, val was int, has been written, move on
                    } catch (NumberFormatException nfm) { }
                    
                    
                    sheet.addCell(new jxl.write.Label(colInd, rowInd, cellContents, format));
                }
                rowInd++;
            }
        } catch (Exception e) {
            System.out.println("ERROR BUILDING RESULTS SPREADSHEET, COPYING CSV");
            System.out.println(e);
            System.exit(0);
        }
    }
    
}
