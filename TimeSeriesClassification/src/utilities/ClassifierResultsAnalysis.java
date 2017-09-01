
package utilities;

import ResultsProcessing.MatlabController;
import development.MultipleClassifiersPairwiseTest;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.function.Function;
import jxl.Workbook;
import jxl.WorkbookSettings;
import jxl.write.WritableCellFormat;
import jxl.write.WritableFont;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import statistics.tests.OneSampleTests;
import statistics.tests.TwoSampleTests;
import utilities.ClassifierResults;
import utilities.StatisticalUtilities;
import utilities.generic_storage.Pair;

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
    
    protected static String matlabFilePath = "matlabfiles/";
    public static double FRIEDMANCDDIA_PVAL = 0.05;
        
    public static boolean buildMatlabDiagrams = false;
    public static boolean testResultsOnly = false;
    
    protected static final Function<ClassifierResults, Double> getAccs = (ClassifierResults cr) -> {return cr.acc;};
    protected static final Function<ClassifierResults, Double> getBalAccs = (ClassifierResults cr) -> {return cr.balancedAcc;};
    protected static final Function<ClassifierResults, Double> getAUROCs = (ClassifierResults cr) -> {return cr.meanAUROC;};
    protected static final Function<ClassifierResults, Double> getNLLs = (ClassifierResults cr) -> {return cr.nll;};
    protected static final Function<ClassifierResults, Double> getF1s = (ClassifierResults cr) -> {return cr.f1;};
    protected static final Function<ClassifierResults, Double> getPrecisions = (ClassifierResults cr) -> {return cr.precision;};
    protected static final Function<ClassifierResults, Double> getRecalls = (ClassifierResults cr) -> {return cr.recall;};
    protected static final Function<ClassifierResults, Double> getSensitivities = (ClassifierResults cr) -> {return cr.sensitivity;};
    protected static final Function<ClassifierResults, Double> getSpecificities = (ClassifierResults cr) -> {return cr.specificity;};

    private static final String testLabel = "TEST";
    private static final String trainLabel = "TRAIN";
    private static final String trainTestDiffLabel = "TRAINTESTDIFFS";
    
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
        
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getDefaultStatistics() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        stats.add(new Pair<>("BALACC", getBalAccs));
        stats.add(new Pair<>("AUROC", getAUROCs));
        stats.add(new Pair<>("NLL", getNLLs));
        return stats;
    }
    
        
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getAllStatistics() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        stats.add(new Pair<>("BALACC", getBalAccs));
        stats.add(new Pair<>("AUROC", getAUROCs));
        stats.add(new Pair<>("NLL", getNLLs));
        stats.add(new Pair<>("F1", getF1s));
        stats.add(new Pair<>("Prec", getPrecisions));
        stats.add(new Pair<>("Recall", getRecalls));
        stats.add(new Pair<>("Sens", getSensitivities));
        stats.add(new Pair<>("Spec", getSpecificities));
        return stats;
    }
    
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getAccuracyStatisticOnly() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        return stats;
    }
    
    protected static void writeTableFile(String filename, String tableName, double[][] accs, String[] cnames, String[] dsets) {
        OutFile out=new OutFile(filename);
        out.writeLine(tableName + ":" + tabulate(accs, cnames, dsets));
//        out.writeLine("\navg:" + mean(accs));
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
    protected static String[] writeStatisticSummaryFile(String outPath, String filename, String statName, double[][][] statPerFold, double[][] statPerDset, double[][] ranks, double[][] stddevsFoldAccs, String[] cnames, String[] dsets) {   
        StringBuilder suppressedSummaryStats = new StringBuilder();
        suppressedSummaryStats.append(header(cnames)).append("\n");
        suppressedSummaryStats.append("Avg"+statName+":").append(mean(statPerDset)).append("\n");
        suppressedSummaryStats.append("Avg"+statName+"_RANK:").append(mean(ranks)).append("\n");
        
        StringBuilder summaryStats = new StringBuilder();
        
        summaryStats.append(statName).append(header(cnames)).append("\n");
        summaryStats.append("avgOverDsets:").append(mean(statPerDset)).append("\n");
        summaryStats.append("stddevOverDsets:").append(stddev(statPerDset)).append("\n");
        summaryStats.append("avgStddevsOverFolds:").append(mean(stddevsFoldAccs)).append("\n");
        summaryStats.append("avgRankOverDsets:").append(mean(ranks)).append("\n");
        summaryStats.append("stddevsRankOverDsets:").append(stddev(ranks)).append("\n");

        String[] wdl = winsDrawsLosses(statPerDset, cnames, dsets);
        String[] sig01wdl = sigWinsDrawsLosses(0.01, statPerDset, statPerFold, cnames, dsets);
        String[] sig05wdl = sigWinsDrawsLosses(0.05, statPerDset, statPerFold, cnames, dsets);
        
        (new File(outPath+"/WinsDrawsLosses/")).mkdir();
        OutFile outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLFLAT_"+statName+".csv");
        outwdl.writeLine(wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLSig01_"+statName+".csv");
        outwdl.writeLine(sig01wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLSig05_"+statName+".csv");
        outwdl.writeLine(sig05wdl[1]);
        outwdl.closeFile();
        
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLFLAT_"+statName+".csv");
        outwdl.writeLine(wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLSig01_"+statName+".csv");
        outwdl.writeLine(sig01wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLSig05_"+statName+".csv");
        outwdl.writeLine(sig05wdl[2]);
        outwdl.closeFile();
        
        
        OutFile out=new OutFile(outPath+filename+"_"+statName+"_SUMMARY.csv");
        
        out.writeLine(summaryStats.toString());
        
        out.writeLine(wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig01wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig05wdl[0]);
        out.writeLine("\n");
        
        String cliques = "";
        try {
            //System.out.println(filename+"_"+statistic+".csv");
            out.writeLine(MultipleClassifiersPairwiseTest.runTests(outPath+filename+"_"+statName+".csv").toString());       
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
    
    protected static String[] writeStatisticOnSplitFiles(String outPath, String filename, String evalSet, String statName, double[][][] foldVals, String[] cnames, String[] dsets) {
        outPath += evalSet + "/";
        
        double[][] dsetVals = findAvgsOverFolds(foldVals);
        double[][] stddevsFoldVals = findStddevsOverFolds(foldVals);
        double[][] ranks = findRanks(dsetVals);
        
        //BEFORE ordering, write the individual folds files
        writePerFoldFiles(outPath+evalSet+"FOLD"+statName+"S/", foldVals, cnames, dsets, evalSet);
        
        int[] ordering = findOrdering(ranks);
        ranks = order(ranks, ordering);
        cnames = order(cnames, ordering);
        
        foldVals = order(foldVals, ordering);
        dsetVals = order(dsetVals, ordering);
        stddevsFoldVals = order(stddevsFoldVals, ordering);
        
        if (evalSet.equalsIgnoreCase("TEST")) {
            //qol for cd dia creation, make a copy of all the raw test stat files in a common folder, one for pairwise, one for freidman
            File f = new File(outPath);
            String cdFolder = f.getParentFile().getParent() + "/cddias/";
            (new File(cdFolder)).mkdirs();
            OutFile out = new OutFile(cdFolder+"readme.txt");
            out.writeLine("remember that nlls are auto-negated now for cd dia ordering\n");
            out.writeLine("and that basic notepad wont show the line breaks properly, view (cliques especially) in notepad++");
            out.closeFile();
            for (String subFolder : new String[] { "pairwise", "friedman" }) {
                (new File(cdFolder+subFolder+"/")).mkdirs();
                String cdName = cdFolder+subFolder+"/"+cdFileName(filename,statName)+".csv";
                //meta hack for qol, negate the nll (sigh...) for correct ordering on dia
                if (statName.contains("NLL")) {
                    double[][] negatedDsetVals = new double[dsetVals.length][dsetVals[0].length];
                    for (int i = 0; i < dsetVals.length; i++) {
                        for (int j = 0; j < dsetVals[i].length; j++) {
                            negatedDsetVals[i][j] = dsetVals[i][j] * -1;
                        }
                    }
                    writeTableFileRaw(cdName, negatedDsetVals, cnames);
                } else {
                    writeTableFileRaw(cdName, dsetVals, cnames);
                } 
            } //end qol
        }
        
        writeTableFile(outPath+filename+"_"+evalSet+statName+"RANKS.csv", evalSet+statName+"RANKS", ranks, cnames, dsets);
        writeTableFile(outPath+filename+"_"+evalSet+statName+".csv", evalSet+statName, dsetVals, cnames, dsets);
        writeTableFileRaw(outPath+filename+"_"+evalSet+statName+"RAW.csv", dsetVals, cnames); //for matlab stuff
        writeTableFile(outPath+filename+"_"+evalSet+statName+"STDDEVS.csv", evalSet+statName+"STDDEVS", stddevsFoldVals, cnames, dsets);
        return writeStatisticSummaryFile(outPath, filename, evalSet+statName, foldVals, dsetVals, ranks, stddevsFoldVals, cnames, dsets); 
    }
    
    /**
     * Essentially just a wrapper for what writeStatisticOnSplitFiles does, in the simple case that we just have a 3d array of test accs and want summaries for it
     * Mostly for legacy results not in the classifier results file format 
     */
    public static void summariseTestAccuracies(String outPath, String filename, double[][][] testFolds, String[] cnames, String[] dsets) {
        writeStatisticOnSplitFiles(outPath, filename, testLabel, "ACC", testFolds, cnames, dsets);
    }
    
    protected static String[] writeStatisticFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, Pair<String, Function<ClassifierResults, Double>> evalStatistic, String[] cnames, String[] dsets) {
        String statName = evalStatistic.var1;
        outPath += statName + "/";
        new File(outPath).mkdirs();        
        
        double[][][] testFolds = getInfo(results, evalStatistic.var2, testLabel);
        
        if (!testResultsOnly) {
            double[][][] trainFolds = getInfo(results, evalStatistic.var2, trainLabel);
            double[][][] trainTestDiffsFolds = findTrainTestDiffs(trainFolds, testFolds);
            writeStatisticOnSplitFiles(outPath, filename, trainLabel, statName, trainFolds, cnames, dsets); //TODO
            writeStatisticOnSplitFiles(outPath, filename, trainTestDiffLabel, statName, trainTestDiffsFolds, cnames, dsets);
        }
        
        return writeStatisticOnSplitFiles(outPath, filename, testLabel, statName, testFolds, cnames, dsets);
    }
    
    /**
     * for legacy code, will call the overloaded version with acc,balacc,nll,auroc as the default statisitics
     */
    public static void writeAllEvaluationFiles(String outPath, String expname, ArrayList<ClassifierEvaluation> results, String[] dsets) {  
        writeAllEvaluationFiles(outPath, expname, getDefaultStatistics(), results, dsets);
    }
    
    public static void writeAllEvaluationFiles(String outPath, String expname, ArrayList<Pair<String,Function<ClassifierResults,Double>>> statistics, ArrayList<ClassifierEvaluation> results, String[] dsets) {
        //hacky housekeeping
        MultipleClassifiersPairwiseTest.beQuiet = true;
        OneSampleTests.beQuiet = true;
        
        outPath += expname + "/";
        new File(outPath).mkdirs();
        
        OutFile bigSummary = new OutFile(outPath + expname + "_BIGglobalSummary.csv");
        OutFile smallSummary = new OutFile(outPath + expname + "_SMALLglobalSummary.csv");
        
        String[] cnames = getNames(results);
        String [] statCliques = new String[statistics.size()];
        String [] statNames = new String[statistics.size()];
        
        for (int i = 0; i < statistics.size(); ++i) {
            Pair<String, Function<ClassifierResults, Double>> stat = statistics.get(i);
            
            String[] summary = writeStatisticFiles(outPath, expname, results, stat, cnames, dsets);
            
            bigSummary.writeString(stat.var1+":");
            bigSummary.writeLine(summary[0]);
            
            smallSummary.writeString(stat.var1+":");
            smallSummary.writeLine(summary[1]);
            
            statNames[i] = stat.var1;
            statCliques[i] = summary[2];
        }
        
        bigSummary.closeFile();
        smallSummary.closeFile();
        
        buildResultsSpreadsheet(outPath, expname, statistics);
        
        //write these even if not actually making the dias this execution
        writeCliqueHelperFiles(outPath + "/cdDias/pairwise/", expname, statNames, statCliques); 
        if(buildMatlabDiagrams)
            buildCDDias(outPath, expname, statNames, statCliques);
    }
    
    protected static void writeCliqueHelperFiles(String cdCSVpath, String expname, String[] stats, String[] cliques) {
        //temp workaround, just write the cliques and readin again from matlab for ease of checking/editing for pairwise edge cases
        for (int i = 0; i < stats.length; i++) {
            OutFile out = new OutFile (cdCSVpath + cdFileName(expname, stats[i]) + "_cliques.txt");
            out.writeString(cliques[i]);
            out.closeFile();
        }
    }
    
    protected static void buildCDDias(String outpath, String expname, String[] stats, String[] cliques) {        
        MatlabController proxy = MatlabController.getInstance();
        proxy.eval("addpath(genpath('"+matlabFilePath+"'))");
        proxy.eval("buildDiasInDirectory('"+outpath+"/cdDias/friedman/"+"', 0, "+FRIEDMANCDDIA_PVAL+")"); //friedman 
        proxy.eval("buildDiasInDirectory('"+outpath+ "/cdDias/pairwise/"+"', 1)");  //pairwise
        proxy.discconnectMatlab();
    }
        
    protected static void writePerFoldFiles(String outPath, double[][][] folds, String[] cnames, String[] dsets, String splitLabel) {
        new File(outPath).mkdirs();
        
        StringBuilder headers = new StringBuilder("folds:");
        for (int f = 0; f < folds[0][0].length; f++)
            headers.append(","+f);
        
        for (int c = 0; c < folds.length; c++) {
            OutFile out=new OutFile(outPath + cnames[c]+"_"+splitLabel+"FOLDS.csv");
            out.writeLine(headers.toString());
            
            for (int d = 0; d < folds[c].length; d++) {
                out.writeString(dsets[d]);
                for (int f = 0; f < folds[c][d].length; f++)
                    out.writeString("," + folds[c][d][f]);
                out.writeLine("");
            }
            
            out.closeFile();
        }
        
        
        OutFile out = new OutFile(outPath + "TEXASPLOT_"+splitLabel+".csv");
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
    
    protected static String mean(double[][] res) {
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
    
    protected static int[] findOrdering(double[][] r) {
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
    
    protected static String[] order(String[] s, int[] ordering) {
        String[] res = new String[s.length];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    protected static double[][] order(double[][] s, int[] ordering) {
        double[][] res = new double[s.length][];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    protected static double[][][] order(double[][][] s, int[] ordering) {
        double[][][] res = new double[s.length][][];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    /**
     * @param accs [classifiers][acc on datasets]
     * @return [classifiers][rank on dataset]
     */
    protected static double[][] findRanks(double[][] accs) {
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
    
    protected static double[][][] getInfo(ArrayList<ClassifierEvaluation> res, Function<ClassifierResults, Double> getter, String trainortest) {
        double[][][] info = new double[res.size()][res.get(0).testResults.length][res.get(0).testResults[0].length];
        for (int i = 0; i < res.size(); i++) {
            if (trainortest.equalsIgnoreCase(trainLabel))
                for (int j = 0; j < res.get(i).trainResults.length; j++)
                    for (int k = 0; k < res.get(i).trainResults[j].length; k++)
                        info[i][j][k] = getter.apply(res.get(i).trainResults[j][k]);
            else if (trainortest.equalsIgnoreCase(testLabel))
                for (int j = 0; j < res.get(i).testResults.length; j++)
                    for (int k = 0; k < res.get(i).testResults[j].length; k++)
                        info[i][j][k] = getter.apply(res.get(i).testResults[j][k]);
            else {
                System.out.println("teh fook? getInfo(), trainortest="+trainortest);
                System.exit(0);
            }
        }
        return info;
    }
        
    protected static String[] getNames(ArrayList<ClassifierEvaluation> res) {
        String[] names = new String[res.size()];
        for (int i = 0; i < res.size(); i++)
            names[i] = res.get(i).classifierName;
        return names;
    }
    
    protected static void buildResultsSpreadsheet(String basePath, String expName, ArrayList<Pair<String,Function<ClassifierResults,Double>>> statistics) {        
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
        
        for (int i = 0; i < statistics.size(); i++) {
            String statName = statistics.get(i).var1;
            String path = basePath + statName + "/";
            buildStatSheets(wb, expName, path, statName, i);
        }
        
        try {
            wb.write();
            wb.close();      
        } catch (Exception e) { 
            System.out.println("ERROR WRITING AND CLOSING RESULTS SPREADSHEET");
            System.out.println(e);
            System.exit(0);
        }
    }
    
    protected static void buildStatSheets(WritableWorkbook wb, String expName, String filenameprefix, String statName, int statIndex) {
        final int initialSummarySheetOffset = 1;
        int numSubStats = 3;    
        int testOffset = 0;
        
        if (!testResultsOnly) {
            numSubStats = 5; 
            testOffset = 2;
            
            WritableSheet trainSheet = wb.createSheet(statName+"Train", initialSummarySheetOffset+statIndex*numSubStats+0);
            String trainCSV = filenameprefix + trainLabel + "/" + expName + "_" +trainLabel+statName+".csv";
            copyCSVIntoSheet(trainSheet, trainCSV);

            WritableSheet trainTestDiffSheet = wb.createSheet(statName+"TrainTestDiffs", initialSummarySheetOffset+statIndex*numSubStats+1);
            String trainTestDiffCSV = filenameprefix + trainTestDiffLabel + "/" + expName + "_" +trainTestDiffLabel+statName+".csv";
            copyCSVIntoSheet(trainTestDiffSheet, trainTestDiffCSV);
        }
        
        WritableSheet testSheet = wb.createSheet(statName+"Test", initialSummarySheetOffset+statIndex*numSubStats+0+testOffset);
        String testCSV = filenameprefix + testLabel + "/" + expName + "_" +testLabel+statName+".csv";
        copyCSVIntoSheet(testSheet, testCSV);
        
        WritableSheet rankSheet = wb.createSheet(statName+"TestRanks", initialSummarySheetOffset+statIndex*numSubStats+1+testOffset);
        String rankCSV = filenameprefix + testLabel + "/" + expName + "_" +testLabel+statName+"RANKS.csv";
        copyCSVIntoSheet(rankSheet, rankCSV);
        
        WritableSheet summarySheet = wb.createSheet(statName+"TestSigDiffs", initialSummarySheetOffset+statIndex*numSubStats+2+testOffset);
        String summaryCSV = filenameprefix + testLabel + "/" + expName + "_" +testLabel+statName+"_SUMMARY.csv";
        copyCSVIntoSheet(summarySheet, summaryCSV);
        
        
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
