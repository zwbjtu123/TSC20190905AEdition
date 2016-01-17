package tsc_algorithms;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import java.util.HashMap;
import java.util.Map.Entry;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;

/**
 * BOSS classifier to be used with known parameters, for boss with parameter search, use BOSSEnsemble.
 * 
 * Params: wordLength, alphabetSize, windowLength, normalise?
 * 
 * @author James
 */
public class BOSS implements Classifier {
    
    private String [][] SFAwords; //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
    public ArrayList<Bag> bags; //histograms of words of the current wordlength with numerosity reduction applied (if selected)
    private double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;
    
    private double inverseSqrtWindowSize;
    private int windowSize;
    private int wordLength;
    private int alphabetSize;
    private boolean norm;
    
    private boolean numerosityReduction = true; 
    
    private static String[] alphabetSymbols = { "a","b","c","d","e","f","g","h","i","j" };
    private String[] alphabet = null;
    
    private static final long serialVersionUID = 1L;

    public BOSS(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        
        generateAlphabet();
    }
    
    /**
     * Used when shortening histograms, copies 'meta' data over, but with shorter 
     * word length, actual shortening happens separately
     */
    private BOSS(BOSS boss, int wordLength) {
        this.wordLength = wordLength;
        
        this.windowSize = boss.windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.numerosityReduction = boss.numerosityReduction; 
        this.alphabet = boss.alphabet;
        
        this.SFAwords = boss.SFAwords;
        this.breakpoints = boss.breakpoints;
        
        bags = new ArrayList<>(boss.bags.size());
    }
    
    public static class Bag extends HashMap<String, Integer> {
        double classVal;
        
        public Bag() {
            super();
        }
        
        public Bag(int classValue) {
            super();
            classVal = classValue;
        }

        public double getClassVal() { return classVal; }
        public void setClassVal(double classVal) { this.classVal = classVal; }       
    }

    public int getWindowSize() {
        return windowSize;
    }

    public int getWordLength() {
        return wordLength;
    }

    public int getAlphabetSize() {
        return alphabetSize;
    }

    public boolean isNorm() {
        return norm;
    }
    
    /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? } 
     */
    public int[] getParameters() {
        return new int[] { wordLength, alphabetSize, windowSize };
    }
    
    public void generateAlphabet() {
        alphabet = new String[alphabetSize];
        for (int i = 0; i < alphabetSize; ++i)
            alphabet[i] = alphabetSymbols[i];
    }
    
    private double[][] slidingWindow(double[] data) {
        int numWindows = data.length-windowSize+1;
        double[][] subSequences = new double[numWindows][windowSize];
        
        for (int windowStart = 0; windowStart < numWindows; ++windowStart) { 
            //copy the elements windowStart to windowStart+windowSize from data into 
            //the subsequence matrix at row windowStart
            System.arraycopy(data,windowStart,subSequences[windowStart],0,windowSize);
        }
        
        return subSequences;
    }
    
//        /**
//     * Gets sliding windows from data (a timeseries) and the std deviation of each window, stored 
//     * in windows and stdDevs respectively
//     * 
//     * @param data original timeseries [FINAL]
//     * @param windows sliding windows will be stored here
//     * @param stdDevs stddev of each window will be stored here
//     */
//    private void slidingWindow(final double[] data, double[][] windows, double[] stdDevs) {
//        int numWindows = data.length-windowSize+1;
//        windows = new double[numWindows][windowSize];
//        stdDevs = new double[numWindows];
//        
//        double inverseWindowSize = 1.0 / windowSize;
//        
//        for (int windowStart = 0; windowStart < numWindows; ++windowStart) { 
//            //copy the elements windowStart to windowStart+windowSize from data into 
//            //the subsequence matrix at position win
//            //and calculate the stddev of this window
//            double sum = 0.0;
//            double squareSum = 0.0;
//            for (int i = 0; i < windowSize; i++) {
//                sum += data[windowStart+i];
//                squareSum += data[windowStart+i]*data[windowStart+i];
//                windows[windowStart][i] = data[windowStart+i];
//            }
//            
//            double mean = sum * inverseWindowSize;
//            double variance = squareSum * inverseWindowSize - mean*mean;
//            stdDevs[windowStart] = Math.sqrt(variance);
//        }
//    }
    
    private double[][] performDFT(double[][] windows) {
        double[][] dfts = new double[windows.length][wordLength];
        for (int i = 0; i < windows.length; ++i) {
            dfts[i] = DFT(windows[i]);
        }
        return dfts;
    }
    
    private double stdDev(double[] series) {
        double sum = 0.0;
        double squareSum = 0.0;
        for (int i = 0; i < windowSize; i++) {
            sum += series[i];
            squareSum += series[i]*series[i];
        }

        double mean = sum / series.length;
        double variance = squareSum / series.length - mean*mean;
        return variance > 0 ? Math.sqrt(variance) : 1.0;
    }
    
    /**
     * Performs DFT but calculates only wordLength/2 coefficients instead of the 
     * full transform, and skips the first coefficient if it is to be normalised
     * 
     * @return double[] size wordLength, { real1, imag1, ... realwl/2, imagwl/2 }
     */
    private double[] DFT(double[] series) {
        //taken from FFT.java but 
        //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
        //instead of Complex[] size n/2
        
        //also, only calculating first wordlength/2 coefficients (output values) instead of 
        //entire transform, as it will be low pass filtered anyway, and skipping first coefficient
        //if the data is to be normalised
        int n=series.length;
        int outputLength = wordLength/2;
        int start = (norm ? 1 : 0);
        
        //normalize the disjoint windows and sliding windows by dividing them by their standard deviation 
        //all Fourier coefficients are divided by sqrt(windowSize)

        double normalisingFactor = inverseSqrtWindowSize / stdDev(series);
        
        double[] dft=new double[outputLength*2];
        
        for (int k = start; k < start + outputLength; k++) {  // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) {  // For each input element
                sumreal +=  series[t]*Math.cos(2*Math.PI * t * k / n);
                sumimag += -series[t]*Math.sin(2*Math.PI * t * k / n);
            }
            dft[(k-start)*2]   = sumreal * normalisingFactor;
            dft[(k-start)*2+1] = sumimag * normalisingFactor;
        }
        return dft;
    }
    
    private double[][] disjointWindows(double [] data) {
        int amount = (int)Math.ceil(data.length/(double)windowSize);
        double[][] subSequences = new double[amount][windowSize];
        
        for (int win = 0; win < amount; ++win) { 
            int offset = Math.min(win*windowSize, data.length-windowSize);
            
            //copy the elements windowStart to windowStart+windowSize from data into 
            //the subsequence matrix at position windowStart
            System.arraycopy(data,offset,subSequences[win],0,windowSize);
        }
        
        return subSequences;
    }
    
//    /**
//     * Gets disjoint windows from data (a timeseries) and the std deviation of each window, stored 
//     * in windows and stdDevs respectively
//     * 
//     * @param data original timeseries [FINAL]
//     * @param windows disjoint windows will be stored here
//     * @param stdDevs stddev of each window will be stored here
//     */
//    private void disjointWindows(final double[] data, double[][] windows, double[] stdDevs) {
//        int amount = (int)Math.ceil(data.length/(double)windowSize);
//        double[][] subSequences = new double[amount][windowSize];
//        windows = new double[amount][windowSize];
//        stdDevs = new double[amount];
//        
//        double inverseWindowSize = 1.0 / windowSize;
//        
//        for (int win = 0; win < amount; ++win) { 
//            int offset = Math.min(win*windowSize, data.length-windowSize);
//            
//            //copy the elements windowStart to windowStart+windowSize from data into 
//            //the subsequence matrix at position win
//            //and calculate the stddev of this window
//            double sum = 0.0;
//            double squareSum = 0.0;
//            for (int i = 0; i < windowSize; i++) {
//                sum += data[offset+i];
//                squareSum += data[offset+i]*data[offset+i];
//                windows[win][i] = data[offset+i];
//            }
//            
//            double mean = sum * inverseWindowSize;
//            double variance = squareSum * inverseWindowSize - mean*mean;
//            stdDevs[win] = Math.sqrt(variance);
//        }
//        
//    }
    
    private double[][] MCB(Instances data) {
        double[][][] dfts = new double[data.numInstances()][][];
        
        int sample = 0;
        for (Instance inst : data) {
            dfts[sample++] = performDFT(disjointWindows(toArrayNoClass(inst))); //approximation
        }
        
        int numInsts = dfts.length;
        int numWindowsPerInst = dfts[0].length;
        int totalNumWindows = numInsts*numWindowsPerInst;

        breakpoints = new double[wordLength][alphabetSize]; 
        
        for (int letter = 0; letter < wordLength; ++letter) { //for each dft coeff
            
            //extract this column from all windows in all instances
            double[] column = new double[totalNumWindows];
            for (int inst = 0; inst < numInsts; ++inst)
                for (int window = 0; window < numWindowsPerInst; ++window) {
                    //rounding dft coefficients to reduce noise
                    column[(inst * numWindowsPerInst) + window] = Math.round(dfts[inst][window][letter]);   
                }
            
            //sort, and run through to find breakpoints for equi-depth bins
            Arrays.sort(column);
            
            double binIndex = 0;
            double targetBinDepth = (double)totalNumWindows / (double)alphabetSize; 
            
            for (int bp = 0; bp < alphabetSize-1; ++bp) {
                binIndex += targetBinDepth;
                breakpoints[letter][bp] = column[(int)binIndex];
            }

            breakpoints[letter][alphabetSize-1] = Double.MAX_VALUE; //last one can always = infinity
        }
    
        return breakpoints;
    }
    
    /**
     * Builds a brand new boss bag from the passed fourier transformed data, rather than from
     * looking up existing transforms from earlier builds. 
     * 
     * to be used e.g to transform new test instances
     */
    private Bag createBagSingle(double[][] dfts) {
        Bag bag = new Bag();
        String lastWord = "";
        
        for (double[] d : dfts) {
            String word = createWord(d);
            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord))
                continue;

            Integer val = bag.get(word);
            if (val == null)
                val = 0;
            bag.put(word, ++val);   
            
            lastWord = word;
        }
        
        return bag;
    }
    
    private String createWord(double[] dft) {
        String word = "";
        for (int l = 0; l < wordLength; ++l) {//for each letter
            for (int bp = 0; bp < alphabetSize; ++bp) {//run through breakpoints until right one found
                if (dft[l] <= breakpoints[l][bp]) {
                    word += alphabet[bp]; //add corresponding letter to word
                    break;
                }
            }
        }

        return word;
    }
    
    /**
     * Assumes class index, if present, is last
     * @return data of passed instance in a double array with the class value removed if present
     */
    private static double[] toArrayNoClass(Instance inst) {
        int length = inst.numAttributes();
        if (inst.classIndex() >= 0)
            --length;
        
        double[] data = new double[length];
        
        for (int i=0, j=0; i < inst.numAttributes(); ++i)
            if (inst.classIndex() != i)
                data[j++] = inst.value(i);
        
        return data;
    }
    
    /**
     * @return BOSSTransform-ed bag, built using current parameters
     */
    public Bag BOSSTransform(Instance inst) {
        double[][] dfts = performDFT(slidingWindow(toArrayNoClass(inst))); //approximation     
        Bag bag = createBagSingle(dfts); //discretisation/bagging
        bag.setClassVal(inst.classValue());
        return bag;
    }
    
//    /**
//     * Creates and returns new boss instance with shortened wordLength and corresponding
//     * histograms, the boss instance passed in is UNCHANGED, if wordLengths are same, does nothing,
//     * just returns passed in boss instance
//     * 
//     * @param newWordLength wordLength to shorten it to
//     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
//     */
//    public static BOSS shortenHistograms(int newWordLength, final BOSS oldBoss) throws Exception {
//        if (newWordLength == oldBoss.wordLength) //case of first iteration of word length search in ensemble
//            return oldBoss;
//        if (newWordLength > oldBoss.wordLength)
//            throw new Exception("Cannot incrementally INCREASE word length, current:"+oldBoss.wordLength+", requested:"+newWordLength);
//        if (newWordLength < 2)
//            throw new Exception("Invalid wordlength requested, current:"+oldBoss.wordLength+", requested:"+newWordLength);
//       
//        //copies/updates meta data
//        BOSS newBoss = new BOSS(oldBoss, newWordLength); 
//        
//        //shorten/copy actual histograms
//        for (Bag bag : oldBoss.bags)
//            newBoss.bags.add(shortenHistogram(newWordLength, bag));
//            
//        return newBoss;
//    }
//    
//    private static Bag shortenHistogram(int newWordLength, Bag oldBag) {
//        Bag newBag = new Bag();
//        
//        for (Entry<String, Integer> origWord : oldBag.entrySet()) {
//            String shortWord = origWord.getKey().substring(0, newWordLength);
//            
//            Integer val = newBag.get(shortWord);
//            if (val == null)
//                val = 0;
//                
//            newBag.put(shortWord, val + origWord.getValue());
//        }
//        
//        newBag.setClassVal(oldBag.getClassVal());
//        
//        return newBag;
//    }
    
        /**
     * Shortens all bags in this BOSS instance (histograms) to the newWordLength, if wordlengths
     * are same, instance is UNCHANGED
     * 
     * @param newWordLength wordLength to shorten it to
     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
     */
    public BOSS buildShortenedBags(int newWordLength) throws Exception {
        if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
            return this;
        if (newWordLength > wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);
       
        BOSS newBoss = new BOSS(this, newWordLength);
        
        //build hists with new word length from SFA words, and copy over the class values of original insts
        for (int i = 0; i < bags.size(); ++i) {
            Bag newBag = createBagFromWords(newWordLength, SFAwords[i]);   
            newBag.setClassVal(bags.get(i).getClassVal());
            newBoss.bags.add(newBag);
        }
        
        return newBoss;
    }
    
    private Bag shortenBag(int newWordLength, int bagIndex) {
        Bag newBag = new Bag();
        
        for (String word : SFAwords[bagIndex]) {
            String shortWord = word.substring(0, newWordLength);
            
            Integer val = newBag.get(shortWord);
            if (val == null)
                val = 0;
                
            newBag.put(shortWord, val + 1);
        }
        
        return newBag;
    }
    
    /**
     * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
     */
    private Bag createBagFromWords(int thisWordLength, String[] words) {
        Bag bag = new Bag();
        String lastWord = "";
        
        for (String word : words) {
            if (wordLength != thisWordLength)
                word = word.substring(0, thisWordLength);
            
            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord))
                continue;

            Integer val = bag.get(word);
            if (val == null)
                val = 0;
            bag.put(word, ++val);   
            
            lastWord = word;
        }
        
        return bag;
    }
    
    private String[] createSFAwords(Instance inst) throws Exception {
        double[][] dfts = performDFT(slidingWindow(toArrayNoClass(inst))); //approximation     
        String[] words = new String[dfts.length];
        for (int window = 0; window < dfts.length; ++window) 
            words[window] = createWord(dfts[window]);//discretisation
            
        return words;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data
 
        SFAwords = new String[data.numInstances()][];
        bags = new ArrayList<>(data.numInstances());
        
        for (int inst = 0; inst < data.numInstances(); ++inst) {
            SFAwords[inst] = createSFAwords(data.get(inst));
            
            Bag bag = createBagFromWords(wordLength, SFAwords[inst]);
            bag.setClassVal(data.get(inst).classValue());
            bags.add(bag);
        }
    }

    /**
     * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a)
     * @return distance FROM instA TO instB
     */
    public double BOSSdistance(Bag instA, Bag instB) {
        double dist = 0.0;
        
        //find dist only from values in instA
        for (Entry<String, Integer> entry : instA.entrySet()) {
            Integer valA = entry.getValue();
            Integer valB = instB.get(entry.getKey());
            if (valB == null)
                valB = 0;
            dist += (valA-valB)*(valA-valB);
        }
        
        return dist;
    }
    
       /**
     * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a).
     * 
     * Quits early if the dist-so-far is greater than bestDist (assumed is in fact the dist still squared), and returns Double.MAX_VALUE
     * 
     * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
     */
    public double BOSSdistance(Bag instA, Bag instB, double bestDist) {
        double dist = 0.0;
        
        //find dist only from values in instA
        for (Entry<String, Integer> entry : instA.entrySet()) {
            Integer valA = entry.getValue();
            Integer valB = instB.get(entry.getKey());
            if (valB == null)
                valB = 0;
            dist += (valA-valB)*(valA-valB);
            
            if (dist > bestDist)
                return Double.MAX_VALUE;
        }
        
        return dist;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Bag testBag = BOSSTransform(instance);
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

        //find dist FROM testBag TO all trainBags
        for (int i = 0; i < bags.size(); ++i) {
            double dist = BOSSdistance(testBag, bags.get(i), bestDist); 
            
            if (dist < bestDist) {
                bestDist = dist;
                nn = bags.get(i).getClassVal();
            }
        }
        
        return nn;
    }
    
    /**
     * Used within BOSSEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild 
     * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies 
     * the instance at the index passed while ignoring its own corresponding histogram 
     * 
     * @param test index of instance to classify
     * @return classification
     */
    public double classifyInstance(int test) {
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

        Bag testBag = bags.get(test);
        
        for (int i = 0; i < bags.size(); ++i) {
            if (i == test) //skip 'this' one, leave-one-out
                continue;
            
            double dist = BOSSdistance(testBag, bags.get(i), bestDist); 
            
            if (dist < bestDist) {
                bestDist = dist;
                nn = bags.get(i).getClassVal();
            }
        }
        
        return nn;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        basicTest();
        System.out.println("\n\n\n\n");
        tonyTest();
    }
    
     public static void basicTest() {
        System.out.println("BOSSBasicTest\n\n");
        try {
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST.arff");
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");

            System.out.println(train.relationName());
            
            BOSS boss = new BOSS(8,4,16,true);
            System.out.println(boss.getWordLength() + " " + boss.getAlphabetSize() + " " + boss.getWindowSize() + " " + boss.isNorm());
            
            System.out.println("Training starting");
            long start = System.nanoTime();
            boss.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");

            System.out.println("Breakpoints: ");
            for (int i = 0; i < boss.breakpoints.length; i++) {
                System.out.print("Letter "  + i + ": ");
                for (int j = 0; j < boss.breakpoints[i].length; j++) {
                    System.out.print(boss.breakpoints[i][j] + " ");
                }
                System.out.println("");
            }
            
            System.out.println("\nTesting starting");
            start = System.nanoTime();
            double acc = ClassifierTools.accuracy(test, boss);
            double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Testing done (" + testTime + "s)");
            
            System.out.println("\nACC: " + acc);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
     
     public static void tonyTest() {
        System.out.println("BOSS Sanity Checks\n");
        DecimalFormat df = new DecimalFormat("##.####");
        int[] p={8,10,12,14,16};
        try {
            String pr="ItalyPowerDemand";
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\"+pr+"\\"+pr+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\"+pr+"\\"+pr+"_TEST.arff");
            System.out.println("Problem ="+pr+" has "+(train.numAttributes()-1)+" atts");
            double maxAcc=0;
            int bestP=0;
            int bestW=0;
            for(int k:p){
                for(int w=10;w<train.numAttributes()-1;w+=1){
                    BOSS b=new BOSS(k,4,w,false);
                    double a=ClassifierTools.stratifiedCrossValidation(train, b, 10, w);
                    if(a>maxAcc){
                        maxAcc=a;
                        bestP=k;
                        bestW=w;
                        System.out.println("Current best train p="+k+" w ="+w+" acc = "+a);
                    }
                }
            }
            BOSS b=new BOSS(bestP,4,bestW,false);
            System.out.println("BEST p="+bestP+" w = "+bestW+" acc ="+ClassifierTools.singleTrainTestSplitAccuracy(b, train, test));
               
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
     }
}
