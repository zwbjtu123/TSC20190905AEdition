package tsc_algorithms;

import development.DataSets;
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
    
    public ArrayList<Bag> bags;
    double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;
    
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
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.numerosityReduction = boss.numerosityReduction; 
        this.alphabet = boss.alphabet;
        
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
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize } 
     */
    public int[] getParameters() {
        return new int[] { wordLength, alphabetSize, windowSize};
    }
    
    public void generateAlphabet() {
        alphabet = new String[alphabetSize];
        for (int i = 0; i < alphabetSize; ++i)
            alphabet[i] = alphabetSymbols[i];
    }
    
    private double[][] slidingWindow(double[] data) {
        double[][] subSequences = new double[data.length-windowSize+1][windowSize];
        
        for (int windowStart = 0; windowStart+windowSize-1 < data.length; ++windowStart) { 
            //copy the elements windowStart to windowStart+windowSize from data into 
            //the subsequence matrix at row windowStart
            System.arraycopy(data,windowStart,subSequences[windowStart],0,windowSize);
        }
        
        return subSequences;
    }
    
    private double[][] performDFT(double[][] windows) {
        double[][] dfts = new double[windows.length][wordLength];
        for (int i = 0; i < windows.length; ++i) {
            dfts[i] = DFT(windows[i]);
        }
        return dfts;
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
        
        double[] dft=new double[outputLength*2];
        
        for (int k = start; k < start + outputLength; k++) {  // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) {  // For each input element
                sumreal +=  series[t]*Math.cos(2*Math.PI * t * k / n);
                sumimag += -series[t]*Math.sin(2*Math.PI * t * k / n);
            }
            dft[(k-start)*2]   = sumreal;
            dft[(k-start)*2+1] = sumimag;
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
    
    private double[][] MCB(Instances data) {
        double[][][] dfts = new double[data.numInstances()][][];
        
        int in = 0;
        for (Instance inst : data) {
            dfts[in++] = performDFT(disjointWindows(toArrayNoClass(inst))); //approximation
        }
        
        int numInsts = dfts.length;
        int numWindowsPerInst = dfts[0].length;
        int totalNumWindows = numInsts*numWindowsPerInst;

        breakpoints = new double[wordLength][alphabetSize]; 
        
        for (int letter = 0; letter < wordLength; ++letter) { //for each dft coeff
            
            //extract this column from all windows in all instances
            double[] column = new double[totalNumWindows];
            for (int inst = 0; inst < numInsts; ++inst)
                for (int window = 0; window < numWindowsPerInst; ++window) 
                    column[(inst * numWindowsPerInst) + window] = dfts[inst][window][letter];
            
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
    
    private Bag createBag(double[][] dfts) {
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
                if (dft[l] < breakpoints[l][bp]) {
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
        Bag bag = createBag(dfts); //discretisation
        bag.setClassVal(inst.classValue());
        return bag;
    }
    
    /**
     * Creates and returns new boss instance with shortened wordLength and corresponding
     * histograms, the boss instance passed in is UNCHANGED, if wordLengths are same, does nothing,
     * just returns passed in boss instance
     * 
     * @param newWordLength wordLength to shorten it to
     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
     */
    public static BOSS shortenHistograms(int newWordLength, final BOSS oldBoss) throws Exception {
        if (newWordLength == oldBoss.wordLength) //case of first iteration of word length search in ensemble
            return oldBoss;
        if (newWordLength > oldBoss.wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+oldBoss.wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+oldBoss.wordLength+", requested:"+newWordLength);
       
        //copies/updates meta data
        BOSS newBoss = new BOSS(oldBoss, newWordLength); 
        
        //shorten/copy actual histograms
        for (Bag bag : oldBoss.bags)
            newBoss.bags.add(shortenHistogram(newWordLength, bag));
            
        return newBoss;
    }
    
    private static Bag shortenHistogram(int newWordLength, Bag oldBag) {
        Bag newBag = new Bag();
        
        for (Entry<String, Integer> origWord : oldBag.entrySet()) {
            String shortWord = origWord.getKey().substring(0, newWordLength);
            
            Integer val = newBag.get(shortWord);
            if (val == null)
                val = 0;

            newBag.put(shortWord, val + origWord.getValue());
        }
        
        newBag.setClassVal(oldBag.getClassVal());
        
        return newBag;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data
 
        bags = new ArrayList<>(data.numInstances());
        for (Instance inst : data)
            bags.add(BOSSTransform(inst));
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
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Bag testBag = BOSSTransform(instance);
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

        //find dist FROM testBag TO all trainBags
        for (int i = 0; i < bags.size(); ++i) {
            double dist = BOSSdistance(testBag, bags.get(i)); 
            
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
            
            double dist = BOSSdistance(testBag, bags.get(i)); 
            
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
//        basicTest();      
        tonyTest();
    }
     public static void tonyTest() {
        System.out.println("BOSS Sanity Checks\n");
        DecimalFormat df = new DecimalFormat("##.####");
        int[] p={8,10,12,14,16}; 
        try {
            String pr="ItalyPowerDemand";
            Instances train = ClassifierTools.loadData(DataSets.problemPath+pr+"/"+pr+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData(DataSets.problemPath+pr+"/"+pr+"_TEST.arff");
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
     public static void basicTest() {
        System.out.println("BOSSBasicTest\n\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");

            System.out.println(train.relationName());
            
            BOSS boss = new BOSS(8,4,100,true);
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
}
