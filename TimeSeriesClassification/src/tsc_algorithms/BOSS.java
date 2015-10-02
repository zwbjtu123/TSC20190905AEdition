//TODO 
// ---equi-depth binning, how to handle non-int depths
// e.g 19 letters, 4 bins =>  5/5/5/4? 4/5/5/5? etc
// or naive imp: 19/4 = 4    =>   4/4/4/7
// or 19.0/4.0 = 4.66 => 5/5/5/4 
//
// but if 17 lets, 4 bins, 17.0/4.0 = 4.25 => 5/5/5/2
//
// ---bossdistance
// not symmetrical, so 'which' distance to use , d(1,2) or d(2,1) 
// aka always to dist FROM test histogram TO bag histogram? or visa versa
// does it make a differnece?

package tsc_algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import java.util.HashMap;
import java.util.Map.Entry;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.filters.timeseries.FFT;
import weka.filters.timeseries.FFT.Complex;

/**
 * BOSS classifier to be used with known parameters, for boss with parameter search,
 * use BOSSEnsemble.
 * 
 * Params: wordLength, alphabetSize, windowLength, normalise?
 * 
 * @author James
 */
public class BOSS implements Classifier {

    public static boolean debug = true;
    
    public ArrayList<Bag> bags;
    
    private int windowSize;
    private int wordLength;
    private int alphabetSize;
    private boolean norm;
    
    private boolean numerosityReduction = false; 
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
     * Used when shortening histograms, copies 'meta' data over, but with new 
     * word length
     */
    private BOSS(BOSS boss, int wordLength) {
        this.wordLength = wordLength;
        
        this.windowSize = boss.windowSize;
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.numerosityReduction = boss.numerosityReduction; 
        this.alphabet = boss.alphabet;
        
        bags = new ArrayList<>();
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
    
    public void generateAlphabet() {
        String[] alphabetSymbols = { "a","b","c","d","e","f","g","h","i","j" };
        
        alphabet = new String[alphabetSize];
        for (int i = 0; i < alphabetSize; ++i)
            alphabet[i] = alphabetSymbols[i];
    }
    
    private double[][] slidingWindow(double[] data) {
        double[][] subSequences = new double[data.length-windowSize+1][windowSize];
        
        for (int windowStart = 0; windowStart+windowSize-1 < data.length; ++windowStart) { 
            //copy the elements windowStart to windowStart+windowSize from data into 
            //the subsequence matrix at position windowStart
            System.arraycopy(data,windowStart,subSequences[windowStart],0,windowSize);
        }
        
        return subSequences;
    }
    
    private double[][] performDFT(double[][] windows) {
        double[][] dfts = new double[windows.length][wordLength];
        FFT fft = new FFT();
        for (int i = 0; i < windows.length; ++i) {
            double[] dft = DFT(windows[i]);
            int startIndex = norm ? 2 : 0; //drop first fourier coeff (mean value) if normalising 
            System.arraycopy(dft, startIndex, dfts[i], 0, wordLength); 
            //take first 'wordlength' (WL) values, i.e WL/2 real values, WL/2 imag values
        }
        return dfts;
    }
    
    public double[] DFT(double[] series) {
        //taken from FFT.java but 
        //converted to return just a double[] size n*2, { real1, imag1, real2, imag2 .... }
        //instead of Complex[] size n
    
        //todo check if can shorten process based on knowledge of word length, instead 
        //of performing full dft every time and then only taking first few values 
        
        int n=series.length;
        double[] dft=new double[n*2];
        for (int k = 0; k < n; k++) {  // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < series.length; t++) {  // For each input element
                    sumreal +=  series[t]*Math.cos(2*Math.PI * t * k / n);
                    sumimag += -series[t]*Math.sin(2*Math.PI * t * k / n);
            }
            dft[k*2]=sumreal;
            dft[k*2+1]=sumimag;
        }
        return dft;
    }
    
    private double[][] MCB(double[][] dfts) {
        //will return a list of breakpoints for each column (each letter)
        double[/*column*/][/*list of breakpoints*/] breakpoints = new double[dfts[0].length][]; 
        
        for (int c = 0; c < dfts[0].length; ++c) { //go along each column (dft coeff), i.e each letter
            
            //extract column from all windows
            double[] column = new double[dfts.length];
            for (int r = 0; r < column.length; ++r) 
                column[r] = dfts[r][c];
            
            //sort, and run through to find breakpoints for equi-depth bins
            Arrays.sort(column);
            
            int binDepth = 0, bin = 0;
            double targetBinDepth = (double)dfts.length / (double)alphabetSize; 
            
            breakpoints[c] = new double[alphabetSize];
            for (int r = 0; r < column.length; ++r) {
                if (binDepth++ > targetBinDepth) {
                    breakpoints[c][bin++] = column[r];
                    binDepth = 0;
                }
                breakpoints[c][alphabetSize-1] = Double.MAX_VALUE; //last one always = infinity
            }
        }
    
        return breakpoints;
    }
    
    private Bag toSFABag(double[][] dfts, double[][] breakpoints) {
        Bag bag = new Bag();
        String lastWord = "";
        
//        debug("numwindows:"+dfts.length+" numletters:"+dfts[0].length);
//        debug(dfts.length + " = " + breakpoints.length + " ?");
//        debug(breakpoints[0].length + " = " + alphabetSize + " ?");
        
        for (int w = 0; w < dfts.length; ++w) { //for each window 
            
            assert(dfts.length == wordLength);
            
            //build word
            String word = "";
            for (int l = 0; l < wordLength; ++l) {//for each letter
                
                assert(breakpoints[l].length == alphabetSize);
                
                for (int bp = 0; bp < alphabetSize; ++bp) {//run through breakpoints until right one found
//                    debug("w:"+w+" l:"+l+" bp:"+bp);
                    if (dfts[w][l] < breakpoints[l][bp]) {
                        word += alphabet[bp]; //add corresponding letter to word
                        break;
                    }
                }
            }
            
            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord))
                continue;
            else {
                Integer val = bag.get(word);
                if (val == null)
                    val = 0;
                bag.put(word, ++val);   
            }
            
            lastWord = word;
        }
        
        return bag;
    }
    
    /**
     * Converts passed instance to a double array with the class value removed if present
     * @param inst
     * @return 
     */
    public static double[] toArrayNoClass(Instance inst) {
        int length = inst.numAttributes();
        if (inst.classIndex() >= 0)
            --length;
        
        double[] data = new double[length];
        
        for (int i=0, j=0; i < inst.numAttributes(); ++i) { 
            if (inst.classIndex() != i)
                data[j++] = inst.value(i);
        }
        
        return data;
    }
    
    public Bag BOSSTransform(Instance inst) {
        double[] data = toArrayNoClass(inst);
        
        double[][] dfts = performDFT(slidingWindow(data)); //approximation    
        double[][] breakpoints = MCB(dfts); //breakpoints for discretization   
        Bag bag = toSFABag(dfts, breakpoints); //form words and put in bag
        bag.setClassVal(inst.classValue());
        return bag;
    }
    
    /**
     * Creates and returns NEW boss instance with shortened wordLength and corresponding
     * histograms, the boss instance passed in is UNCHANGED.
     * 
     * @param newWordLength wordLength to shorten it to, if equal, does nothing
     * @param boss built boss classifier with wordLength >= newWordLength
     * @return NEW boss classifier with newWordLength
     * @throws Exception if newWordLength > wordLength
     */
    public static BOSS shortenHistograms(int newWordLength, final BOSS oldBoss) throws Exception {
        if (newWordLength == oldBoss.wordLength)
            return oldBoss;
        if (newWordLength > oldBoss.wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+oldBoss.wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested: " + newWordLength);
       
        BOSS newBoss = new BOSS(oldBoss, newWordLength); 
        
        int i = 0;
        try { 
            for (Bag bag : oldBoss.bags) {
                newBoss.bags.add(shortenHistogram(newWordLength, bag));
                i++;
            }
        } catch (Exception e) {
            System.out.println("oldwl:"+oldBoss.wordLength);
            System.out.println("newwl:"+newWordLength);
            System.out.println("oldboss:"+oldBoss);
            System.out.println("newboss:"+newBoss);
            System.out.println("oldBossbagsize:"+oldBoss.bags.size());
            System.out.println("newBossbagsize:"+newBoss.bags.size());
            System.out.println("got to bag " + i);
        }
            
            
        return newBoss;
    }
    
    public static Bag shortenHistogram(int newWordLength, Bag bag) {
        
        Bag newBag = new Bag();
        
        for (Entry<String, Integer> origWord : bag.entrySet()) {
            String shortWord = origWord.getKey().substring(0, newWordLength);
            
            Integer val = newBag.get(shortWord);
            if (val == null)
                val = 0;
            
            newBag.put(shortWord, val + origWord.getValue());
        }
        
        return newBag;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        bags = new ArrayList<>(data.numInstances());
        
        for (Instance inst : data)
            bags.add(BOSSTransform(inst));
    }

    /**
     * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a)
     * 
     * @param testInst
     * @param trainInst
     * @return 
     */
    public double BOSSdistance(Bag testInst, Bag trainInst) {
        double dist = 0.0;
        
        for (Entry<String, Integer> entry : testInst.entrySet()) {
            Integer testVal = entry.getValue();
            Integer trainVal = trainInst.get(entry.getKey());
            if (trainVal == null)
                trainVal = 0;
            dist += (testVal-trainVal)*(testVal-trainVal);
        }
        
        return dist;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Bag testBag = BOSSTransform(instance);
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

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
     * the classifier every time (since n-1 histograms would be identical each time anyway), therefore this classifies 
     * the instance at the index passed while ignoring its own corresponding histogram 
     * 
     * @param inst to classify
     * @param leaveOut histogram to ignore
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
        throw new UnsupportedOperationException("Not supported yet. i did something here but it was dumb so removed until i fix"); //To change body of generated methods, choose Tools | Templates.
//        Bag testBag = BOSSTransform(instance);
//
//        double[] dist = new double[instance.numClasses()];
//        double sum = 0.0;
//        for (int i = 0; i < dist.length; ++i) {
//            dist[i] = BOSSdistance(testBag, bags.get(i)); 
//            sum+=dist[i];
//        }
//
//        if (sum != 0)
//            for (int i = 0; i < dist.length; ++i)
//                dist[i] /= sum;
//        
//        return dist;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        System.out.println("BOSStest\n\n");
        
        try {
            Instances train = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\Car\\\\Car_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\Car\\\\Car_TEST.arff");
            
            BOSS boss = new BOSS(8,4,100,false);
            boss.buildClassifier(train);
            
            System.out.println("ACC: " + ClassifierTools.accuracy(test, boss));
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        
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
     
    private static void printarr(double[] a) {
        for (int i = 0; i < a.length; i++)
            System.out.print(a[i] + " ");
        System.out.println("");
    }
    
    private static void debug(String msg) {
        if (debug) 
            System.out.println(msg);
    }
}
