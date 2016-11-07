package tsc_algorithms.boss;


import utilities.generic_storage.ComparablePair;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import utilities.ClassifierTools;
import utilities.BitWord;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

/**
 * BOSSSpatialPyramids classifier to be used with known parameters, for boss with parameter search, use BOSSSpatialPyramidsEnsemble.
 * 
 * Params: wordLength, alphabetSize, windowLength, normalise?
 * 
 * @author James Large. Enhanced by original author Patrick Schaefer
 */
public class BOSSSpatialPyramids implements Classifier, Serializable {
    
    protected BitWord [][] SFAwords; //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
    public ArrayList<SPBag> bags; //histograms of words of the current wordlength with numerosity reduction applied (if selected)
    protected double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;
    
    protected double inverseSqrtWindowSize;
    protected int windowSize;
    protected int wordLength;
    protected int alphabetSize;
    protected boolean norm;
    
    protected int levels = 0;
    protected double levelWeighting = 0.5;
    protected int seriesLength;
    
    protected boolean numerosityReduction = true; 
    
    protected static final long serialVersionUID = 1L;

    public BOSSSpatialPyramids(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        
        this.levels = levels;
    }
    
    /**
     * Used when shortening histograms, copies 'meta' data over, but with shorter 
     * word length, actual shortening happens separately
     */
    public BOSSSpatialPyramids(BOSSSpatialPyramids boss, int wordLength) {
        this.wordLength = wordLength;
        
        this.windowSize = boss.windowSize;
        this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.numerosityReduction = boss.numerosityReduction; 
        //this.alphabet = boss.alphabet;
        
        this.SFAwords = boss.SFAwords;
        this.breakpoints = boss.breakpoints;
        
        this.levelWeighting = boss.levelWeighting;
        this.levels = boss.levels;
        this.seriesLength = boss.seriesLength;
        
        bags = new ArrayList<>(boss.bags.size());
    }
    

//    public BOSSSpatialPyramids_Redo(BOSSSpatialPyramids_Redo boss, double weighting) {
//        this.wordLength = boss.wordLength;
//        
//        this.windowSize = boss.windowSize;
//        this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
//        this.alphabetSize = boss.alphabetSize;
//        this.norm = boss.norm;
//        this.numerosityReduction = boss.numerosityReduction; 
//        //this.alphabet = boss.alphabet;
//        
//        this.SFAwords = boss.SFAwords;
//        this.breakpoints = boss.breakpoints;
//        
//        this.levelWeighting = weighting;
//        this.levels = boss.levels;
//        this.seriesLength = boss.seriesLength;
//        
//        bags = new ArrayList<>(boss.bags.size());
//    }
    
    //map of <word, level> => count
    public static class SPBag extends HashMap<ComparablePair<BitWord, Integer>, Double> {
        double classVal;
        
        public SPBag() {
            super();
        }
        
        public SPBag(int classValue) {
            super();
            classVal = classValue;
        }

        public double getClassVal() { return classVal; }
        public void setClassVal(double classVal) { this.classVal = classVal; }       
    }

    public int getWindowSize() { return windowSize; }
    public int getWordLength() { return wordLength; }
    public int getAlphabetSize() { return alphabetSize; }
    public boolean isNorm()     { return norm; }
    public int getLevels()   { return levels; }
    public double getLevelWeighting() { return levelWeighting; }
    
    /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? } 
     */
    public int[] getParameters() {
        return new int[] { wordLength, alphabetSize, windowSize };
    }
    
    public void clean() {
        SFAwords = null;
    }
    
    protected double[][] slidingWindow(double[] data) {
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
    
    protected double[][] performDFT(double[][] windows) {
        double[][] dfts = new double[windows.length][wordLength];
        for (int i = 0; i < windows.length; ++i) {
            dfts[i] = DFT(windows[i]);
        }
        return dfts;
    }
    
    protected double stdDev(double[] series) {
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
    protected double[] DFT(double[] series) {
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
    
    private double[] DFTunnormed(double[] series) {
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
        
        double[] dft = new double[outputLength*2];
        double twoPi = 2*Math.PI / n;
        
        for (int k = start; k < start + outputLength; k++) {  // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) {  // For each input element
                sumreal +=  series[t]*Math.cos(twoPi * t * k);
                sumimag += -series[t]*Math.sin(twoPi * t * k);
            }
            dft[(k-start)*2]   = sumreal;
            dft[(k-start)*2+1] = sumimag;
        }
        return dft;
    }
    
    private double[] normalizeDFT(double[] dft, double std) {
      double normalisingFactor = (std > 0? 1.0 / std : 1.0) * inverseSqrtWindowSize;
      for (int i = 0; i < dft.length; i++) {
        dft[i] *= normalisingFactor;
      }
      return dft;
    }
    
    private double[][] performMFT(double[] series) {
        // ignore DC value?
        int startOffset = norm ? 2 : 0;
        int l = wordLength;
        l = l + l % 2; // make it even
        double[] phis = new double[l];
        for (int u = 0; u < phis.length; u += 2) {
            double uHalve = -(u + startOffset) / 2;
            phis[u] = realephi(uHalve, windowSize);
            phis[u + 1] = complexephi(uHalve, windowSize);
        }
        // means and stddev for each sliding window
        int end = Math.max(1, series.length - windowSize + 1);
        double[] means = new double[end];
        double[] stds = new double[end];
        calcIncreamentalMeanStddev(windowSize, series, means, stds);
        // holds the DFT of each sliding window
        double[][] transformed = new double[end][];
        double[] mftData = null;
        for (int t = 0; t < end; t++) {
            // use the MFT
            if (t > 0) {
                for (int k = 0; k < l; k += 2) {
                    double real1 = (mftData[k] + series[t + windowSize - 1] - series[t - 1]);
                    double imag1 = (mftData[k + 1]);
                    double real = complexMulReal(real1, imag1, phis[k], phis[k + 1]);
                    double imag = complexMulImag(real1, imag1, phis[k], phis[k + 1]);
                    mftData[k] = real;
                    mftData[k + 1] = imag;
                }
            } // use the DFT for the first offset
            else {
                mftData = Arrays.copyOf(series, windowSize);
                mftData = DFTunnormed(mftData);
            }
            // normalization for lower bounding
            transformed[t] = normalizeDFT(Arrays.copyOf(mftData, l), stds[t]);
        }
        return transformed;
    }
    private void calcIncreamentalMeanStddev(int windowLength, double[] series, double[] means, double[] stds) {
        double sum = 0;
        double squareSum = 0;
        // it is faster to multiply than to divide
        double rWindowLength = 1.0 / (double) windowLength;
        double[] tsData = series;
        for (int ww = 0; ww < windowLength; ww++) {
            sum += tsData[ww];
            squareSum += tsData[ww] * tsData[ww];
        }
        means[0] = sum * rWindowLength;
        double buf = squareSum * rWindowLength - means[0] * means[0];
        stds[0] = buf > 0 ? Math.sqrt(buf) : 0;
        for (int w = 1, end = tsData.length - windowLength + 1; w < end; w++) {
            sum += tsData[w + windowLength - 1] - tsData[w - 1];
            means[w] = sum * rWindowLength;
            squareSum += tsData[w + windowLength - 1] * tsData[w + windowLength - 1] - tsData[w - 1] * tsData[w - 1];
            buf = squareSum * rWindowLength - means[w] * means[w];
            stds[w] = buf > 0 ? Math.sqrt(buf) : 0;
        }
    }

    private static double complexMulReal(double r1, double im1, double r2, double im2) {
        return r1 * r2 - im1 * im2;
    }

    private static double complexMulImag(double r1, double im1, double r2, double im2) {
        return r1 * im2 + r2 * im1;
    }

    private static double realephi(double u, double M) {
        return Math.cos(2 * Math.PI * u / M);
    }

    private static double complexephi(double u, double M) {
        return -Math.sin(2 * Math.PI * u / M);
    }
    
    protected double[][] disjointWindows(double [] data) {
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
    
    protected double[][] MCB(Instances data) {
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
                    column[(inst * numWindowsPerInst) + window] = Math.round(dfts[inst][window][letter]*100.0)/100.0;   
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
    protected SPBag createSPBagSingle(double[][] dfts) {
        SPBag bag = new SPBag();
        BitWord lastWord = new BitWord();
        
        int wInd = 0;
        int trivialMatchCount = 0;
        
        for (double[] d : dfts) {
            BitWord word = createWord(d);
//            //add to bag, unless num reduction applies
//            if (numerosityReduction && word.equals(lastWord))
//                continue;
//
//            addWordToPyramid(word, wInd, bag);
//            
////            ComparablePair<BitWord, Integer> key = new ComparablePair<>(word, 0);
////            Integer val = bag.get(key);
////            if (val == null)
////                val = 0;
////            bag.put(key, ++val);   
//            
//            lastWord = word;
//            ++wInd;
            
            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord)) {
                ++trivialMatchCount;
                ++wInd;
            }
            else {
                //if a run of equivalent words, those words essentially representing the same 
                //elongated pattern. still apply numerosity reduction, however use the central
                //time position of the elongated pattern to represent its position
                addWordToPyramid(word, wInd - (trivialMatchCount/2), bag);

                lastWord = word;
                trivialMatchCount = 0;
                ++wInd;
            }
        }
        
        applyPyramidWeights(bag);
        
        return bag;
    }
    
    protected BitWord createWord(double[] dft) {
        BitWord word = new BitWord(wordLength);
        for (int l = 0; l < wordLength; ++l) {//for each letter
            for (int bp = 0; bp < alphabetSize; ++bp) {//run through breakpoints until right one found
                if (dft[l] <= breakpoints[l][bp]) {
                    word.push(bp); //add corresponding letter to word
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
    protected static double[] toArrayNoClass(Instance inst) {
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
     * @return BOSSSpatialPyramidsTransform-ed bag, built using current parameters
     */
    public SPBag BOSSSpatialPyramidsTransform(Instance inst) {
//        double[][] dfts = performDFT(slidingWindow(toArrayNoClass(inst))); //approximation     
//        SPBag bag = createSPBagSingle(dfts); //discretisation/bagging
//        bag.setClassVal(inst.classValue());
        
        double[][] mfts = performMFT(toArrayNoClass(inst)); //approximation     
        SPBag bag2 = createSPBagSingle(mfts); //discretisation/bagging
        bag2.setClassVal(inst.classValue());
        
//        if (!bag2.equals(bag)) {
//          System.err.println("Error!");
//        }
        return bag2;
    }
//    public SPBag BOSSSpatialPyramidsTransform(Instance inst) {
//        double[][] dfts = performDFT(slidingWindow(toArrayNoClass(inst))); //approximation     
//        SPBag bag = createSPBagSingle(dfts); //discretisation/bagging
//        bag.setClassVal(inst.classValue());
//        return bag;
//    }
    
//    /**
//     * Creates and returns new boss instance with shortened wordLength and corresponding
//     * histograms, the boss instance passed in is UNCHANGED, if wordLengths are same, does nothing,
//     * just returns passed in boss instance
//     * 
//     * @param newWordLength wordLength to shorten it to
//     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
//     */
//    public static BOSSSpatialPyramids_Redo shortenHistograms(int newWordLength, final BOSSSpatialPyramids_Redo oldBoss) throws Exception {
//        if (newWordLength == oldBoss.wordLength) //case of first iteration of word length search in ensemble
//            return oldBoss;
//        if (newWordLength > oldBoss.wordLength)
//            throw new Exception("Cannot incrementally INCREASE word length, current:"+oldBoss.wordLength+", requested:"+newWordLength);
//        if (newWordLength < 2)
//            throw new Exception("Invalid wordlength requested, current:"+oldBoss.wordLength+", requested:"+newWordLength);
//       
//        //copies/updates meta data
//        BOSSSpatialPyramids_Redo newBoss = new BOSSSpatialPyramids_Redo(oldBoss, newWordLength); 
//        
//        //shorten/copy actual histograms
//        for (SPBag bag : oldBoss.bags)
//            newBoss.bags.add(shortenHistogram(newWordLength, bag));
//            
//        return newBoss;
//    }
//    
//    private static SPBag shortenHistogram(int newWordLength, SPBag oldSPBag) {
//        SPBag newSPBag = new SPBag();
//        
//        for (Entry<String, Integer> origWord : oldSPBag.entrySet()) {
//            String shortWord = origWord.getKey().substring(0, newWordLength);
//            
//            Integer val = newSPBag.get(shortWord);
//            if (val == null)
//                val = 0;
//                
//            newSPBag.put(shortWord, val + origWord.getValue());
//        }
//        
//        newSPBag.setClassVal(oldSPBag.getClassVal());
//        
//        return newSPBag;
//    }
    
        /**
     * Shortens all bags in this BOSSSpatialPyramids_Redo instance (histograms) to the newWordLength, if wordlengths
     * are same, instance is UNCHANGED
     * 
     * @param newWordLength wordLength to shorten it to
     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
     */
    public BOSSSpatialPyramids buildShortenedSPBags(int newWordLength) throws Exception {
        if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
            return this;
        if (newWordLength > wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);
       
        BOSSSpatialPyramids newBoss = new BOSSSpatialPyramids(this, newWordLength);
        
        //build hists with new word length from SFA words, and copy over the class values of original insts
        for (int i = 0; i < bags.size(); ++i) {
            SPBag newSPBag = createSPBagFromWords(newWordLength, SFAwords[i], true);   
            newSPBag.setClassVal(bags.get(i).getClassVal());
            newBoss.bags.add(newSPBag);
        }
        
        return newBoss;
    }
    
    protected SPBag shortenSPBag(int newWordLength, int bagIndex) {
        SPBag newSPBag = new SPBag();
        
        for (BitWord word : SFAwords[bagIndex]) {
            BitWord shortWord = new BitWord(word);
            shortWord.shortenByFourierCoefficient();
            
            Double val = newSPBag.get(shortWord);
            if (val == null)
                val = 0.0;
                
            newSPBag.put(new ComparablePair<BitWord, Integer>(shortWord, 0), val + 1.0);
        }
        
        return newSPBag;
    }
    
    /**
     * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
     * @param wordLengthSearching if true, length of each SFAwords word assumed to be 16, 
     *      and need to shorten it to whatever actual value needed in this particular version of the 
     *      classifier. if false, this is a standalone classifier with pre-defined wordlength (etc),
     *      and therefore sfawords are that particular length already, no need to shorten
     */
    protected SPBag createSPBagFromWords(int thisWordLength, BitWord[] words, boolean wordLengthSearching) {
        SPBag bag = new SPBag();
        BitWord lastWord = new BitWord();
        
        int wInd = 0;
        int trivialMatchCount = 0; //keeps track of how many words have been the same so far
        
        for (BitWord w : words) {
            BitWord word = new BitWord(w);
            if (wordLengthSearching)
                word.shorten(16-thisWordLength); //TODO hack, word.length=16=maxwordlength, wordLength of 'this' BOSSSpatialPyramids instance unreliable, length of SFAwords = maxlength
            
            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord)) {
                ++trivialMatchCount;
                ++wInd;
            }
            else {
                //if a run of equivalent words, those words essentially representing the same 
                //elongated pattern. still apply numerosity reduction, however use the central
                //time position to represent its position
                addWordToPyramid(word, wInd - (trivialMatchCount/2), bag);

                lastWord = word;
                trivialMatchCount = 0;
                ++wInd;
            }
        }
        
        applyPyramidWeights(bag);
        
        return bag;
    }
    
    protected void changeNumLevels(int newLevels) {
        //curently, simply remaking bags from words
        //alternatively: un-weight all bags, add(run through SFAwords again)/remove levels, re-weight all
        
        if (newLevels == this.levels)
            return;
        
        this.levels = newLevels;
        
        for (int inst = 0; inst < bags.size(); ++inst) {
            SPBag bag = createSPBagFromWords(wordLength, SFAwords[inst], true); //rebuild bag
            bag.setClassVal(bags.get(inst).classVal);
            bags.set(inst, bag); //overwrite old
        }
    }
    
    protected void applyPyramidWeights(SPBag bag) {
        for (Entry<ComparablePair<BitWord, Integer>, Double> ent : bag.entrySet()) {
            //find level that this quadrant is on
            int quadrant = ent.getKey().var2;
            int qEnd = 0; 
            int level = 0; 
            while (qEnd < quadrant) {
                int numQuadrants = (int)Math.pow(2, ++level);
                qEnd+=numQuadrants;
            }
            
            double val = ent.getValue() * (Math.pow(levelWeighting, levels-level-1)); //weighting ^ (levels - level)
            bag.put(ent.getKey(), val); 
        }
    }
    
    protected void addWordToPyramid(BitWord word, int wInd, SPBag bag) {
        int qStart = 0; //for this level, whats the start index for quadrants
        //e.g level 0 = 0
        //    level 1 = 1
        //    level 2 = 3
        for (int l = 0; l < levels; ++l) {
            //need to do the cell finding thing in the regular grid
            int numQuadrants = (int)Math.pow(2, l);
            int quadrantSize = seriesLength / numQuadrants;
            int pos = wInd + (windowSize/2); //use the middle of the window as its position
            int quadrant = qStart + (pos/quadrantSize); 
            
            ComparablePair<BitWord, Integer> key = new ComparablePair<>(word, quadrant);
            Double val = bag.get(key);

            if (val == null)
                val = 0.0;
            bag.put(key, ++val);   
            
            qStart += numQuadrants;
        }
    }

    protected BitWord[] createSFAwords(Instance inst) throws Exception {
//        double[][] dfts = performDFT(slidingWindow(toArrayNoClass(inst))); //approximation     
//        String[] words = new String[dfts.length];
//        for (int window = 0; window < dfts.length; ++window) 
//            words[window] = createWord(dfts[window]);//discretisation
            
        double[][] dfts2 = performMFT(toArrayNoClass(inst)); //approximation     
        BitWord[] words2 = new BitWord[dfts2.length];
        for (int window = 0; window < dfts2.length; ++window) 
            words2[window] = createWord(dfts2[window]);//discretisation
        
        return words2;
    }
   
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSSpatialPyramids_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        seriesLength = data.numAttributes()-1;
        
        breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data
 
        SFAwords = new BitWord[data.numInstances()][];
        bags = new ArrayList<>(data.numInstances());
        
        for (int inst = 0; inst < data.numInstances(); ++inst) {
            SFAwords[inst] = createSFAwords(data.get(inst));
            
            SPBag bag = createSPBagFromWords(wordLength, SFAwords[inst], false);
            bag.setClassVal(data.get(inst).classValue());
            bags.add(bag);
        }
    }

    /**
     * Computes BOSSSpatialPyramids distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a)
     * @return distance FROM instA TO instB
     */
    public double BOSSSpatialPyramidsDistance(SPBag instA, SPBag instB) {
        double dist = 0.0;
        
        //find dist only from values in instA
        for (Entry<ComparablePair<BitWord, Integer>, Double> entry : instA.entrySet()) {
            Double valA = entry.getValue();
            Double valB = instB.get(entry.getKey());
            if (valB == null)
                valB = 0.0;
            dist += (valA-valB)*(valA-valB);
        }
        
        return dist;
    }
    
       /**
     * Computes BOSSSpatialPyramids distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a).
     * 
     * Quits early if the dist-so-far is greater than bestDist (assumed is in fact the dist still squared), and returns Double.MAX_VALUE
     * 
     * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
     */
    public double BOSSSpatialPyramidsDistance(SPBag instA, SPBag instB, double bestDist) {
        double dist = 0.0;
        
        //find dist only from values in instA
        for (Entry<ComparablePair<BitWord, Integer>, Double> entry : instA.entrySet()) {
            Double valA = entry.getValue();
            Double valB = instB.get(entry.getKey());
            if (valB == null)
                valB = 0.0;
            dist += (valA-valB)*(valA-valB);
            
            if (dist > bestDist)
                return Double.MAX_VALUE;
        }
        
        return dist;
    }
    
    public double histogramIntersection(SPBag instA, SPBag instB) {
        //min vals of keys that exist in only one of the bags will always be 0
        //therefore want to only bother looking at counts of words in both bags
        //therefore will simply loop over words in a, skipping those that dont appear in b
        //no need to loop over b, since only words missed will be those not in a anyway
        
        double sim = 0.0;
        
        for (Entry<ComparablePair<BitWord, Integer>, Double> entry : instA.entrySet()) {
            Double valA = entry.getValue();
            Double valB = instB.get(entry.getKey());
            if (valB == null)
                continue;
            
            sim += Math.min(valA,valB);
        }
        
        return sim;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        SPBag testSPBag = BOSSSpatialPyramidsTransform(instance);
        
        double bestSimilarity = 0.0;
        double nn = -1.0;

        for (int i = 0; i < bags.size(); ++i) {
            double similarity = histogramIntersection(testSPBag, bags.get(i));
            
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                nn = bags.get(i).getClassVal();
            }
        }
        
        //if no bags had ANY similarity, just randomly guess 0
        //found that this occurs in <1% of test cases for certain parameter sets 
        //in the ensemble 
        if (nn == -1.0)
            nn = 0.0; 
        
        return nn;
    }
    
    /**
     * Used within BOSSSpatialPyramidsEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild 
     * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies 
     * the instance at the index passed while ignoring its own corresponding histogram 
     * 
     * @param test index of instance to classify
     * @return classification
     */
    public double classifyInstance(int test) {
        double bestSimilarity = 0.0;
        double nn = -1.0;

        SPBag testSPBag = bags.get(test);
        
        for (int i = 0; i < bags.size(); ++i) {
            if (i == test) //skip 'this' one, leave-one-out
                continue;
            
            double similarity = histogramIntersection(testSPBag, bags.get(i));
            
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
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
    
    public static void main(String[] args) throws Exception{
        //Minimum working example
        String dataset = "BeetleFly";
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TEST.arff");
        
        //example parameters found through the ensemble, 1.0 training accuracy reported
        int windowSize = 10;
        int alphabetSize = 4;
        int wordLength = 58;
        int levels = 2;
        boolean norm = true;
        
        Classifier c = new BOSSSpatialPyramids(windowSize, alphabetSize, wordLength, norm, levels); 
        c.buildClassifier(train);
        double accuracy = ClassifierTools.accuracy(test, c);
        
        System.out.println("BOSSSpatialPyramids("+windowSize+","+wordLength+","+alphabetSize+","+norm+","+levels+") accuracy on " + dataset + " fold 0 = " + accuracy);
        
        //Other examples/tests
//        detailedFold0Test(dataset);
    }
    
    public static void detailedFold0Test(String dset) {
        System.out.println("BOSSSpatialPyramids DetailedTest\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
            System.out.println(train.relationName());
            
            int windowSize = 10;
            int alphabetSize = 4;
            int wordLength = 58;
            int levels = 2;
            boolean norm = true;
            
            BOSSSpatialPyramids boss = new BOSSSpatialPyramids(windowSize, alphabetSize, wordLength, norm, levels);
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
}
