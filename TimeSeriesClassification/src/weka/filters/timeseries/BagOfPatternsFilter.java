package weka.filters.timeseries;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;
import javax.swing.InputVerifier;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.NormalizeCase;
import weka.filters.SimpleBatchFilter;

/**
 * Filter to transform time series into a bag of patterns representation.
 * i.e pass a sliding window over each series
 * normalise and convert each window to sax form
 * build a histogram of non-trivially matching patterns
 * 
 * Resulting in a bag (histogram) of patterns (SAX words) describing the high-level
 * structure of each timeseries
 *
 * default sliding window size = 100
 * default paainterval size = 6
 * default saxalphabetsize = 3
 * 
 * @author James
 */
public class BagOfPatternsFilter extends SimpleBatchFilter {

    public TreeSet<String> dictionary;
    
    private final int windowSize;
    private final int numIntervals;
    private final int alphabetSize;
    private boolean useRealAttributes = true;
    
    private boolean numerosityReduction = false; //can expand to different types of nr
    //like those in senin implementation later, if wanted
    
    private FastVector alphabet = null;
    
    private static final long serialVersionUID = 1L;

    public BagOfPatternsFilter(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.numIntervals = PAA_intervalsPerWindow;
        this.alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        alphabet = SAX.getAlphabet(SAX_alphabetSize);
    }
    
    public int getWindowSize() {
        return numIntervals;
    }
    
    public int getNumIntervals() {
        return numIntervals;
    }

    public int getAlphabetSize() {
        return alphabetSize;
    }
    
    public void useRealValuedAttributes(boolean b){
        useRealAttributes = b;
    }
    
    public void performNumerosityReduction(boolean b){
        numerosityReduction = b;
    }
    
    private HashMap<String, Integer> buildHistogram(double[][] patterns) {
        
        HashMap<String, Integer> hist = new HashMap<>();

        for (int i = 0; i < patterns.length; ++i) {   
            //convert to string                
            String word = "";
            for (int j = 0; j < patterns[i].length; ++j)
                word += (String) alphabet.get((int)patterns[i][j]);

            
            Integer val = hist.get(word);
            if (val == null)
                val = 0;
            
            hist.put(word, val+1);
        }
        
        return hist;
    }
    
    public HashMap<String, Integer> buildBag(Instance series) throws Exception {
        double[] data = series.toDoubleArray();

        //remove class attribute if needed
        double[] temp;
        int c = series.classIndex();
        if(c >= 0) {
            temp=new double[data.length-1];
            System.arraycopy(data,0,temp,0,c); //assumes class attribute is in last index
            data=temp;
        }
        
        //generate all subsequences of this instance
        double[][] patterns = slidingWindow(data);

//        System.out.println("SLIDINGWINDOW");
//        for(int i = 0; i < patterns.length; ++i) {
//            System.out.print("pattern" + i + ":\t");
//            for (int j = 0; j < patterns[i].length; ++j) 
//                System.out.print(patterns[i][j] + " ");
//            
//            System.out.println("");
//        }
        
        for (int i = 0; i < patterns.length; ++i) {
            try {
                NormalizeCase.standardNorm(patterns[i]);
            } catch(Exception e) { //TONY COMMENT: Adapt this so catches a specific exception (ArithmeticException)?
                //throws exception if zero variance
                //if zero variance, all values in window the same 
                //'normalised' version should essentially be all 0s? 
                for (int j = 0; j < patterns[i].length; ++j)
                    patterns[i][j] = 0;
            }
            patterns[i] = SAX.convertSequence(patterns[i], alphabetSize, numIntervals);
        }
       
//        System.out.println("SAXD");
//        for(int i = 0; i < patterns.length; ++i) {
//            System.out.print("pattern" + i + ":\t");
//            for (int j = 0; j < patterns[i].length; ++j) 
//                System.out.print(patterns[i][j] + " ");
//            
//            System.out.println("");
//        }
        
        if (numerosityReduction)    
            patterns = removeTrivialMatches(patterns);
        
//        System.out.println("REDUCED");
//        for(int i = 0; i < patterns.length; ++i) {
//            System.out.print("pattern" + i + ":\t");
//            for (int j = 0; j < patterns[i].length; ++j) 
//                System.out.print(patterns[i][j] + " ");
//            
//            System.out.println("");
//        }
        
        return buildHistogram(patterns);
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
    
    public double[][] removeTrivialMatches(double[][] patterns) {
        ArrayList<Integer> toKeep = new ArrayList<>(patterns.length);
        
        toKeep.add(0); 
        
        for (int i = 1; i < patterns.length; ++i)
            if (!identicalPattern(patterns[i], patterns[i-1]))
                toKeep.add(i);
            
        double[][] keptPatterns = new double[toKeep.size()][];
        
        for (int i = 0; i < keptPatterns.length; ++i)
            keptPatterns[i] = patterns[toKeep.get(i)];
        
        return keptPatterns;
    }
    
    private boolean identicalPattern(double[] a, double[] b) {
        for (int i = 0; i < a.length; ++i)
            if (a[i] != b[i])
                return false;
        
        return true;
    }
  
    @Override
    protected Instances determineOutputFormat(Instances inputFormat)
            throws Exception {
        
        //Check all attributes are real valued, otherwise throw exception
        for (int i = 0; i < inputFormat.numAttributes(); i++) {
            if (inputFormat.classIndex() != i) {
                if (!inputFormat.attribute(i).isNumeric()) {
                    throw new Exception("Non numeric attribute not allowed for BoP conversion");
                }
            }
        }

        FastVector attributes = new FastVector();
        for (String word : dictionary) 
            attributes.add(new Attribute(word));
        
        Instances result = new Instances("BagOfPatterns_" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        
        if (inputFormat.classIndex() >= 0) {	//Classification set, set class 
            //Get the class values as a fast vector			
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            
            result.insertAttributeAt(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals), result.numAttributes());
            result.setClassIndex(result.numAttributes() - 1);
        }
 
        return result;
    }

    @Override
    public String globalInfo() {
        return null;
    }

    @Override
    public Instances process(Instances input) 
            throws Exception {
        
        Instances inputCopy = new Instances(input);
        
        
        ArrayList< HashMap<String, Integer> > bags = new ArrayList<>(inputCopy.numInstances());
        dictionary = new TreeSet<>();
        
        for (int i = 0; i < inputCopy.numInstances(); i++) {
            bags.add(buildBag(inputCopy.get(i)));
            dictionary.addAll(bags.get(i).keySet());
        }
        
        Instances output = determineOutputFormat(inputCopy); //now that dictionary is known, set up output
        
        for (int i = 0; i < inputCopy.numInstances(); ++i) {
            double[] bag = bagToArray(bags.get(i));
            
            output.add(new SparseInstance(1.0, bag));
            output.get(i).setClassValue(inputCopy.get(i).classValue()); 
        }
        
        return output;
    }

    public double[] bagToArray(HashMap<String, Integer> bag) {
        double[] res = new double[dictionary.size()];
            
        int j = 0;
        for (String word : dictionary) {
            Integer val = bag.get(word);
            if (val != null)
                res[j] += val;
            ++j;
        }

        return res;
    }

    public String getRevision() {
        // TODO Auto-generated method stub
        return null;
    }

    public static void main(String[] args) {
        System.out.println("BoPtest\n\n");

//        try {
//            Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TwoClassV1.arff");
//            test.deleteAttributeAt(0); //just name of bottle
//          
//            BagOfPatterns bop = new BagOfPatterns(6,3,100);  
//            bop.useRealValuedAttributes(false);
//            Instances result = bop.process(test);
//            
//            System.out.println(result);
//        }
//        catch (Exception e) {
//            System.out.println(e);
//            e.printStackTrace();
//        }
    }
}
