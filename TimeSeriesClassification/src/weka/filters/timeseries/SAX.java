package weka.filters.timeseries;

import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Filter to reduce dimensionality of and discretise a normalised time series into SAX form. 
 * 
 * Attributes can be in two forms - discrete alphabet or real values 0 to alphabetsize-1
 * 
 * Default num of intervals = 10
 * Default alphabet size = 3
 *
 * @author James
 */
public class SAX extends SimpleBatchFilter {

    private int numIntervals = 10;
    private int alphabetSize = 3;
    private boolean useRealAttributes = false;
    private FastVector alphabet = null;
    
    
    private static final long serialVersionUID = 1L;
    
    //individual strings for each symbol in the alphabet, up to ten symbols
    private static final String[] alphabetSymbols = { "a","b","c","d","e","f","g","h","i","j" };
    
    public int getNumIntervals() {
        return numIntervals;
    }

    public int getAlphabetSize() {
        return alphabetSize;
    }
    
    public FastVector getAlphabet() {
        return alphabet;
    }
    
    public void setNumIntervals(int intervals) {
        numIntervals = intervals;
    }
    
    public void setAlphabetSize(int alphasize) {
        alphabetSize = alphasize;
    }
    
    public void useRealValuedAttributes(boolean b){
        useRealAttributes = b;
    }
    
    public void generateAlphabet() {
        alphabet = new FastVector();
        for (int i = 0; i < alphabetSize; ++i)
            alphabet.addElement(alphabetSymbols[i]);
    }

    //lookup table for the breakpoints for a gaussian curve where the area under 
    //curve T between Ti and Ti+1 = 1/a, 'a' being the size of the alphabet.
    //columns up to a=10 stored
    //lit. suggests that a = 3 or 4 is bet in almost all cases, up to 6 or 7 at most
    //for specific datasets
    public double[] generateBreakpoints(int alphabetSize) 
            throws Exception {
        
    	double maxVal = Double.MAX_VALUE;
    	double[] breakpoints = null;
        
    	switch(alphabetSize) {
            case 2: {  breakpoints = new double[]{ 0, maxVal }; break; }
            case 3: {  breakpoints = new double[]{-0.43, 0.43, maxVal }; break; }
            case 4: {  breakpoints = new double[]{-0.67, 0, 0.67, maxVal }; break; }
            case 5: {  breakpoints = new double[]{-0.84, -0.25, 0.25, 0.84, maxVal }; break; }
            case 6: {  breakpoints = new double[]{-0.97, -0.43, 0, 0.43, 0.97, maxVal }; break; }
            case 7: {  breakpoints = new double[]{-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, maxVal }; break; }
            case 8: {  breakpoints = new double[]{-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, maxVal}; break; }
            case 9: {  breakpoints = new double[]{-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, maxVal}; break; }
            case 10: { breakpoints = new double[]{-1.28 -0.84 -0.52 -0.25, 0.0, 0.25, 0.52, 0.84, 1.28, maxVal}; break; }
            
            default: { 
                throw new Exception("No breakpoints stored for alphabet size " + alphabetSize); 
            }
    	}
    	
    	return breakpoints;
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat)
            throws Exception {
        
        //Check all attributes are real valued, otherwise throw exception
        for (int i = 0; i < inputFormat.numAttributes(); i++) {
            if (inputFormat.classIndex() != i) {
                if (!inputFormat.attribute(i).isNumeric()) {
                    throw new Exception("Non numeric attribute not allowed for SAX conversion");
                }
            }
        }
        
        //Set up instances size and format. 
        FastVector attributes = new FastVector();
        
        //If the alphabet is to be considered as discrete values (i.e non real), 
        //generate nominal values based on alphabet size
        if(!useRealAttributes)
            generateAlphabet();
         
        Attribute att;
        String name;
    
        for (int i = 0; i < numIntervals; i++) {
            name = "SAXInterval_" + i;

            if (!useRealAttributes)
                att = new Attribute(name, alphabet);
            else
                att = new Attribute(name);

            attributes.addElement(att);
        }

        if (inputFormat.classIndex() >= 0) {	//Classification set, set class 
            //Get the class values as a fast vector			
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            attributes.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        
        Instances result = new Instances("SAX" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
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
        
        Instances output = determineOutputFormat(input);
        
        //Convert input to PAA format
        PAA paa = new PAA();
        paa.setNumIntervals(numIntervals);
        input = paa.process(input);
        
        //Now convert PAA -> SAX
        for (int i = 0; i < input.numInstances(); i++) {
            double[] data = input.instance(i).toDoubleArray();
            
            //remove class attribute if needed
            double[] temp;
            int c = input.classIndex();
            if(c >= 0) {
                temp=new double[data.length-1];
                System.arraycopy(data,0,temp,0,c); //assumes class attribute is in last index
                data=temp;
            }
            
            double[] intervals = convertSequence(data);
            
            //Now in SAX form, extract out the terms and set the attributes of new instance
            Instance newInstance;
            if (input.classIndex() >= 0)
                newInstance = new DenseInstance(numIntervals + 1);
            else
                newInstance = new DenseInstance(numIntervals);

            for (int j = 0; j < numIntervals; j++) {
                newInstance.setValue(j, intervals[j]);
                
                
                //from doc:
                //'setValue(attIndex, value).. 
                //...
                //...
                //value - the new attribute value (If the corresponding attribute is nominal (or a string) 
                //      then this is the new value's index as a double).'
                
                //so shouldnt need to do any checks or anything for useRealAttribute, 
                //should be handled automatically based on whatever happened in 
                //determineOutputFormat
            }
                
            if (input.classIndex() >= 0)
                newInstance.setValue(output.classIndex(), input.instance(i).classValue());

            output.add(newInstance);
        }
        
        return output;
    }
    
    public double[] convertSequence(double[] data) 
            throws Exception {
        double[] gaussianBreakpoints = generateBreakpoints(alphabetSize);
        
        for (int i = 0; i < numIntervals; ++i) {
            //SAX conversion
            //find symbol corresponding to each mean
            for (int j = 0; j < alphabetSize; ++j)
                if (data[i] < gaussianBreakpoints[j]) {
                    data[i] = j;
                    break;
                }
        }
        
        return data;
    }

    public String getRevision() {
        // TODO Auto-generated method stub
        return null;
    }

    public static void main(String[] args) {
        System.out.println("SAXtest\n\n");
        
        try {
            Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TwoClassV1.arff");
            test.deleteAttributeAt(0); //just name of bottle
            
            Normalize norm = new Normalize();
            norm.setScale(2.0);
            norm.setTranslation(-1.0);
            norm.setInputFormat(test);
            Instances normTest = Filter.useFilter(test, norm);
            
            SAX sax = new SAX();
            sax.setNumIntervals(50);
            sax.setAlphabetSize(3);     
            sax.useRealValuedAttributes(false);
            Instances result = sax.process(normTest);
            
            System.out.println(normTest);
            System.out.println("\n\n\nResults:\n\n");
            System.out.println(result);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }

}
