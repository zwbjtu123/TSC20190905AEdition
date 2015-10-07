package tsc_algorithms;

import static JamesStuff.BasicExperiments.UCRvsmdata;
import fileIO.OutFile;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.timeseries.BagOfPatternsFilter;
import weka.filters.timeseries.SAX;

/**
 * Classifier using SAX and Vector Space Model.
 * 
 * In training, generates class weighting matrix for SAX patterns found in the series,
 * in testing uses cosine similarity to find most similar class 
 * 
 * @author James
 */
public class SAXVSM implements Classifier {

    Instances transformedData;
    Instances corpus;

    private BagOfPatternsFilter bop;
    private int PAA_intervalsPerWindow;
    private int SAX_alphabetSize;
    private int windowSize;
    
    private FastVector alphabet;
    
    private final boolean useParamSearch; //does user want parameter search to be performed
    
    public static String[] SAXVSMPaperDatasets = {     
        "Adiac",
        "Beef",
        "CBF",
        "Coffee",
        "ECG200",
        "FaceAll",
        "FaceFour",
        "fish",
        "GunPoint",
        "Lightning2",
        "Lightning7",
        "OliveOil",
        "OSULeaf",
        "SyntheticControl",
        "SwedishLeaf",
        "Trace",
        "TwoPatterns",
        "wafer",
        "yoga"
    };
    
    public static double[] SAXVSMPaperErrors = {     
        0.381, 0.033, 0.002, 0.0, 0.140, 0.207, 0.0, 0.017, 0.007, 
        0.196, 0.301, 0.100, 0.107, 0.010, 0.251, 0.0, 0.004, 0.0006, 0.164
    };
    
    public SAXVSM() {
        this.PAA_intervalsPerWindow = -1;
        this.SAX_alphabetSize = -1;
        this.windowSize = -1;

        useParamSearch = true;
    }
    
    public SAXVSM(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
        this.SAX_alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
        alphabet = SAX.getAlphabet(SAX_alphabetSize);
        
        useParamSearch = false;
    }
    
    public int getPAA_intervalsPerWindow() {
        return PAA_intervalsPerWindow;
    }

    public int getSAX_alphabetSize() {
        return SAX_alphabetSize;
    }

    public int getWindowSize() {
        return windowSize;
    }
    
    /**
     * Performs cross validation on given data for varying parameter values, returns 
     * parameter set which yielded greatest accuracy
     * 
     * @param data Data to perform cross validation testing on
     * @return { numIntervals, alphabetSize, slidingWindowSize } 
     */
    public static int[] parameterSearch(Instances data) throws Exception {

        double bestAcc = 0.0;
        int bestAlpha = 0, bestWord = 0, bestWindowSize = 0;
        int numTests = 5;

        int minWinSize = (int)((data.numAttributes()-1) / 10.0);
        int maxWinSize = (int)((data.numAttributes()-1) / 2.0);
//        int winInc = (int)((maxWinSize - minWinSize) / 10.0); 
        int winInc = 1;
      
        for (int alphaSize = 2; alphaSize <= 8; alphaSize++) {
            for (int winSize = minWinSize; winSize <= maxWinSize; winSize+=winInc) {
                for (int wordSize = 2; wordSize <= winSize/2; wordSize*=2) { //lin BoP suggestion            
                    SAXVSM vsm = new SAXVSM(wordSize,alphaSize,winSize);
                    
                    double acc = vsm.crossValidate(data); //leave-one-out without doing bop transformation every fold (still applying tfxidf)
//                    double acc = ClassifierTools.crossValidationWithStats(vsm, data, data.numInstances())[0][0];//leave-one-out cv
//                    double acc = ClassifierTools.crossValidationWithStats(vsm, data, 2)[0][0];//2-fold

                    System.out.println(acc);
                    
                    if (acc > bestAcc) {
                        bestAcc = acc;
                        bestAlpha = alphaSize;
                        bestWord = wordSize;
                        bestWindowSize = winSize;
                    }
                }
            }
        }

        return new int[] { bestWord, bestAlpha, bestWindowSize};
    }
    
    private double crossValidate(Instances data) throws Exception {
        double correct = 0;
        
        transformedData = bop.process(data);
        
        for (int i = 0; i < data.numInstances(); ++i) {
            corpus = tfxidf(transformedData, i);
            
            if (classifyInstance(data.get(i)) == data.get(i).classValue())
                ++correct;
        }
            
        return correct /  data.numInstances();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("SAXVSM_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        if (useParamSearch) {
            int[] params = parameterSearch(data);
            
            this.PAA_intervalsPerWindow = params[0];
            this.SAX_alphabetSize = params[1];
            this.windowSize = params[2];
            
            bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
            alphabet = SAX.getAlphabet(SAX_alphabetSize);
        }
        
        if (PAA_intervalsPerWindow<0)
            throw new Exception("SAXVSM_BuildClassifier: Invalid PAA word size: " + PAA_intervalsPerWindow);
        if (PAA_intervalsPerWindow>windowSize)
            throw new Exception("SAXVSM_BuildClassifier: Invalid PAA word size, bigger than sliding window size: "
                    + PAA_intervalsPerWindow + "," + windowSize);
        if (SAX_alphabetSize<0 || SAX_alphabetSize>10)
            throw new Exception("SAXVSM_BuildClassifier: Invalid SAX alphabet size (valid=2-10): " + SAX_alphabetSize);
        if (windowSize<0 || windowSize>data.numAttributes()-1)
            throw new Exception("SAXVSM_BuildClassifier: Invalid sliding window size: " 
                    + windowSize + " (series length "+ (data.numAttributes()-1) + ")");
        
        transformedData = bop.process(data);
        
        corpus = tfxidf(transformedData);
    }
    
    /**
     * Given a set of *individual* series transformed into bop form, will return a corpus 
     * containing *class* bags made from that data with tfxidf weigh applied
     */
    public Instances tfxidf(Instances bopData) {
        return tfxidf(bopData, -1); //include all instances into corpus
    }
    
    /**
     * If skip = 0 to numInstances, will not include instance at that index into the corpus
     * Part of leave one out cv, while avoiding unnecessary repeats of the BoP transformation 
     */
    private Instances tfxidf(Instances bopData, int skip) {
        int numClasses = bopData.numClasses();
        int numInstances = bopData.numInstances();
        int numTerms = bopData.numAttributes()-1; //minus class attribute
        
        double[] classValues = bopData.attributeToDoubleArray(bopData.classIndex());
        //initialise class weights
        double[][] classWeights = new double[numClasses][];
        for (int i = 0; i < numClasses; ++i) {
            classWeights[i] = new double[numTerms];
            for (int j = 0; j < numTerms; ++j)
                classWeights[i][j] = 0; //cant remember whether java defaults array elements to 0, to be absolutely sure
        }

        //build class bags
        for (int i = 0; i < numInstances; ++i) {
            if (i == skip) //skip 'this' one, for leave-one-out cv
                continue;

            for (int j = 0; j < numTerms; ++j)
                classWeights[(int)classValues[i]][j] += bopData.get(i).value(j);
        }
            
        //apply tf x idf
        for (int i = 0; i < numTerms; ++i) { //for each term
            double df = 0; //document frequency
            //'number of bags where term t appears' 
            //NOT how many times it appears in total, e.g appears 3 times in bag a, 2 times in bag b, df = 2, NOT 5
            
            for (int j = 0; j < numClasses; ++j) //find how many classes this term appears in
                if (classWeights[j][i] != 0)
                    ++df;
            
            if (df != 0) { //if it appears
                if (df != numClasses) { //but not in all, apply weighting
                    for (int j = 0; j < numClasses; ++j) 
                        if (classWeights[j][i] != 0) 
                            classWeights[j][i] = Math.log(1 + classWeights[j][i]) * Math.log(numClasses / df);                
                }
                else { //appears in all
                    //done to avoid log calculations
                    //if df == num classes -> idf = log(N/df) = log(1) = 0 surely
                    //but not mentioned in paper, only df == 0 case mentioned, to avoid divide by 0
                    for (int j = 0; j < numClasses; ++j) 
                        classWeights[j][i] = 0;
                }      
            }
        }
        
        Instances tfxidfCorpus = new Instances(bopData, numClasses);
        for (int i = 0; i < numClasses; ++i)
            tfxidfCorpus.add(new SparseInstance(1.0, classWeights[i]));
        
        return tfxidfCorpus;
    }

    /**
     * Takes two vectors of equal length, and computes the cosine similarity between them.
     * 
     * @param a
     * @param b
     * @return  a.b / ( |a|*|b| )
     * @throws java.lang.Exception if a.length != b.length
     */
    public double cosineSimilarity(double[] a, double[] b) throws Exception {
        if (a.length != b.length)
            throw new Exception("Cannot calculate cosine similarity between vectors of different lengths "
                    + "(" + a.length + ", " + b.length + ")");
        
        double dotProd = 0.0, aMag = 0.0, bMag = 0.0;
        
        for (int i = 0; i < a.length; ++i) {
            dotProd += a[i]*b[i];
            aMag += a[i]*a[i];
            bMag += b[i]*b[i];
        }
        
        if (aMag == 0 || bMag == 0 || dotProd == 0)
            return 0;
        
        return dotProd / (Math.sqrt(aMag) * Math.sqrt(bMag));
    }
    
    /**
     * Takes two vectors, and computes the cosine similarity between them using the first n values in each vector.
     * 
     * To be used when e.g one or both vectors have class values as the last element, only compute
     * similarity up to values size-1
     * 
     * @param a
     * @param b
     * @param n Elements 0 to n-1 will be computed for similarity, elements n to size-1 ignored
     * @return  a.b / ( |a|*|b| )
     * @throws java.lang.Exception if n > a.length or b.length
     */
    public double cosineSimilarity(double[] a, double[] b, int n) throws Exception {
        if (n > a.length || n > b.length)
            throw new IllegalArgumentException("SAXVSM_CosineSimilarity n greater than vector lengths "
                    + "(a:" + a.length + ", b:" + b.length + " n:" + n + ")");
        
        double dotProd = 0.0, aMag = 0.0, bMag = 0.0;
        
        for (int i = 0; i < n; ++i) {
            dotProd += a[i]*b[i];
            aMag += a[i]*a[i];
            bMag += b[i]*b[i];
        }
        
        if (aMag == 0 || bMag == 0 || dotProd == 0)
            return 0;
        
        return dotProd / (Math.sqrt(aMag) * Math.sqrt(bMag));
    }
    
    private void intervalNorm(double[] r) {
        //copied from NormalizeCase and altered to work on a single double array 
        //rather than full instances

        double max=Double.MIN_VALUE, min=Double.MAX_VALUE;
        
        for(int j = 0 ; j < r.length ; j++) {
            if(r[j]>max)
                max=r[j];
            if(r[j]<min)
                min=r[j];
        }
        
        for(int j = 0; j < r.length; j++)
            r[j] = (r[j] - min) / (max - min);
        
    }
    
     /**
      * **BROKEN**
      * java.lang.ArrayIndexOutOfBoundsException: 2
	at weka.core.SparseInstance.toDoubleArray(SparseInstance.java:425)
	at tsc_algorithms.SAXVSM.classifyInstance(SAXVSM.java:340)
      * 
      * Used as part of a leave-one-out crossvalidation, to skip having to rebuild 
      * the classifier every time (since n-1 histograms would be identical each time anyway), therefore this classifies 
      * the instance at the index passed while ignoring its own corresponding histogram 
      * 
      * @param test index of instance to classify
      * @return classification
      */
    private double classifyInstance(int test) throws Exception {
        
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;

        double[] termFreqs = transformedData.get(test).toDoubleArray();
        
        for (int i = 0; i < corpus.numInstances(); ++i) {
            double dist = cosineSimilarity(corpus.get(i).toDoubleArray(), termFreqs, termFreqs.length - 1); 
            
            if (dist < bestDist) {
                bestDist = dist;
                nn = corpus.get(i).classValue();
            }
        }
        
        return nn;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        
        //int numClasses = classWeights.length;
        int numClasses = corpus.numInstances();
        
        double[] distribution = distributionForInstance(instance);
        
        double maxIndex = 0, max = distribution[0];
        for (int i = 1; i < numClasses; ++i)
            if (distribution[i] > max) {
                max = distribution[i];
                maxIndex = i;
            }
        
        return maxIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int numClasses = corpus.numInstances();
        
        double[] termFreqs = bop.bagToArray(bop.buildBag(instance));
        
        double[] similarities = new double[numClasses];
        

        double sum = 0.0;
        for (int i = 0; i < numClasses; ++i) {
            //similarities[i] = cosineSimilarity(classWeights[i], termFreqs); 
            similarities[i] = cosineSimilarity(corpus.get(i).toDoubleArray(), termFreqs, termFreqs.length); 
            sum+=similarities[i];
        }

        if (sum != 0) {

            for (int i = 0; i < numClasses; ++i) {
                similarities[i] /= sum;
            }
        }
        
        return similarities;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        System.out.println("SAXVSM");
        
        try {  
            //aaa();
            //paramSearchTest();
            //crossValidationTest();
            //fullTest();
            
            String path = "C:\\Users\\JamesL\\Documents\\UEA\\Internship\\DATA\\";
//            Instances train = ClassifierTools.loadData(path+"Coffee\\Coffee_TRAIN");
//            Instances test = ClassifierTools.loadData(path+"Coffee\\Coffee_TEST");
            
            Instances train = ClassifierTools.loadData(path+"Car\\Car_TRAIN");
            Instances test = ClassifierTools.loadData(path+"Car\\Car_TEST");
            
            basicTest(train, test);
            
            //very small dataset for testing by eye
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\Sheet2_3.arff");
            
            //two class, decent size
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TwoClassV1.arff");
            
            //five class, large size
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\FiveClassV1.arff");
//            all.deleteAttributeAt(0); //just name of bottle        
//            
//            int trainNum = (int) (all.numInstances() * 0.7);
//            int testNum = all.numInstances() - trainNum;
//            
//            Instances train = new Instances(all, 0, trainNum);
//            Instances test = new Instances(all, trainNum, testNum);
            
            
            
//            Randomize rand = new Randomize();
//            rand.setInputFormat(all);
//            for (int i = 0; i < all.numInstances(); ++i) {
//                rand.input(all.get(i));
//            }
//            rand.batchFinished();
//            
//            int trainNum = (int) (all.numInstances() * 0.7);
//            int testNum = all.numInstances() - trainNum;
//            
//            Instances train = new Instances(all, trainNum);
//            for (int i = 0; i < trainNum; ++i) 
//                train.add(rand.output());
//            
//            Instances test = new Instances(all, testNum);
//            for (int i = 0; i < testNum; ++i) 
//                test.add(rand.output());
                  
//            basicTest(train, test);
//            fullTest(train, test);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        
    }
    
    public static void paramSearchTest() throws Exception {
        Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\FiveClassV1.arff");
        all.deleteAttributeAt(0); //just name of bottle        

        int trainNum = (int) (all.numInstances() * 0.7);
        int testNum = all.numInstances() - trainNum;

        Instances train = new Instances(all, 0, trainNum);
        Instances test = new Instances(all, trainNum, testNum);

        int[] params = parameterSearch(train);

        SAXVSM vsm = new SAXVSM(params[0],params[1],params[2]);
        vsm.buildClassifier(train);

        System.out.println(ClassifierTools.accuracy(test, vsm));
    }
    
    public static void crossValidationTest() throws Exception {
        //five class, large size
        Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\FiveClassV1.arff");
        all.deleteAttributeAt(0); //just name of bottle     
        
        SAXVSM vsm = new SAXVSM(8,4,50);
        
        double[][] cv = ClassifierTools.crossValidationWithStats(vsm, all, 10);
        
        for (int i = 0; i < cv.length; i++) {
            for (int j = 0; j < cv[i].length; j++) {
                System.out.print(cv[i][j] + " ");
            }
            System.out.println("");
        }
                
    }
    
    public static void basicTest(Instances train, Instances test) throws Exception {

        SAXVSM vsm = new SAXVSM(8,4,100);
        vsm.buildClassifier(train);     
        System.out.println("\n\nACCURACY1 " + ClassifierTools.accuracy(test, vsm));
        
        SAXVSM vsm2 = new SAXVSM();
        vsm2.buildClassifier(train);     
        System.out.println("\n\nACCURACY2 " + ClassifierTools.accuracy(test, vsm2));
    }
    
    public static void fullTest() throws Exception {
        System.out.println("SAXVSM_FullTest\n\n");

        String path="C:\\Temp\\TESTDATA\\TSC Problems\\";
        
        OutFile out = new OutFile("SAXVSMpapertests.csv");  
        out.writeLine("data, paramSearchError, paperError, , wordlen, alphasize, winsize");
     
        int i =0;
        for (String fname : SAXVSMPaperDatasets) {
            Instances train = ClassifierTools.loadData(path+fname+"\\"+fname+"_TRAIN");
            Instances test = ClassifierTools.loadData(path+fname+"\\"+fname+"_TEST");

            SAXVSM vsm = new SAXVSM(); //user param search
            vsm.buildClassifier(train);

            double err = 1.0-ClassifierTools.accuracy(test, vsm);
            
            out.writeLine(fname + ", " + err + ", " + SAXVSMPaperErrors[i] + ", , " + vsm.getPAA_intervalsPerWindow() + ", " + vsm.getSAX_alphabetSize() + ", " + vsm.getWindowSize());
            System.out.println(fname + ", " + err + ", " + SAXVSMPaperErrors[i++] + ", " + vsm.getPAA_intervalsPerWindow() + ", " + vsm.getSAX_alphabetSize() + ", " + vsm.getWindowSize());
        }
        
        out.closeFile();
    }
    
    public static void aaa() throws Exception {
//        Instances train = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\Car\\\\Car_TRAIN.arff");
//        Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\Car\\\\Car_TEST.arff");
        
//        Instances train = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\GunPoint\\\\GunPoint_TRAIN.arff");
//        Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\GunPoint\\\\GunPoint_TEST.arff");
        
//        Instances train = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\CBF\\\\CBF_TRAIN.arff");
//        Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\CBF\\\\CBF_TEST.arff");
        
//        Instances train = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\ChlorineConcentration\\\\ChlorineConcentration_TRAIN.arff");
//        Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\ChlorineConcentration\\\\ChlorineConcentration_TEST.arff");
        
        String path="C:\\Temp\\TESTDATA\\TSC Problems\\";
        
//        OutFile out = new OutFile("MYsaxvsmtests.csv");
//   
//        out.writeLine("data, accuracy, error");
     
        for (String fname : UCRvsmdata) {
            System.out.println("DATA: " + fname);
            
            Instances train = ClassifierTools.loadData(path+fname+"\\"+fname+"_TRAIN");
            Instances test = ClassifierTools.loadData(path+fname+"\\"+fname+"_TEST");

            SAXVSM vsm = new SAXVSM(8,4,100); //fixed params
//            SAXVSM vsm = new SAXVSM(); //user param search
            vsm.buildClassifier(train);

            double acc = ClassifierTools.accuracy(test, vsm);
            double err= 1.0-acc;
            
//            out.writeLine(fname + ", " + acc + ", " + err);
            
            
            System.out.println("ACCURACY " + acc);
            System.out.println("ERROR " + err + "\n");
        }

//        out.closeFile();
    }
    

    
    @Override
    public String toString() { 
        return "SAXVSM";
    }

}
