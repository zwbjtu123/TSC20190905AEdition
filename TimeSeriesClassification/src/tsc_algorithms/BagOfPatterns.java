package tsc_algorithms;

import fileIO.OutFile;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.BagOfPatternsFilter;
import weka.filters.timeseries.SAX;

/**
 * Converts instances into Bag Of Patterns form, then applies 1NN 
 * 
 * @author James
 */
public class BagOfPatterns implements Classifier {

    public Instances matrix;
    public kNN knn;
    
    private BagOfPatternsFilter bop;
    private int PAA_intervalsPerWindow;
    private int SAX_alphabetSize;
    private int windowSize;
    
    private FastVector alphabet;
    
    private final boolean useParamSearch; //does user want parameter search to be performed
    
    public BagOfPatterns() {
        this.PAA_intervalsPerWindow = -1;
        this.SAX_alphabetSize = -1;
        this.windowSize = -1;

        knn = new kNN(); //defaults to 1NN, Euclidean distance

        useParamSearch=true;
    }
    
    public BagOfPatterns(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
        this.SAX_alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);       
        knn = new kNN(); //default to 1NN, Euclidean distance
        alphabet = SAX.getAlphabet(SAX_alphabetSize);
        
        useParamSearch=false;
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
    public static int[] parameterSearch(Instances data) {
//        System.out.println("LinBoP_ParamSearch\n\n");
  
        double bestAcc = 0.0;
        int bestAlpha = 0, bestWord = 0, bestWindowSize = 0;
        int numTests = 5;

        //Paper calls for some random window size search range of 15%-36% data length
        //Doesn't say whether it checks EVERY size in that range, currently assuming it does
        int minWinSize = (int)((data.numAttributes()-1) * (15.0/100.0));
        int maxWinSize = (int)((data.numAttributes()-1) * (36.0/100.0));
        int winInc = 1; //check every size in range
//        int winInc = (int)((maxWinSize - minWinSize) / 10.0); //check 10 values within that range
//        if (winInc < 1) winInc = 1;

        for (int alphaSize = 2; alphaSize <= 10; alphaSize++) {
            for (int winSize = minWinSize; winSize <= maxWinSize; winSize+=winInc) {
                for (int wordSize = 2; wordSize <= winSize/2; wordSize*=2) { //lin BoP suggestion
         
                    BagOfPatterns bop = new BagOfPatterns(wordSize, alphaSize, winSize);

                    double acc = ClassifierTools.crossValidationWithStats(bop, data, data.numInstances())[0][0];//leave-one-out cv
//                    double acc = ClassifierTools.crossValidationWithStats(vsm, data, 2)[0][0];//2-fold cv

                    if (acc > bestAcc) {
                        bestAcc = acc;
                        bestAlpha = alphaSize;
                        bestWord = wordSize;
                        bestWindowSize = winSize;
                    }
                }
            }
        }

//        System.out.println("\n\nbest accuracy: " + bestAcc);
//        System.out.println("best alphabet size: " + bestAlpha);
//        System.out.println("best num intervals: " + bestWord);
//        System.out.println("best window size: " + bestWindowSize);  
        
//        System.out.println(data.relationName() + " params: i/a/w/acc = "+bestWord+"/"+bestAlpha+"/"+bestWindowSize+"/"+bestAcc);
        
        return new int[] { bestWord, bestAlpha, bestWindowSize};
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("LinBoP_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        if (useParamSearch) {
            int[] params = parameterSearch(data);
            
            this.PAA_intervalsPerWindow = params[0];
            this.SAX_alphabetSize = params[1];
            this.windowSize = params[2];
            
            bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
            alphabet = SAX.getAlphabet(SAX_alphabetSize);
        }
        
        if (PAA_intervalsPerWindow<0)
            throw new Exception("LinBoP_BuildClassifier: Invalid PAA word size: " + PAA_intervalsPerWindow);
        if (PAA_intervalsPerWindow>windowSize)
            throw new Exception("LinBoP_BuildClassifier: Invalid PAA word size, bigger than sliding window size: "
                    + PAA_intervalsPerWindow + "," + windowSize);
        if (SAX_alphabetSize<0 || SAX_alphabetSize>10)
            throw new Exception("LinBoP_BuildClassifier: Invalid SAX alphabet size (valid=2-10): " + SAX_alphabetSize);
        if (windowSize<0 || windowSize>data.numAttributes()-1)
            throw new Exception("LinBoP_BuildClassifier: Invalid sliding window size: " 
                    + windowSize + " (series length "+ (data.numAttributes()-1) + ")");
        
        Instances dataCopy = new Instances(data);
        matrix = bop.process(data);
        
        knn.buildClassifier(matrix);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //convert to proper form
        double[] hist = bop.bagToArray(bop.buildBag(instance));
        
        Instances newInsts = new Instances(matrix, 1); //copy attribute data
        newInsts.add(new SparseInstance(1.0, hist));

//        new NormalizeCase().intervalNorm(newInsts);

        return knn.classifyInstance(newInsts.firstInstance());
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        //convert to proper form
        double[] hist = bop.bagToArray(bop.buildBag(instance));
        
        Instances newInsts = new Instances(matrix, 1); //copy attribute data
        newInsts.add(new SparseInstance(1.0, hist));

//        new NormalizeCase().intervalNorm(newInsts);
        
        return knn.distributionForInstance(newInsts.firstInstance());
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        System.out.println("BagofPatternsTest\n\n");
        
        try {
//            //very small dataset for testing by eye
////            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\Sheet2_Train2.arff");
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TwoClassV1.arff");
//            all.deleteAttributeAt(0); //just name of bottle
//            
//            
//            int trainNum = (int) (all.numInstances() * 0.7);
//            int testNum = all.numInstances() - trainNum;
//            
//            Instances train = new Instances(all, 0, trainNum);
//            Instances test = new Instances(all, trainNum, testNum);
//            
////            System.out.println("RAW TRAIN DATA");
////            System.out.println(train);
////            
////            System.out.println("\nRAW TEST DATA");
////            System.out.println(test);
//            
//            LinBagOfPatterns bop = new LinBagOfPatterns(6,3,100);
//            bop.buildClassifier(train);
//            
//            System.out.println(bop.matrix);
// 
////            System.out.println("\n\nDICTIONARY");
////            System.out.println(bop.dictionaryAttributes);
////            
////            System.out.println("\n\nTRAIN BOP");
////            System.out.println(bop.matrix);
////            
////            Instances testM = new Instances(bop.matrix, test.numInstances()); //copy attribute data
////            for (int i = 0; i < test.numInstances(); i++) {
////                double[] hist = bop.buildBag(test.get(i));
////                testM.add(new SparseInstance(1.0, hist));
////            }
////            
////            System.out.println("\n\nTEST BOP");
////            System.out.println(testM);
////            
////            System.out.println("");
////            for (int i = 0; i < test.numInstances(); i++) {
////                System.out.println(bop.classifyInstance(test.get(i)));
////            }
////            
////            System.out.println("");
////            for (int i = 0; i < test.numInstances(); ++i) {
////                double[] dist = bop.distributionForInstance(test.get(i));
////                
////                for (int j = 0; j < dist.length; ++j) 
////                    System.out.print(dist[j] + " ");
////                
////                System.out.println("");
////            }
//            
//            System.out.println("\nACCURACY TEST");
//            System.out.println(ClassifierTools.accuracy(test, bop));

            String fname = "LinBoPpapertests.csv";
            fullTest(fname);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        
    }
    
    @Override
    public String toString() { 
        return "BagOfPatterns";
    }
    
    //currently this goes through the papers shown on the bop paper
    //classifies using parameters shown there, parameter found using search
    //prints error rates, as well as paper rates shown on paper, and the correspoding parameters used
    public static void fullTest(String outFileName) throws Exception {
        System.out.println("LinBoP_FullTest\n\n");

        String path="C:\\Temp\\TESTDATA\\TSC Problems\\";
        
        OutFile out = new OutFile(outFileName);  
        out.writeLine("data, paramSearch, paperParams, given, , MYwordlen, MYalphasize, MYwinsize, , GIVwordlen, GIValphasize, GIVwinsize");
     
        int i = 0;
        for (String fname : UCRnames) {
            Instances train = ClassifierTools.loadData(path+fname+"\\"+fname+"_TRAIN");
            Instances test = ClassifierTools.loadData(path+fname+"\\"+fname+"_TEST");

            BagOfPatterns bopsearch = new BagOfPatterns(); //use param search
            bopsearch.buildClassifier(train);
            double searchErr = 1.0-ClassifierTools.accuracy(test, bopsearch);
            
            int[] params = BagOfPatterns.getUCRParameters(UCRnames[i]);
            BagOfPatterns bopfixed = new BagOfPatterns(params[1], params[2], params[0]); //use params given
            bopfixed.buildClassifier(train);
            double fixedErr = 1.0-ClassifierTools.accuracy(test, bopfixed);
            
            out.writeLine(fname + ", " + searchErr + ", " + fixedErr + ", " + givenBopErrRates[i] + ", , " 
                    + bopsearch.getPAA_intervalsPerWindow() + ", " + bopsearch.getSAX_alphabetSize() + ", " + bopsearch.getWindowSize() + ", , "
                    + params[1] + ", " + params[2] + ", " + params[0]);
            System.out.println(fname + ", " + searchErr + ", " + fixedErr + ", " + givenBopErrRates[i] + ", , " 
                    + bopsearch.getPAA_intervalsPerWindow() + ", " + bopsearch.getSAX_alphabetSize() + ", " + bopsearch.getWindowSize() + ", , "
                    + params[1] + ", " + params[2] + ", " + params[0]);
            
            ++i;
        }
        
        out.closeFile();
    }
    
    /**
     * Returns a list of optimal parameters (winSize, intervals, alphabetSize) for a SUBSET OF the UCR datasets as given 
     * in the Lin paper, if dataset properties not found, returns default 50, 6, 4.
     * 
     * @param dataset 
     * @return array size 3 { winSize, intervals, alphabetSize }
     */
    public static int[] getUCRParameters(String dataset) throws Exception {
        for (int i = 0; i < UCRnames.length; ++i)
            if (UCRnames[i].equals(dataset))
                return UCRparameters[i];
        
        throw new Exception("No parameter info for UCR dataset \'" + dataset + "\'");
    }
    
    //parameters for winsize/intervals/alphabet for each ucr dataset copied from
    //'Rotation-invarient similarity in time series using bag-of-patterns representation' Lin etal 2012
    //[0] = windowsize
    //[1] = intervals 
    //[2] = alphabetsize
    public static int[][] UCRparameters = {
        { 24, 4, 3 },
        { 32, 4, 9 }, 
        { 32, 4, 4 },
        { 32, 8, 3 },
        { 64, 4, 4 },
        { 40, 8, 4 },
        { 80, 8, 3 },
        { 48, 4, 4 },
        { 32, 4, 4 },
        { 32, 8, 5 },
        { 64, 4, 4 },
        { 128, 8, 4 }, 
        { 64, 8, 4 },
        { 32, 8, 4 },
        { 32, 8, 9 },
        { 80, 4, 9 },
        { 128, 8, 7 },
        { 80, 4, 2 },
        { 48, 4, 3 },
        { 160, 4, 6 }
    };
    
    public static String[] UCRnames = {
        "SyntheticControl",
        "GunPoint",
        "CBF",
        "FaceAll",
        "OSULeaf",
        "SwedishLeaf",
        "fiftywords",
        "Trace",
        "TwoPatterns",
        "wafer",
        "FaceFour",
        "Lightning2",
        "Lightning7",
        "ECG200",
        "Adiac",
        "yoga",
        "fish",
        "Beef",
        "Coffee",
        "OliveOil"
    };
    
    public static double[] givenBopErrRates = {
        0.037,
        0.027,
        0.013,
        0.219,
        0.256,
        0.198,
        0.466,
        0.0,
        0.129,
        0.003,
        0.023,
        0.164,
        0.466,
        0.150,
        0.432,
        0.170,
        0.074,
        0.433,
        0.036,
        0.133
    };
}
