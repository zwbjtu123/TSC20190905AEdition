package weka.classifiers;

import static JamesStuff.BasicExperiments.UCRvsmdata;
import fileIO.OutFile;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.timeseries.BagOfPatterns;
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

    Instances corpus;
    //double[][] classWeights; //class weight matrix
    public kNN knn;
    
    private BagOfPatterns bop;
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
      
        knn = new kNN(); //default to 1NN, Euclidean distance

        useParamSearch = true;
    }
    
    public SAXVSM(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
        this.SAX_alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        bop = new BagOfPatterns(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
        knn = new kNN(); //default to 1NN, Euclidean distance
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
    public static int[] parameterSearch(Instances data) {
//        System.out.println("SAXVSM_ParamSearch\n\n");
  
        double bestAcc = 0.0;
        int bestAlpha = 0, bestWord = 0, bestWindowSize = 0;
        int numTests = 5;

        int minWinSize = (int)((data.numAttributes()-1) / 10.0);
        int maxWinSize = (int)((data.numAttributes()-1) / 2.0);
        int winInc = (int)((maxWinSize - minWinSize) / 10.0); 

//        System.out.println("serieslength:"+(data.numAttributes()-1)+"    max:"+maxWinSize+"   min:"+minWinSize+"    inc:"+winInc);
        
        for (int alphaSize = 2; alphaSize <= 10; alphaSize++) {
            for (int winSize = minWinSize; winSize <= maxWinSize; winSize+=winInc) {
                for (int wordSize = 2; wordSize <= winSize/2; wordSize*=2) { //lin BoP suggestion
//                for (int interSize = 2; interSize <= 10; interSize++) { //arbitrary, minimizing dictionary size suggestion
                
                    SAXVSM vsm = new SAXVSM(wordSize,alphaSize,winSize);
                    double acc = ClassifierTools.crossValidationWithStats(vsm, data, data.numInstances())[0][0];//leave-one-out cv
//                    double acc = ClassifierTools.crossValidationWithStats(vsm, data, 2)[0][0];//2-fold
                    
//                    System.out.println("i/a/w/acc : "+wordSize+"/"+alphaSize+"/"+winSize+"/"+acc);
              
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
            throw new Exception("SAXVSM_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        if (useParamSearch) {
            int[] params = parameterSearch(data);
            
            this.PAA_intervalsPerWindow = params[0];
            this.SAX_alphabetSize = params[1];
            this.windowSize = params[2];
            
            bop = new BagOfPatterns(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
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
        
        data = bop.process(data);
//        
//        System.out.println("\n\nBOP");
//        System.out.println(data);
//        System.out.println("\n\n");
//        
        int numClasses = data.numClasses();
        int numInstances = data.numInstances();
        int numTerms = bop.dictionary.size();
        
        double[] classValues = data.attributeToDoubleArray(data.classIndex());
        //initialise class weights
        double[][] classWeights = new double[numClasses][];
        for (int i = 0; i < numClasses; ++i) {
            classWeights[i] = new double[numTerms];
            for (int j = 0; j < numTerms; ++j)
                classWeights[i][j] = 0; //cant remember whether java defaults array elements to 0, to be absolutely sure
        }

        //build class bags
        for (int i = 0; i < numInstances; ++i)
            for (int j = 0; j < numTerms; ++j)
                classWeights[(int)classValues[i]][j] += data.get(i).value(j);
        
        
//        //TESTING TO SEE IF THIS MAKES A DIFFERENCE
//        for (int i = 0; i < numClasses; ++i) 
//            intervalNorm(classWeights[i]);
//        
        
//        System.out.println("\n\npre weighting");
//        for (int i = 0; i < numClasses; ++i) {
//            for (int j = 0; j < numTerms; ++j)
//                System.out.print(classWeights[i][j] +" ");
//            System.out.println("");
//        }
        
        //apply tf x idf
        for (int i = 0; i < numTerms; ++i) { //for each term
            double df = 0; //document frequency
            //as i read it 'number of bags where term t appears' 
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
                    //check this just in case, done to avoid log calculations
                    //if df == num classes -> idf = log(N/df) = log(1) = 0 surely
                    //but not mentioned in paper, only df == 0 case mentioned, to avoid divide by 0
                    for (int j = 0; j < numClasses; ++j) 
                        classWeights[j][i] = 0;
                }      
            }
        }
        
//        System.out.println("\n\npost weighting");
//        for (int i = 0; i < numClasses; ++i) {
//            for (int j = 0; j < numTerms; ++j)
//                System.out.print(String.format("%3f ", classWeights[i][j]));
//            System.out.println("");
//        }
        
        
//        //        //TESTING TO SEE IF THIS MAKES A DIFFERENCE
//        for (int i = 0; i < numClasses; ++i) 
//            intervalNorm(classWeights[i]);
//        
        
        
        corpus = new Instances(data, numClasses);
        for (int i = 0; i < numClasses; ++i)
            corpus.add(new SparseInstance(1.0, classWeights[i]));
        
        knn.buildClassifier(corpus);
    }
    
//    @Override
//    public void buildClassifier(Instances data) throws Exception {
//         
//        if (data.classIndex() != data.numAttributes()-1)
//            throw new Exception("SAXVSM_BuildClassifier: Class attribute not set as last attribute in dataset");
//        
//        int numClasses = data.numClasses();
//        int numInstances = data.numInstances();
//        int numTerms = bop.dictionaryIndices.size();
//        
//        //store class values
//        double[] classValues = data.attributeToDoubleArray(data.classIndex());
//        
//        //convert each series to bop form
//        double[][] bopmatrix = new double[numInstances][];
//        for (int i = 0; i < numInstances; i++)
//            bopmatrix[i] = bop.buildBag(data.get(i));
//                
//        //initialise class weights
//        classWeights = new double[numClasses][];
//        for (int i = 0; i < numClasses; ++i) {
//            classWeights[i] = new double[numTerms];
//            for (int j = 0; j < numTerms; ++j)
//                classWeights[i][j] = 0; //cant remember whether java defaults doubles to 0, meh
//        }
//
//        //build class bags
//        for (int i = 0; i < numInstances; ++i)
//            for (int j = 0; j < numTerms; ++j)
//                classWeights[(int)classValues[i]][j] += bopmatrix[i][j];
//
//               
////DEBUG
////        
////        System.out.println("numClasses: " + numClasses);
////        System.out.println("numTerms: " + numTerms);
////        System.out.println("numInsts: " + numInstances);
////        System.out.print("classvalues:");
////        for (int i = 0; i < numInstances; ++i) 
////            System.out.print(classValues[i] + " ");
////        System.out.println("\n");
////        
////        double origSize = 0.0;
////        for (int i = 0; i < numInstances; ++i) 
////            for (int j = 0; j < numTerms; ++j)
////                origSize += bopmatrix[i][j];
////        
////        double afterSize = 0.0;
////        for (int i = 0; i < numClasses; ++i) 
////            for (int j = 0; j < numTerms; ++j)
////                afterSize += classWeights[i][j];
////        
////        System.out.println("should be equal: " + origSize + " " + afterSize);
////        
////        for (int i = 0; i < numClasses; ++i) {
////            double classSize = 0.0;
////            for (int j = 0; j < numTerms; ++j)
////                classSize += classWeights[i][j];
////            
////            System.out.println("class" + i + ": " + classSize);
////        }
////
////        System.out.println("\n\npre weighting");
////        for (int i = 0; i < numClasses; ++i) {
////            for (int j = 0; j < numTerms; ++j)
////                System.out.print(classWeights[i][j] +" ");
////            System.out.println("");
////        }
// //ENDDEBUG
//        
//        //apply tf x idf
//        for (int i = 0; i < numTerms; ++i) { //for each term
//            double df = 0; //document frequency
//            //as i read it 'number of bags where term t appears' 
//            //NOT how many times it appears in total, e.g appears 3 times in bag a, 2 times in bag b, df = 2, NOT 5
//            
//            for (int j = 0; j < numClasses; ++j) //find how many classes this term appears in
//                if (classWeights[j][i] != 0)
//                    ++df;
//            
//            if (df != 0) { //if it appears
//                if (df != numClasses) { //but not in all, apply weighting
//                    for (int j = 0; j < numClasses; ++j) 
//                        if (classWeights[j][i] != 0) 
//                            classWeights[j][i] = Math.log(1 + classWeights[j][i]) * Math.log(numClasses / df);                
//                }
//                else { //appears in all
//                    //check this just in case, done to avoid log calculations
//                    //if df == num classes -> idf = log(N/df) = log(1) = 0 surely
//                    //but not mentioned in paper, only df == 0 case mentioned, to avoid divide by 0
//                    for (int j = 0; j < numClasses; ++j) 
//                        classWeights[j][i] = 0;
//                }      
//            }
//                
//        }
// 
// //DEBUG
////        System.out.println("\n\npost weighting");
////        for (int i = 0; i < numClasses; ++i) {
////            for (int j = 0; j < numTerms; ++j)
////                System.out.print(classWeights[i][j] +" ");
////            System.out.println("");
////        }
// //ENDDEBUG
//        
//        //this code tried using instance.mergeinstance, KEEPING EVERYTHING SPARSE
//        //'merge' actually just adds on the values of the second onto the first
//        //i.e a = [1,2,3], b = [2,3,4], a.merge(b) = [1,2,3,2,3,4]
////        Instances classMatrix = new Instances(matrix, matrix.numClasses());
////        
////        //merge each series in each CLASS
////        double[] classValues = matrix.attributeToDoubleArray(matrix.classIndex());
////        
////        for (int c = 0; c < matrix.numClasses(); ++c) {
////            boolean firstInClass = true;
////            Instance firstClassInst = null;
////            
////            for (int i = 0; i < classValues.length; ++i) {
////                if (classValues[i] == c)
////                    if (firstInClass) {
////                        firstClassInst = matrix.get(i);
////                        firstInClass = false;
////                    }
////                    else
////                        firstClassInst = firstClassInst.mergeInstance(matrix.get(i));
////            }
////            
////            if (firstClassInst != null)
////                classMatrix.add(firstClassInst);
////            
////        }
////        
////        System.out.println("initialmatrix");
////        System.out.println("numClasses: " + matrix.numClasses());
////        System.out.println("numInsts: " + matrix.numInstances());
////        System.out.println("\nnewmatrix");
////        System.out.println("numClasses: " + classMatrix.numClasses());
////        System.out.println("numInsts: " + classMatrix.numInstances());
////        
////        System.out.println("\n\n" + classMatrix);
//    }

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
        //int numClasses = classWeights.length;
        int numClasses = corpus.numInstances();
        
        double[] termFreqs = bop.bagToArray(bop.buildBag(instance));
        intervalNorm(termFreqs);
        
//        return knn.distributionForInstance(instance);
        
        double[] similarities = new double[numClasses];
        
//        for (int i = 0; i < termFreqs.length; ++i) 
//            System.out.print(termFreqs[i] + " ");
//        System.out.println("");
        
//        System.out.print("SIMILARITIES ");
        double sum = 0.0;
        for (int i = 0; i < numClasses; ++i) {
            //similarities[i] = cosineSimilarity(classWeights[i], termFreqs); 
            similarities[i] = cosineSimilarity(corpus.get(i).toDoubleArray(), termFreqs, termFreqs.length); 
            sum+=similarities[i];
//            System.out.print(similarities[i] + " ");
        }
//        System.out.println("");
        
        
//        System.out.println("DISTRIBUTIIONS");
        if (sum != 0) {
            //you moron, want all to sum to 1, not just all in range 0 to 1
            //PROBABILITY that case is in each class
            //intervalNorm(similarities);
            
            for (int i = 0; i < numClasses; ++i) {
                similarities[i] /= sum;
//                System.out.println(similarities[i]);
            }
        }
//        System.out.println("");
        
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
            fullTest();
            
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
        
//        System.out.println("SAXVSMtest\n\n");
//        
//        System.out.println("PRE");
//        System.out.println("train");
//        System.out.println(train);
//        System.out.println("\ntest");
//        System.out.println(test);
        
        SAXVSM vsm = new SAXVSM(8,4,100);
        vsm.buildClassifier(train);
//
//        System.out.println("\nCORPUS");
//        System.out.println(vsm.corpus);
//        System.out.println("\n\n");
//        
        System.out.println("\n\nACCURACY " + ClassifierTools.accuracy(test, vsm));
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
