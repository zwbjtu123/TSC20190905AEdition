/*
 * copyright: Anthony Bagnall
 * NOTE: As shapelet extraction can be time consuming, there is an option to output shapelets
 * to a text file (Default location is in the root dir of the project, file name "defaultShapeletOutput.txt").
 *
 * Default settings are TO NOT PRODUCE OUTPUT FILE - unless file name is changed, each successive filter will
 * overwrite the output (see "setLogOutputFile(String fileName)" to change file dir and name).
 *
 * To reconstruct a filter from this output, please see the method "createFilterFromFile(String fileName)".
 */
package weka.filters.timeseries.shapelet_transforms;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.ListIterator;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import weka.core.*;
import weka.core.shapelet.*;
import weka.filters.SimpleBatchFilter;

/**
 * A filter to transform a dataset by k shapelets. Once built on a training set,
 * the filter can be used to transform subsequent datasets using the extracted
 * shapelets.
 * <p>
 * See <a
 * href="http://delivery.acm.org/10.1145/2340000/2339579/p289-lines.pdf?ip=139.222.14.198&acc=ACTIVE%20SERVICE&CFID=221649628&CFTOKEN=31860141&__acm__=1354814450_3dacfa9c5af84445ea2bfd7cc48180c8">Lines,
 * J., Davis, L., Hills, J., Bagnall, A.: A shapelet transform for time series
 * classification. In: Proc. 18th ACM SIGKDD (2012)</a>
 *
 * @author Jason Lines
 */
public class FullShapeletTransform extends SimpleBatchFilter
{

    protected boolean cacheDoubleArrays = false;
    protected double[][] cachedDoubleArray;
    //Variables for experiments
    protected static long subseqDistOpCount;
    protected TreeMap<Double, Integer> classDistributions;
    
    //logFile
    PrintWriter opLogFile = null;
    PrintWriter pruneLogFile = null;
    String logFileName;
    public void setLogFileName(String s)
    {
        logFileName = s;
    }
    
    public void writeToLogFile(PrintWriter pw, String pattern, Object... args)
    {
        if(pw != null)
        {
            pw.printf(pattern, args);
            pw.flush();
        }
    }
    
    public void writeToLogFile(String pattern, Object... args)
    {
        writeToLogFile(opLogFile, pattern, args);
    }
    
    

    @Override
    public String globalInfo()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    protected boolean supressOutput = false; // defaults to print in System.out AS WELL as file, set to true to stop printing to console
    protected int minShapeletLength;
    protected int maxShapeletLength;
    protected int numShapelets;
    protected boolean shapeletsTrained;
    protected ArrayList<Shapelet> shapelets;
    protected String ouputFileLocation = "defaultShapeletOutput.txt"; // default store location
    protected boolean recordShapelets = true; // default action is to write an output file
    protected boolean roundRobin = false;

    public static int DEFAULT_NUMSHAPELETS = 100;
    public static int DEFAULT_MINSHAPELETLENGTH = 3;
    public static int DEFAULT_MAXSHAPELETLENGTH = 23;

    protected QualityMeasures.ShapeletQualityMeasure qualityMeasure;
    protected QualityMeasures.ShapeletQualityChoice qualityChoice;
    protected boolean useCandidatePruning;
    protected boolean useSeparationGap = false;
    protected boolean useRoundRobin = false;

    public void setUseSeparationGap(boolean b)
    {
        useSeparationGap = b;
    }

    public void setUseRoundRobin(boolean b)
    {
        useRoundRobin = b;
    }

    protected int candidatePruningStartPercentage;

    protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
    protected int[] dataSourceIDs;

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public FullShapeletTransform()
    {
        this(DEFAULT_NUMSHAPELETS, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
    }

    /**
     * Constructor for generating a shapelet transform from an ArrayList of
     * Shapelets.
     *
     * @param shapes
     */
    public FullShapeletTransform(ArrayList<Shapelet> shapes)
    {
        this();
        this.shapelets = shapes;
        this.shapeletsTrained = true;
        this.numShapelets = shapelets.size();
    }

    /**
     * Single param constructor: Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     */
    public FullShapeletTransform(int k)
    {
        this(k, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public FullShapeletTransform(int k, int minShapeletLength, int maxShapeletLength)
    {
        this(k, minShapeletLength, maxShapeletLength, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Full, exhaustive, constructor for a filter. Quality measure set via enum,
     * invalid selection defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     * @param qualityChoice the shapelet quality measure to be used with this
     * filter
     */
    public FullShapeletTransform(int k, int minShapeletLength, int maxShapeletLength, weka.core.shapelet.QualityMeasures.ShapeletQualityChoice qualityChoice)
    {
        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
        this.numShapelets = k;
        this.shapelets = new ArrayList<>();
        this.shapeletsTrained = false;
        this.useCandidatePruning = false;
        this.qualityChoice = qualityChoice;

        setQualityMeasure(qualityChoice);
    }

    /**
     * Returns the set of shapelets for this transform as an ArrayList.
     *
     * @return An ArrayList of Shapelets representing the shapelets found for
     * this Shapelet Transform.
     */
    public ArrayList<Shapelet> getShapelets()
    {
        return this.shapelets;
    }

    /**
     * Set the transform to round robin the data or not. This transform defaults
     * round robin to false to keep the instances in the same order as the
     * original data. If round robin is set to true, the transformed data will
     * be reordered which can make it more difficult to use the ensemble.
     *
     * @param val
     */
    public void setRoundRobin(boolean val)
    {
        this.roundRobin = val;
    }

    /**
     * Supresses filter output to the console; useful when running timing
     * experiments.
     */
    public void supressOutput()
    {
        this.supressOutput = true;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This
     * speeds up the transform processing time.
     */
    public void useCandidatePruning()
    {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = 10;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This
     * speeds up the transform processing time.
     *
     * @param percentage the percentage of data to be precocessed before pruning
     * is initiated. In most cases the higher the percentage the less effective
     * pruning becomes
     */
    public void useCandidatePruning(int percentage)
    {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = percentage;
    }

    /**
     * Mutator method to set the number of shapelets to be stored by the filter.
     *
     * @param k the number of shapelets to be generated
     */
    public void setNumberOfShapelets(int k)
    {
        this.numShapelets = k;
    }

    /**
     *
     * @return
     */
    public int getNumberOfShapelets()
    {
        return numShapelets;
    }

    /**
     * Turns off log saving; useful for timing experiments where speed is
     * essential.
     */
    public void turnOffLog()
    {
        this.recordShapelets = false;
    }

    /**
     * Set file path for the filter log. Filter log includes shapelet quality,
     * seriesId, startPosition, and content for each shapelet.
     *
     * @param fileName the updated file path of the filter log
     */
    public void setLogOutputFile(String fileName)
    {
        this.recordShapelets = true;
        this.ouputFileLocation = fileName;
    }

    /**
     *
     * @return
     */
    public boolean foundShapelets()
    {
        return shapeletsTrained;
    }

    /**
     * Mutator method to set the minimum and maximum shapelet lengths for the
     * filter.
     *
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public void setShapeletMinAndMax(int minShapeletLength, int maxShapeletLength)
    {
        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
    }

    /**
     * Mutator method to set the quality measure used by the filter. As with
     * constructors, default selection is information gain unless another valid
     * selection is specified.
     *
     * @return
     */
    public QualityMeasures.ShapeletQualityChoice getQualityMeasure()
    {
        return qualityChoice;
    }

    /**
     *
     * @param qualityChoice
     */
    public void setQualityMeasure(QualityMeasures.ShapeletQualityChoice qualityChoice)
    {
        this.qualityChoice = qualityChoice;
        switch (qualityChoice)
        {
            case F_STAT:
                this.qualityMeasure = new QualityMeasures.FStat();
                break;
            case KRUSKALL_WALLIS:
                this.qualityMeasure = new QualityMeasures.KruskalWallis();
                break;
            case MOODS_MEDIAN:
                this.qualityMeasure = new QualityMeasures.MoodsMedian();
                break;
            default:
                this.qualityMeasure = new QualityMeasures.InformationGain();
        }
    }

    /**
     *
     * @param f
     */
    public void setCandidatePruning(boolean f)
    {
        this.useCandidatePruning = f;
        this.candidatePruningStartPercentage = f ? 10 : 100;
    }

    /**
     * Sets the format of the filtered instances that are output. I.e. will
     * include k attributes each shapelet distance and a class value
     *
     * @param inputFormat the format of the input data
     * @return a new Instances object in the desired output format
     */
    //TODO: Fix depecrated FastVector
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException
    {

        if (this.numShapelets < 1)
        {
            throw new IllegalArgumentException("ShapeletFilter not initialised correctly - please specify a value of k that is greater than or equal to 1");
        }

        //Set up instances size and format.
        //int length = this.numShapelets;
        int length = this.shapelets.size();
        FastVector atts = new FastVector();
        String name;
        for (int i = 0; i < length; i++)
        {
            name = "Shapelet_" + i;
            atts.addElement(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0)
        {
            //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++)
            {
                vals.addElement(target.value(i));
            }
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Shapelets" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0)
        {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    protected void inputCheck(Instances dataInst) throws IllegalArgumentException
    {
        if (numShapelets < 1)
        {
            throw new IllegalArgumentException("Number of shapelets initialised incorrectly - please select value of k (Usage: setNumberOfShapelets");
        }

        int maxPossibleLength;
        maxPossibleLength = dataInst.instance(0).numAttributes();

        if (dataInst.classIndex() >= 0)
        {
            maxPossibleLength -= 1;
        }

        if (minShapeletLength < 1 || maxShapeletLength < 1 || maxShapeletLength < minShapeletLength || maxShapeletLength > maxPossibleLength)
        {
            throw new IllegalArgumentException("Shapelet length parameters initialised incorrectly");
        }
    }

    /**
     * The main logic of the filter; when called for the first time, k shapelets
     * are extracted from the input Instances 'data'. The input 'data' is
     * transformed by the k shapelets, and the filtered data is returned as an
     * output.
     * <p>
     * If called multiple times, shapelet extraction DOES NOT take place again;
     * once k shapelets are established from the initial call to process(), the
     * k shapelets are used to transform subsequent Instances.
     * <p>
     * Intended use:
     * <p>
     * 1. Extract k shapelets from raw training data to build filter;
     * <p>
     * 2. Use the filter to transform the raw training data into transformed
     * training data;
     * <p>
     * 3. Use the filter to transform the raw testing data into transformed
     * testing data (e.g. filter never extracts shapelets from training data,
     * therefore avoiding bias);
     * <p>
     * 4. Build a classifier using transformed training data, perform
     * classification on transformed test data.
     *
     * @param data the input data to be transformed (and to find the shapelets
     * if this is the first run)
     * @return the transformed representation of data, according to the
     * distances from each instance to each of the k shapelets
     */
    @Override
    public Instances process(Instances data) throws IllegalArgumentException
    {
        //check the input data is correct and assess whether the filter has been setup correctly.
        inputCheck(data);

        //instantiate the caching array here, so it gets refreshed if we're using a test set.
        int dataSize = data.numInstances();

        if (cacheDoubleArrays)
        {
            cachedDoubleArray = new double[dataSize][];
        }

        //checks if the shapelets haven't been found yet, finds them if it needs too.
        if (!shapeletsTrained)
        {
            try
            {
                File f = new File(logFileName+"_opLog.csv");
                //make the dirs on the files parent directors.
                f.getParentFile().mkdirs(); 
                opLogFile = new PrintWriter(f);
                writeToLogFile("TRAIN\n");
                writeToLogFile("candidateId,candidateStartPos,candidateLength,opCount,totalOpCount\n");
                
                f = new File(logFileName+"_pruneLog.csv");
                //make the dirs on the files parent directors.
                f.getParentFile().mkdirs();
                pruneLogFile = new PrintWriter(f);
                writeToLogFile(pruneLogFile,"TRAIN\n");
                writeToLogFile(pruneLogFile,"candidateId, candidateStartPos, candidate.length, prunedSeries, totalPruned\n");
            }
            catch (FileNotFoundException ex)
            {
                System.out.println("Couldn't create log file " + ex);    
            }
            
            trainShapelets(data);
        }

        //build the transformed dataset with the shapelets we've found either on this data, or the previous training data
        return buildTansformedDataset(data);
    }

    protected void trainShapelets(Instances data)
    {

        int dataSize = data.numInstances();
        // shapelets discovery has not yet been caried out, so this must be training data
        dataSourceIDs = new int[dataSize];
        if (roundRobin)
        {
            //Reorder the data in round robin order
            data = roundRobinData(data, dataSourceIDs);
        }
        else
        {
            for (int i = 0; i < dataSize; i++)
            {
                dataSourceIDs[i] = i;
            }
        }

        shapelets = findBestKShapeletsCache(data); // get k shapelets
        shapeletsTrained = true;

        outputPrint(shapelets.size() + " Shapelets have been generated");

        //Reorder the training data and reset the shapelet indexes
        if (roundRobin)
        {
            resetDataOrder(data, dataSourceIDs);
            resetShapeletIndices(shapelets, dataSourceIDs);
        }

    }

    protected Instances buildTansformedDataset(Instances data)
    {
        Instances output = determineOutputFormat(data);

        int dataSize = data.numInstances();
        // for each data, get distance to each shapelet and create new instance
        for (int i = 0; i < dataSize; i++)
        { // for each data
            Instance toAdd = new DenseInstance(shapelets.size() + 1);
            int shapeletNum = 0;
            for (Shapelet s : shapelets)
            {
                double dist = subsequenceDistance(s.content, getToDoubleArrayOfInstance(data, i));
                toAdd.setValue(shapeletNum++, dist);
            }
            toAdd.setValue(shapelets.size(), data.instance(i).classValue());
            output.add(toAdd);
        }
        return output;
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data)
    {
        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        classDistributions = getClassDistributions(data);                       // used to calc info gain

        
        System.out.println("k: " + numShapelets);
        
        //for all time series
        outputPrint("Processing data: ");

        int dataSize = data.numInstances();
        //for all possible time series.
        for (int i = 0; i < dataSize; i++)
        {
            outputPrint("data : " + i);

            double[] wholeCandidate = getToDoubleArrayOfInstance(data, i);

            //changed to pass in the worst of the K-Shapelets.
            Shapelet worstKShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;
            
            seriesShapelets = findShapeletCandidates(data, i, wholeCandidate, worstKShapelet);

            Comparator comp = useSeparationGap ? new Shapelet.ReverseSeparationGap() : new Shapelet.ReverseOrder();
            Collections.sort(seriesShapelets, comp);

            seriesShapelets = removeSelfSimilar(seriesShapelets);

            kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
        }

        this.numShapelets = kShapelets.size();

        recordShapelets(kShapelets);
        printShapelets(kShapelets);

        return kShapelets;
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param numShapelets
     * @param data the data that the shapelets will be taken from
     * @param minShapeletLength
     * @param maxShapeletLength
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapeletsCache(int numShapelets, Instances data, int minShapeletLength, int maxShapeletLength)
    {
        this.numShapelets = numShapelets;
        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
        return findBestKShapeletsCache(data);
    }

    protected ArrayList<Shapelet> findShapeletCandidates(Instances data, int i, double[] wholeCandidate, Shapelet worstKShapelet)
    {
        //get our time series as a double array.
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

        //for all possible lengths
        for (int length = minShapeletLength; length <= maxShapeletLength; length++)
        {
            double[] candidate = new double[length];
            //for all possible starting positions of that length
            for (int start = 0; start <= wholeCandidate.length - length - 1; start++)
            {
                //-1 = avoid classVal - handle later for series with no class val
                // CANDIDATE ESTABLISHED - got original series, length and starting position
                // extract relevant part into a double[] for processing
                System.arraycopy(wholeCandidate, start, candidate, 0, length);

                // znorm candidate here so it's only done once, rather than in each distance calculation
                candidate = zNorm(candidate, false);

                //Initialize bounding algorithm for current candidate
                QualityBound.ShapeletQualityBound qualityBound = initializeQualityBound(classDistributions);

                //Set bound of the bounding algorithm
                if (qualityBound != null && worstKShapelet != null)
                {
                    qualityBound.setBsfQuality(worstKShapelet.qualityValue);
                }

                //compare the shapelet candidate to the other time series.
                Shapelet candidateShapelet = checkCandidate(candidate, data, i, start, qualityBound);

                if (candidateShapelet != null)
                {
                    seriesShapelets.add(candidateShapelet);
                }
            }
        }
        return seriesShapelets;
    }

    /**
     * A method to obtain time taken to find a single best shapelet in the data
     * set
     *
     * @param data the data set to be processed
     * @param minShapeletLength minimum shapelet length
     * @param maxShapeletLength maximum shapelet length
     * @return time in seconds to find the best shapelet
     */
    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength)
    {
        data = roundRobinData(data, null);
        long startTime = System.nanoTime();
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double) (finishTime - startTime) / 1000000000.0;
    }

    protected void recordShapelets(ArrayList<Shapelet> kShapelets)
    {
        if (!this.recordShapelets)
        {
            return;
        }

        try
        {
            //just in case the file doesn't exist or the directories.
            File file = new File(this.ouputFileLocation);
            file.getParentFile().mkdirs();
            FileWriter out = new FileWriter(file);

            for (Shapelet kShapelet : kShapelets)
            {
                out.append(kShapelet.qualityValue + "," + kShapelet.seriesId + "," + kShapelet.startPos + "\n");
                double[] shapeletContent = kShapelet.content;
                for (int j = 0; j < shapeletContent.length; j++)
                {
                    out.append(shapeletContent[j] + ",");
                }
                out.append("\n");
            }
            out.close();
        }
        catch (IOException ex)
        {
            System.out.println("IOException: " + ex);
        }

    }

    protected void printShapelets(ArrayList<Shapelet> kShapelets)
    {
        if (supressOutput)
        {
            return;
        }

        System.out.println();
        System.out.println("Output Shapelets:");
        System.out.println("-------------------");
        System.out.println("informationGain,seriesId,startPos,classVal");
        System.out.println("<shapelet>");
        System.out.println("-------------------");
        System.out.println();
        for (Shapelet kShapelet : kShapelets)
        {
            System.out.println(kShapelet.qualityValue + "," + kShapelet.seriesId + "," + kShapelet.startPos + "," + kShapelet.classValue);
            double[] shapeletContent = kShapelet.content;
            for (int j = 0; j < shapeletContent.length; j++)
            {
                System.out.print(shapeletContent[j] + ",");
            }
            System.out.println();
        }

    }

    /**
     * Private method to combine two ArrayList collections of
     * FullShapeletTransform objects.
     *
     * @param k the maximum number of shapelets to be returned after combining
     * the two lists
     * @param kBestSoFar the (up to) k best shapelets that have been observed so
     * far, passed in to combine with shapelets from a new series
     * @param timeSeriesShapelets the shapelets taken from a new series that are
     * to be merged in descending order of fitness with the kBestSoFar
     * @return an ordered ArrayList of the best k (or less)
     * FullShapeletTransform objects from the union of the input ArrayLists
     */
    //NOTE: could be more efficient here
    protected ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets)
    {
        kBestSoFar.addAll(timeSeriesShapelets);

        Comparator comp = useSeparationGap ? new Shapelet.ReverseSeparationGap() : new Shapelet.ReverseOrder();
        Collections.sort(kBestSoFar, comp);

        if (kBestSoFar.size() < k)
        { // no need to return up to k, as there are not k shapelets yet
            return kBestSoFar;
        }

        ArrayList<Shapelet> newBestSoFar = new ArrayList<>();
        for (int i = 0; i < k; i++)
        {
            newBestSoFar.add(kBestSoFar.get(i));
        }

        return newBestSoFar;
    }

    /**
     *
     * @param classDist
     * @return
     */
    protected QualityBound.ShapeletQualityBound initializeQualityBound(Map<Double, Integer> classDist)
    {
        if (useCandidatePruning)
        {
            if (qualityMeasure instanceof QualityMeasures.InformationGain)
            {
                return new QualityBound.InformationGainBound(classDist, candidatePruningStartPercentage);
            }
            else if (qualityMeasure instanceof QualityMeasures.MoodsMedian)
            {
                return new QualityBound.MoodsMedianBound(classDist, candidatePruningStartPercentage);
            }
            else if (qualityMeasure instanceof QualityMeasures.FStat)
            {
                return new QualityBound.FStatBound(classDist, candidatePruningStartPercentage);
            }
            else if (qualityMeasure instanceof QualityMeasures.KruskalWallis)
            {
                return new QualityBound.KruskalWallisBound(classDist, candidatePruningStartPercentage);
            }
        }
        return null;
    }

    //this is the caching system. 
    protected double[] getToDoubleArrayOfInstance(Instances data, int pos)
    {
        if (!cacheDoubleArrays)
        {
            return data.get(pos).toDoubleArray();
        }

        if (cachedDoubleArray[pos] == null)
        {
            cachedDoubleArray[pos] = data.get(pos).toDoubleArray();
        }

        return cachedDoubleArray[pos];
    }

    /**
     * protected method to remove self-similar shapelets from an ArrayList (i.e.
     * if they come from the same series and have overlapping indicies)
     *
     * @param shapelets the input Shapelets to remove self similar
     * FullShapeletTransform objects from
     * @return a copy of the input ArrayList with self-similar shapelets removed
     */
    protected static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets)
    {
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<Shapelet> outputShapelets = new ArrayList<>();
        int size = shapelets.size();
        boolean[] selfSimilar = new boolean[size];

        for (int i = 0; i < size; i++)
        {
            if (selfSimilar[i])
            {
                continue;
            }

            outputShapelets.add(shapelets.get(i));

            for (int j = i + 1; j < size; j++)
            {
                // no point recalc'ing if already self similar to something
                if ((!selfSimilar[j]) && selfSimilarity(shapelets.get(i), shapelets.get(j)))
                {
                    selfSimilar[j] = true;
                }
            }
        }
        return outputShapelets;
    }

    /**
     * Private method to calculate the class distributions of a dataset. Main
     * purpose is for computing shapelet qualities.
     *
     * @param data the input data set that the class distributions are to be
     * derived from
     * @return a TreeMap<Double, Integer> in the form of <Class Value,
     * Frequency>
     */
    public static TreeMap<Double, Integer> getClassDistributions(Instances data)
    {
        TreeMap<Double, Integer> classDistribution = new TreeMap<>();

        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            classValue = it.next().classValue();

            Integer val = classDistribution.get(classValue);

            val = (val != null) ? val + 1 : 1;
            classDistribution.put(classValue, val);
        }
        
        System.out.println("class dists: " + classDistribution);
        return classDistribution;
    }

    /**
     * protected method to check a candidate shapelet. Functions by passing in
     * the raw data, and returning an assessed Shapelet object.
     *
     * @param candidate the data from the candidate FullShapeletTransform
     * @param data the entire data set to compare the candidate to
     * @param seriesId series id from the dataset that the candidate came from
     * @param startPos start position in the series where the candidate came
     * from
     * @param qualityBound
     * @return a fully-computed FullShapeletTransform, including the quality of
     * this candidate
     */
    protected Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, QualityBound.ShapeletQualityBound qualityBound)
    {

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();


        
        int dataSize = data.numInstances();

        for (int i = 0; i < dataSize; i++)
        {
            //Check if it is possible to prune the candidate
            if (qualityBound != null && qualityBound.pruneCandidate())
            {
                return null;
            }

            double distance = 0.0;
            //don't compare the shapelet to the the time series it came from.
            if (i != seriesId)
            {
                distance = subsequenceDistance(candidate, getToDoubleArrayOfInstance(data, i));
            }

            double classVal = data.instance(i).classValue();
            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            if (qualityBound != null)
            {
                qualityBound.updateOrderLine(orderline.get(orderline.size() - 1));
            }
        }

        // note: early abandon entropy pruning would appear here, but has been ommitted
        // in favour of a clear multi-class information gain calculation. Could be added in
        // this method in the future for speed up, but distance early abandon is more important


        // create a shapelet object to store all necessary info, i.e.
        Shapelet shapelet = new Shapelet(candidate, dataSourceIDs[seriesId], startPos, this.qualityMeasure);
        shapelet.calculateQuality(orderline, classDistributions); 
        shapelet.classValue =  data.instance(seriesId).classValue(); //set classValue of shapelet. (interesing to know).
        return shapelet;
    }

    public static double[] getInfoGain(Instances trans)
    {
        double[] quals = new double[trans.numAttributes() - 1];

        TreeMap map = getClassDistributions(trans);

        for (int i = 0; i < quals.length; i++)
        {
            ArrayList<OrderLineObj> orderline = new ArrayList<>();
            double[] dists = trans.attributeToDoubleArray(i);

            for (int j = 0; j < dists.length; j++)
            {

                double distance = dists[j];
                double classVal = trans.instance(j).classValue();
                orderline.add(new OrderLineObj(distance, classVal));

            }

            QualityMeasures.InformationGain ig = new QualityMeasures.InformationGain();
            double qual = ig.calculateQuality(orderline, map);
            quals[i] = qual;
        }

        return quals;
    }

    /**
     * Calculate the distance between a candidate series and an Instance object
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param timeSeriesIns an Instance object of a whole time series
     * @return the distance between a candidate and a time series
     *
     */
    protected double subseqDistance(double[] candidate, Instance timeSeriesIns)
    {
        return subsequenceDistance(candidate, timeSeriesIns.toDoubleArray());
    }

    /**
     *
     * @param candidate
     * @param timeSeriesIns
     * @return
     */
    public static double subsequenceDistance(double[] candidate, Instance timeSeriesIns)
    {
        return subsequenceDistance(candidate, timeSeriesIns.toDoubleArray());
    }

    /**
     * Calculate the distance between a shapelet candidate and a full time
     * series (both double[]).
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param timeSeries a double[] representation of a whole time series (inc.
     * class value)
     * @return the distance between a candidate and a time series
     */
    public static double subsequenceDistance(double[] candidate, double[] timeSeries)
    {

        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;

        // for all possible subsequences of two
        for (int i = 0; i < timeSeries.length - candidate.length; i++)
        {
            sum = 0;
            // get subsequence of two that is the same lengh as one
            subseq = new double[candidate.length];
            System.arraycopy(timeSeries, i, subseq, 0, candidate.length);

            subseqDistOpCount += candidate.length;

            subseq = zNormalise(subseq, false); // Z-NORM HERE

            //Keep count of fundamental ops for experiment
            subseqDistOpCount += 3 * subseq.length;

            for (int j = 0; j < candidate.length; j++)
            {
                temp = (candidate[j] - subseq[j]);
                sum += temp * temp;
            }

            subseqDistOpCount += candidate.length;

            if (sum < bestSum)
            {
                bestSum = sum;
            }
        }
        return (bestSum == 0.0) ? 0.0 : (1.0 / candidate.length * bestSum);
    }

    /**
     *
     * @param input
     * @param classValOn
     * @return
     */
    protected double[] zNorm(double[] input, boolean classValOn)
    {
        return FullShapeletTransform.zNormalise(input, classValOn);
    }

    /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value
     * (e.g. an full instance might, a candidate shapelet wouldn't)
     * @return a z-normalised version of input
     */
    public static double[] zNormalise(double[] input, boolean classValOn)
    {
        double mean;
        double stdv;

        int classValPenalty = classValOn ? 1 : 0;
        int inputLength = input.length - classValPenalty;

        double[] output = new double[input.length];
        double seriesTotal = 0;

        for (int i = 0; i < inputLength; i++)
        {
            seriesTotal += input[i];
        }

        mean = seriesTotal / (double) inputLength;
        stdv = 0;
        double temp;
        for (int i = 0; i < inputLength; i++)
        {
            temp = (input[i] - mean);
            stdv += temp * temp;
        }

        stdv /= (double) inputLength;

        // if the variance is less than the error correction, just set it to 0, else calc stdv.
        stdv = (stdv < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv);

        for (int i = 0; i < inputLength; i++)
        {
            //if the stdv is 0 then set to 0, else normalise.
            output[i] = (stdv == 0.0) ? 0.0 : ((input[i] - mean) / stdv);
        }

        if (classValOn)
        {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    /**
     * Load a set of Instances from an ARFF
     *
     * @param fileName the file name of the ARFF
     * @return a set of Instances from the ARFF
     */
    public static Instances loadData(String fileName)
    {
        Instances data = null;
        try
        {
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        }
        catch (IOException e)
        {
            System.out.println(" Error =" + e + " in method loadData");
        }
        return data;
    }

    /**
     * A private method to assess the self similarity of two
     * FullShapeletTransform objects (i.e. whether they have overlapping
     * indicies and are taken from the same time series).
     *
     * @param shapelet the first FullShapeletTransform object (in practice, this
     * will be the dominant shapelet with quality >= candidate)
     * @param candidate the second FullShapeletTransform
     * @return
     */
    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate)
    {
        if (candidate.seriesId == shapelet.seriesId)
        {
            if (candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.content.length)
            { //candidate starts within exisiting shapelet
                return true;
            }
            if (shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.content.length)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * A method to read in a FullShapeletTransform log file to reproduce a
     * FullShapeletTransform
     * <p>
     * NOTE: assumes shapelets from log are Z-NORMALISED
     *
     * @param fileName the name and path of the log file
     * @return a duplicate FullShapeletTransform to the object that created the
     * original log file
     * @throws Exception
     */
    public static FullShapeletTransform createFilterFromFile(String fileName) throws Exception
    {
        return createFilterFromFile(fileName, Integer.MAX_VALUE);
    }

    /**
     * Returns a list of the lengths of the shapelets found by this transform.
     *
     * @return An ArrayList of Integers representing the lengths of the
     * shapelets.
     */
    public ArrayList<Integer> getShapeletLengths()
    {
        ArrayList<Integer> shapeletLengths = new ArrayList<>();

        if (this.shapeletsTrained)
        {
            for (Shapelet s : this.shapelets)
            {
                shapeletLengths.add(s.content.length);
            }
        }

        return shapeletLengths;
    }

    /**
     * A method to read in a FullShapeletTransform log file to reproduce a
     * FullShapeletTransform,
     * <p>
     * NOTE: assumes shapelets from log are Z-NORMALISED
     *
     * @param fileName the name and path of the log file
     * @param maxShapelets
     * @return a duplicate FullShapeletTransform to the object that created the
     * original log file
     * @throws Exception
     */
    public static FullShapeletTransform createFilterFromFile(String fileName, int maxShapelets) throws Exception
    {

        File input = new File(fileName);
        Scanner scan = new Scanner(input);
        scan.useDelimiter("\n");

        FullShapeletTransform sf = new FullShapeletTransform();
        ArrayList<Shapelet> shapelets = new ArrayList<>();

        String shapeletContentString;
        String shapeletStatsString;
        ArrayList<Double> content;
        double[] contentArray;
        Scanner lineScan;
        Scanner statScan;
        double qualVal;
        int serID;
        int starPos;

        int shapeletCount = 0;

        while (shapeletCount < maxShapelets && scan.hasNext())
        {
            shapeletStatsString = scan.next();
            shapeletContentString = scan.next();

            //Get the shapelet stats
            statScan = new Scanner(shapeletStatsString);
            statScan.useDelimiter(",");

            qualVal = Double.parseDouble(statScan.next().trim());
            serID = Integer.parseInt(statScan.next().trim());
            starPos = Integer.parseInt(statScan.next().trim());
            //End of shapelet stats

            lineScan = new Scanner(shapeletContentString);
//            System.out.println(shapeletContentString);
            lineScan.useDelimiter(",");

            content = new ArrayList<>();
            while (lineScan.hasNext())
            {
                String next = lineScan.next().trim();
                if (!next.isEmpty())
                {
                    content.add(Double.parseDouble(next));
                }
            }

            contentArray = new double[content.size()];
            for (int i = 0; i < content.size(); i++)
            {
                contentArray[i] = content.get(i);
            }

            contentArray = zNormalise(contentArray, false);

            Shapelet s = new Shapelet(contentArray, qualVal, serID, starPos);

            shapelets.add(s);
            shapeletCount++;
        }
        sf.shapelets = shapelets;
        sf.shapeletsTrained = true;
        sf.numShapelets = shapelets.size();
        sf.setShapeletMinAndMax(1, 1);

        return sf;
    }

    /**
     * Outputs the log file to the appropriate location.
     *
     * @throws Exception
     */
    public void outputLog() throws Exception
    {
        //just in case the file doesn't exist, or the directories.
        File file = new File(this.ouputFileLocation);
        file.getParentFile().mkdirs();

        FileWriter out = new FileWriter(this.ouputFileLocation, file.exists());
        for (Shapelet shapelet : this.shapelets)
        {
            out.append(shapelet.qualityValue + "," + shapelet.seriesId + "," + shapelet.startPos + "\n");
            double[] shapeletContent = shapelet.content;
            for (int j = 0; j < shapeletContent.length; j++)
            {
                out.append(shapeletContent[j] + ",");
            }
            out.append("\n");
        }
        out.close();

    }

    /**
     * Method to reset shapelet indices into the values given in sourcePos
     *
     * @param shapelets
     * @param sourcePos Pointer to array of ints, where old positions of
     * instances are to be stored.
     */
    public static void resetShapeletIndices(ArrayList<Shapelet> shapelets, int[] sourcePos)
    {
        for (Shapelet s : shapelets)
        {
            int pos = s.getSeriesId();
            s.setSeriesID(sourcePos[pos]);
        }
    }

    /**
     * Method to reorder the given Instances into the order given in sourcePos
     *
     * @param data Instances to be reordered
     * @param sourcePos Pointer to array of ints, where old positions of
     * instances are to be stored.
     */
    public static void resetDataOrder(Instances data, int[] sourcePos)
    {
        int dataSize = data.numInstances();
        if (dataSize != sourcePos.length)
        {//ERROR
            System.out.println(" ERROR, cannot reorder, because the series are different lengths");
            return;
        }
        Instance[] newOrder = new Instance[sourcePos.length];
        for (int i = 0; i < sourcePos.length; i++)
        {
            newOrder[sourcePos[i]] = data.instance(i);
        }
        for (int i = 0; i < dataSize; i++)
        {
            data.set(i, newOrder[i]);
        }

    }

    /**
     * Method to reorder the given Instances in round robin order
     *
     * @param data Instances to be reordered
     * @param sourcePos Pointer to array of ints, where old positions of
     * instances are to be stored.
     * @return Instances in round robin order
     */
    public static Instances roundRobinData(Instances data, int[] sourcePos)
    {

        //Count number of classes 
        TreeMap<Double, ArrayList<Instance>> instancesByClass = new TreeMap<>();
        TreeMap<Double, ArrayList<Integer>> positionsByClass = new TreeMap<>();

        //Get class distributions 
        TreeMap<Double, Integer> classDistribution = FullShapeletTransform.getClassDistributions(data);

        //Allocate arrays for instances of every class
        for (Double key : classDistribution.keySet())
        {
            int frequency = classDistribution.get(key);
            instancesByClass.put(key, new ArrayList<Instance>(frequency));
            positionsByClass.put(key, new ArrayList<Integer>(frequency));
        }

        int dataSize = data.numInstances();
        //Split data according to their class memebership
        for (int i = 0; i < dataSize; i++)
        {
            Instance inst = data.instance(i);
            instancesByClass.get(inst.classValue()).add(inst);
            positionsByClass.get(inst.classValue()).add(i);
        }

        //Merge data into single list in round robin order
        Instances roundRobinData = new Instances(data, dataSize);
        for (int i = 0; i < dataSize;)
        {
            //Allocate arrays for instances of every class
            for (Double key : classDistribution.keySet())
            {
                ArrayList<Instance> currentList = instancesByClass.get(key);
                ArrayList<Integer> currentPositions = positionsByClass.get(key);

                if (!currentList.isEmpty())
                {
                    roundRobinData.add(currentList.remove(currentList.size() - 1));
                    if (sourcePos != null && sourcePos.length == dataSize)
                    {
                        sourcePos[i] = currentPositions.remove(currentPositions.size() - 1);
                    }
                    i++;
                }
            }
        }

        return roundRobinData;
    }

    public void outputPrint(String val)
    {
        if (!this.supressOutput)
        {
            System.out.println(val);
        }
    }

    @Override
    public String toString()
    {
        String str = "Shapelets: ";
        for (Shapelet s : shapelets)
        {
            str += s.toString() + "\n";
        }
        return str;
    }

    /**
     *
     * @param data
     * @param minShapeletLength
     * @param maxShapeletLength
     * @return
     * @throws Exception
     */
    public long opCountForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) throws Exception
    {
        data = roundRobinData(data, null);
        subseqDistOpCount = 0;
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        return subseqDistOpCount;
    }

    /**
     * An example use of a FullShapeletTransform
     *
     * @param args command line args. arg[0] should spcify a set of training
     * instances to transform
     */
    public static void main(String[] args)
    {
        try
        {
            // mandatory requirements:  numShapelets (k), min shapelet length, max shapelet length, input data
            // additional information:  log output dir

            // example filter, k = 10, minLength = 20, maxLength = 40, data = , output = exampleOutput.txt
            int k = 10;
            int minLength = 10;
            int maxLength = 20;
//            Instances data= FullShapeletTransform2.loadData("ItalyPowerDemand_TRAIN.arff"); // for example
            Instances data = FullShapeletTransform.loadData(args[0]);

            FullShapeletTransform sf = new FullShapeletTransform(k, minLength, maxLength);
            sf.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
            sf.setLogOutputFile("exampleOutput.txt"); // log file stores shapelet output

            // Note: sf.process returns a transformed set of Instances. The first time that
            //      thisFilter.process(data) is called, shapelet extraction occurs. Subsequent calls to process
            //      uses the previously extracted shapelets to transform the data. For example:
            //
            //      Instances transformedTrain = sf.process(trainingData); -> extracts shapelets and can be used to transform training data
            //      Instances transformedTest = sf.process(testData); -> uses shapelets extracted from trainingData to transform testData
            Instances transformed = sf.process(data);
        }
        catch (IllegalArgumentException e)
        {
            System.out.println("IllegalArgumentException: "+ e);
        }
    }

}
