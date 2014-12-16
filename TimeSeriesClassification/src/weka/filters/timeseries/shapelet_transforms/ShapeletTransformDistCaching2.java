package weka.filters.timeseries.shapelet_transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.TreeMap;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.QualityMeasures.ShapeletQualityChoice;
import weka.core.shapelet.Shapelet;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.getClassDistributions;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.removeSelfSimilar;

/**
 * An optimised filter to transform a dataset by k shapelets.
 *
 * @author Edgaras Baranauskas
 */
public class ShapeletTransformDistCaching2 extends FullShapeletTransform
{

    protected Stats stats;
    protected double[][] data;

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletTransformDistCaching2()
    {
        super();
        stats = null;
        data = null;
    }

    /**
     * Single param constructor: filter is unusable until min/max params are
     * initialised. Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     */
    public ShapeletTransformDistCaching2(int k)
    {
        super(k);
        stats = null;
        data = null;
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletTransformDistCaching2(int k, int minShapeletLength, int maxShapeletLength)
    {
        super(k, minShapeletLength, maxShapeletLength);
        stats = null;
        data = null;
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
    public ShapeletTransformDistCaching2(int k, int minShapeletLength, int maxShapeletLength, ShapeletQualityChoice qualityChoice)
    {
        super(k, minShapeletLength, maxShapeletLength, qualityChoice);
        stats = null;
        data = null;
    }
    
    @Override
    public Instances process(Instances dataInst) throws IllegalArgumentException
    {
        //check the input data is correct and assess whether the filter has been setup correctly.
        inputCheck(dataInst);

        //instantiate the caching array here, so it gets refreshed if we're using a test set.
        if(cacheDoubleArrays)
            cachedDoubleArray = new double[dataInst.numInstances()][];
        
        //checks if the shapelets haven't been found yet, finds them if it needs too.
        if (!shapeletsTrained)
        {
           trainShapelets(dataInst);
        }
        else
        {
            stats = null;
            data = null;
        }

        //build the transformed dataset with the shapelets we've found either on this data, or the previous training data
        return buildTansformedDataset(dataInst);
    }
    
    @Override
    protected Instances buildTansformedDataset(Instances dataInst)
    {
        Instances output = determineOutputFormat(dataInst);

        if (data == null)
        {
            initialiseNormalisedData(dataInst);
        }

        Shapelet s;
        
        //calculate the distances from the shapelets to the data.
        int size = shapelets.size();
        int dataSize = dataInst.numInstances();
        double dist;
        
        //create our data instances
        for(int j = 0; j < dataSize; j++)
        {
            output.add(new DenseInstance(size + 1));
        }
        
        //fill the distances in.
        for (int i = 0; i < size; i++)
        {
            s = shapelets.get(i);
            if (data != null && stats != null)
            {
                stats.computeStats(s.getSeriesId(), data);
            }

            for (int j = 0; j < dataSize; j++)
            {
                if (data != null && stats != null)
                {
                    stats.setCurrentY(j);
                    dist = cachedSubsequenceDistance(s.getStartPos(), s.getContent().length, data[j].length, stats);
                }
                else
                {
                    dist = subseqDistance(s.getContent(), dataInst.instance(j));
                }

                output.instance(j).setValue(i, dist);
            }
        }
            
        //do the classValues.
        for(int j=0; j < dataSize; j++)
        {
            output.instance(j).setValue(size, dataInst.instance(j).classValue());
        }
        
        return output;
    }
    
    protected void initialiseNormalisedData(Instances dataInst)
    {
        stats = new Stats();
        //Normalise all time series for further processing
        int dataSize = dataInst.numInstances();
        
        data = new double[dataSize][];
        for (int i = 0; i < dataSize; i++)
        {
            data[i] = FullShapeletTransform.zNormalise(getToDoubleArrayOfInstance(dataInst, i), true);
        }
    }

    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances dataInst)
    {
        //initliase our normalised data
        initialiseNormalisedData(dataInst);
        int dataSize = dataInst.numInstances();
        
        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        classDistributions = getClassDistributions(dataInst);                       // used to calc info gain

        //for all time series
        outputPrint("Processing data: ");

        //for all possible time series.
        for (int i = 0; i < dataSize; i++)
        {
            outputPrint("data : " + i);
            
            stats.computeStats(i, data);
            
            seriesShapelets = findShapeletCandidates(dataInst, i, data[i], kShapelets);
            
            Comparator comp = useSeparationGap ? new Shapelet.ReverseSeparationGap(): new Shapelet.ReverseOrder();
            Collections.sort(seriesShapelets,comp);
            
            seriesShapelets = removeSelfSimilar(seriesShapelets);
            
            kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
        }

        this.numShapelets = kShapelets.size();

        recordShapelets(kShapelets);
        printShapelets(kShapelets);

        return kShapelets;
    }

    @Override
    protected Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, QualityBound.ShapeletQualityBound qualityBound)
    {
        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        boolean pruned = false;

        int dataSize = data.numInstances();
        for (int i = 0; i < dataSize; i++)
        {
            //Check if it is possible to prune the candidate
            if (qualityBound != null && qualityBound.pruneCandidate())
            {
                pruned = true;
                break;
            }

            double distance = 0.0;
            if (i != seriesId)
            {
                stats.setCurrentY(i);
                distance = cachedSubsequenceDistance(startPos, candidate.length, data.instance(i).numAttributes(), stats);
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
        //If shapelet is pruned then it should no longer be considered in further processing
        if (!pruned)
        {
            // create a shapelet object to store all necessary info, i.e.
            Shapelet shapelet = new Shapelet(candidate, dataSourceIDs[seriesId], startPos, qualityMeasure);
            shapelet.calculateQuality(orderline, classDistributions);
            return shapelet;
        }
        
        return null;
    }

    /**
     * Calculate the distance between a shapelet candidate and a full time
     * series (both double[]).
     *
     * @param startPos start position of the candidate in the whole candidate
     * series
     * @param subLength candidate length
     * @param seriesLength series length
     * @param stats Stats object containing statistics computed for candidate
     * and all time series
     * @return the distance beween a candidate and a time series
     */
    public static double cachedSubsequenceDistance(int startPos, int subLength, int seriesLength, Stats stats)
    {
        double minSum = Double.MAX_VALUE;
        double xMean = stats.getMeanX(startPos, subLength);
        double xStdDev = stats.getStdDevX(startPos, subLength);
        double yMean;
        double yStdDev;
        double crossProd;

        // Scan through all possible subsequences of two
        for (int v = 0; v < seriesLength - subLength; v++)
        {
            yMean = stats.getMeanY(v, subLength);
            yStdDev = stats.getStdDevY(v, subLength);
            crossProd = stats.getSumOfProds(startPos, v, subLength);

            double cXY = 0.0;
            if (xStdDev != 0 && yStdDev != 0)
            {
                cXY = (crossProd - (subLength * xMean * yMean)) / (subLength * xStdDev * yStdDev);
            }

            double dist = 2 * (1 - cXY);

            if (dist < minSum)
            {
                minSum = dist;
            }
        }

        return minSum;
    }

    /**
     * A class for holding relevant statistics for any given candidate series
     * and all time series TO DO: CONVERT IT ALL TO FLOATS
     * Aaron: Changed to floats, why?
     */
    public static class Stats
    {

        private float[][] cummSums;
        private float[][] cummSqSums;
        private float[][][] crossProds;
        private int xIndex;
        private int yIndex;

        /**
         * Default constructor
         */
        public Stats()
        {
            cummSums = null;
            cummSqSums = null;
            crossProds = null;
            xIndex = -1;
            yIndex = -1;
        }

        /**
         * A method to retrieve cumulative sums for all time series processed so
         * far
         *
         * @return cumulative sums
         */
        public float[][] getCummSums()
        {
            return cummSums;
        }

        /**
         * A method to retrieve cumulative square sums for all time series
         * processed so far
         *
         * @return cumulative square sums
         */
        public float[][] getCummSqSums()
        {
            return cummSqSums;
        }

        /**
         * A method to retrieve cross products for candidate series. The cross
         * products are computed between candidate and all time series.
         *
         * @return cross products
         */
        public float[][][] getCrossProds()
        {
            return crossProds;
        }

        /**
         * A method to set current time series that is being examined.
         *
         * @param yIndex time series index
         */
        public void setCurrentY(int yIndex)
        {
            this.yIndex = yIndex;
        }

        /**
         * A method to retrieve the mean value of a whole candidate sub-series.
         *
         * @param startPos start position of the candidate
         * @param subLength length of the candidate
         * @return mean value of sub-series
         */
        public float getMeanX(int startPos, int subLength)
        {
            return (cummSums[xIndex][startPos + subLength] - cummSums[xIndex][startPos]) / subLength;
        }

        /**
         * A method to retrieve the mean value for time series sub-series. Note
         * that the current Y must be set prior invoking this method.
         *
         * @param startPos start position of the sub-series
         * @param subLength length of the sub-series
         * @return mean value of sub-series
         */
        public float getMeanY(int startPos, int subLength)
        {
            return (cummSums[yIndex][startPos + subLength] - cummSums[yIndex][startPos]) / subLength;
        }

        /**
         * A method to retrieve the standard deviation of a whole candidate
         * sub-series.
         *
         * @param startPos start position of the candidate
         * @param subLength length of the candidate
         * @return standard deviation of the candidate sub-series
         */
        public float getStdDevX(int startPos, int subLength)
        {
            return (float) Math.sqrt(((cummSqSums[xIndex][startPos + subLength] - cummSqSums[xIndex][startPos]) / subLength) - (getMeanX(startPos, subLength) * getMeanX(startPos, subLength)));
        }

        /**
         * A method to retrieve the standard deviation for time series
         * sub-series. Note that the current Y must be set prior invoking this
         * method.
         *
         * @param startPos start position of the sub-series
         * @param subLength length of the sub-series
         * @return standard deviation of sub-series
         */
        public float getStdDevY(int startPos, int subLength)
        {
            return (float) Math.sqrt(((cummSqSums[yIndex][startPos + subLength] - cummSqSums[yIndex][startPos]) / subLength) - (getMeanY(startPos, subLength) * getMeanY(startPos, subLength)));
        }

        /**
         * A method to retrieve the cross product of whole candidate sub-series
         * and time series sub-series. Note that the current Y must be set prior
         * invoking this method.
         *
         * @param startX start position of the whole candidate sub-series
         * @param startY start position of the time series sub-series
         * @param length length of the both sub-series
         * @return sum of products for a given overlap between two sub=series
         */
        public float getSumOfProds(int startX, int startY, int length)
        {
            return crossProds[yIndex][startX + length][startY + length] - crossProds[yIndex][startX][startY];
        }

        private float[][] computeCummSums(double[] currentSeries)
        {

            float[][] output = new float[2][];
            output[0] = new float[currentSeries.length];
            output[1] = new float[currentSeries.length];
            output[0][0] = 0;
            output[1][0] = 0;

            //Compute stats for a given series instance
            for (int i = 1; i < currentSeries.length; i++)
            {
                output[0][i] = (float) (output[0][i - 1] + currentSeries[i - 1]);                         //Sum of vals
                output[1][i] = (float) (output[1][i - 1] + (currentSeries[i - 1] * currentSeries[i - 1]));  //Sum of squared vals
            }

            return output;
        }

        private float[][] computeCrossProd(double[] x, double[] y)
        {

            float[][] output = new float[x.length][y.length];

            for (int u = 1; u < x.length; u++)
            {
                for (int v = 1; v < y.length; v++)
                {
                    int t;  //abs(u-v)
                    if (v < u)
                    {
                        t = u - v;
                        output[u][v] = (float) (output[u - 1][v - 1] + (x[v - 1 + t] * y[v - 1]));
                    }
                    else
                    {//else v >= u
                        t = v - u;
                        output[u][v] = (float) (output[u - 1][v - 1] + (x[u - 1] * y[u - 1 + t]));
                    }
                }
            }

            return output;
        }

        /**
         * A method to compute statistics for a given candidate series index and
         * normalised time series
         *
         * @param candidateInstIndex index of the candidate within the time
         * series database
         * @param data the normalised database of time series
         */
        public void computeStats(int candidateInstIndex, double[][] data)
        {

            xIndex = candidateInstIndex;

            //Initialise stats caching arrays
            if (cummSums == null || cummSqSums == null)
            {
                cummSums = new float[data.length][];
                cummSqSums = new float[data.length][];
            }

            crossProds = new float[data.length][][];

            //Process all instances
            for (int i = 0; i < data.length; i++)
            {

                //Check if cummulative sums are already stored for corresponding instance
                if (cummSums[i] == null || cummSqSums[i] == null)
                {
                    float[][] sums = computeCummSums(data[i]);
                    cummSums[i] = sums[0];
                    cummSqSums[i] = sums[1];
                }

                //Compute cross products between candidate series and current series
                crossProds[i] = computeCrossProd(data[candidateInstIndex], data[i]);
            }
        }
    }


    public static void main(String[] args)
    {
        //################ Test 1 ################
        System.out.println("1) Testing cached subsequence distance: ");

        //Create some time series for testing
        System.out.println("\n1.1) Example series: ");
        int seriesLength = 11;
        int numOfSeries = 5;
        double[][] data = new double[numOfSeries][seriesLength];

        int min = -5;
        int max = 5;
        for (int i = 0; i < numOfSeries; i++)
        {
            for (int j = 0; j < seriesLength; j++)
            {
                if (j == seriesLength - 1)
                {
                    data[i][j] = 0;
                }
                else
                {
                    data[i][j] = min + (int) (Math.random() * ((max - min) + 1));
                }
            }
            ShapeletTransform.printSeries(data[i]);
        }

        //Normalise test time series
        System.out.println("\n1.2) Normalised example series: ");
        for (int i = 0; i < numOfSeries; i++)
        {
            data[i] = FullShapeletTransform.zNormalise(data[i], true);
            ShapeletTransform.printSeries(data[i]);
        }

        double total = 0.0;
        for (int i = 0; i < numOfSeries; i++)
        {
            for (int j = 0; j < seriesLength; j++)
            {
                total += data[i][j];
            }

            System.out.println("sum for series " + i + ": " + total);
            total = 0.0;
        }

        seriesLength--;

        //Create stats object
        System.out.println("\n1.3) Unequal distances: ");
        Stats stats = new Stats();

        int minShapeletLength = seriesLength / 2;
        int maxShapeletLength = seriesLength / 2;

        //Every time series instance
        for (int i = 0; i < numOfSeries; i++)
        {
            //Compute statistics for the candidate series and every instance
            stats.computeStats(i, data);

            //Every possible lengh
            for (int length = minShapeletLength; length <= maxShapeletLength; length++)
            {

                //for all possible starting positions of that length
                for (int start = 0; start <= data[i].length - length - 1; start++)
                { //-1 = avoid classVal - handle later for series with no class val
                    // CANDIDATE ESTABLISHED - got original series, length and starting position
                    // extract relevant part into a double[] for processing
                    double[] candidate = new double[length];
                    for (int m = start; m < start + length; m++)
                    {
                        candidate[m - start] = data[i][m];
                    }

                    //Check individual components for completeness.
                    //System.out.println("MEAN: " + computeMean(candidate, false) + " = " + stats.getMeanX(start, length));
                    //System.out.println("STDV: " + computeStdv(candidate, false) + " = " + stats.getStdDevX(start, length));
                    //Compute distance for each candidate
                    for (int j = 0; j < numOfSeries; j++)
                    {
                        stats.setCurrentY(j);

                        double distanceCached = cachedSubsequenceDistance(start, candidate.length, data[j].length, stats);
                        double distanceOriginal = FullShapeletTransform.subsequenceDistance(FullShapeletTransform.zNormalise(candidate, false), data[j]);
                        if (Math.abs(distanceCached - distanceOriginal) > 0.0000000000000000001)
                        {
                            System.out.println("Candidate = " + i + ", startPos = " + start + ", series = " + j + ":\t" + distanceCached + " = " + distanceOriginal);
                        }
                    }
                }
            }
        }
    }

    //Used for testing
    private static double computeMean(double[] input, boolean classValOn)
    {
        double mean;
        double classValPenalty = classValOn ? 1.0 : 0.0;
        double seriesTotal = 0;

        for (int i = 0; i < input.length - classValPenalty; i++)
        {
            seriesTotal += input[i];
        }
        mean = seriesTotal / (input.length - classValPenalty);

        return mean;
    }

    //Used for testing
    private static double computeStdv(double[] input, boolean classValOn)
    {
        double mean = computeMean(input, classValOn);
        double stdv;

        double classValPenalty = classValOn ? 1.0 : 0.0;
        double seriesSquareTotal = 0;

        for (int i = 0; i < input.length - classValPenalty; i++)
        {
            seriesSquareTotal += (input[i] * input[i]);
        }
        stdv = Math.sqrt((seriesSquareTotal - (mean * mean * (input.length - classValPenalty))) / (input.length - classValPenalty));
        return stdv;
    }
}
