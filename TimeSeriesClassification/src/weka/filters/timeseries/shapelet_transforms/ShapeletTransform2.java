package weka.filters.timeseries.shapelet_transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;

/**
 * An optimised filter to transform a dataset by k shapelets.
 *
 * This method uses the distance calculation early abandons described in the
 * "Trillions" KDD paper from Eamonn's group
 *
 * @author Edgaras Baranauskas
 */
public class ShapeletTransform2 extends FullShapeletTransform2
{

    private static long subseqDistOpCount;

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletTransform2()
    {
        super();
    }

    /**
     * Single param constructor: filter is unusable until min/max params are
     * initialised. Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     */
    public ShapeletTransform2(int k)
    {
        super(k);
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletTransform2(int k, int minShapeletLength, int maxShapeletLength)
    {
        super(k, minShapeletLength, maxShapeletLength);
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
    public ShapeletTransform2(int k, int minShapeletLength, int maxShapeletLength, QualityMeasures.ShapeletQualityChoice qualityChoice)
    {
        super(k, minShapeletLength, maxShapeletLength, qualityChoice);
    }

    
    @Override
    protected Instances buildTansformedDataset(Instances data)
    {
        //Reorder the training data and reset the shapelet indexes
        Instances output = determineOutputFormat(data);

        Shapelet s;
        double[][] sortedIndexes;
        // for each data, get distance to each shapelet and create new instance
        int size = shapelets.size();
        int dataSize = data.numInstances();
        
        //create our data instances
        for(int j = 0; j < dataSize; j++)
        {
            output.add(new DenseInstance(size + 1));
        }
        
        double dist;
        for (int i = 0; i < size; i++)
        {
            s = shapelets.get(i);
            sortedIndexes = sortIndexes(s.content);

            for (int j = 0; j < dataSize; j++)
            {
                dist = onlineSubsequenceDistance(s.content, sortedIndexes, data.instance(j));
                output.instance(j).setValue(i, dist);
            }
        }
        
        //do the classValues.
        for(int j=0; j < dataSize; j++)
        {
            output.instance(j).setValue(size, data.instance(j).classValue());
        }

        return output;
    }

    @Override
    protected Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, QualityBound.ShapeletQualityBound qualityBound)
    {
        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        boolean pruned = false;
        double[][] sortedIndexes = sortIndexes(candidate);

        int dataSize = data.numInstances();
        for (int i = 0; i < dataSize; i++)
        {
            //Check if it is possible to prune the candidate
            if (qualityBound != null)
            {
                if (qualityBound.pruneCandidate())
                {
                    pruned = true;
                    break;
                }
            }

            double distance = 0.0;
            if (i != seriesId)
            {
                distance = onlineSubsequenceDistance(candidate, sortedIndexes, data.instance(i));
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

    @Override
    protected double[] zNorm(double[] input, boolean classValOn)
    {
        return optimizedZNormalise(input, classValOn);
    }

    /**
     * Calculate the distance between a candidate series and an Instance object
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param sortedIndices
     * @param timeSeriesIns an Instance object of a whole time series
     * @return the distance between a candidate and a time series
     */
    public static double onlineSubsequenceDistance(double[] candidate, double[][] sortedIndices, Instance timeSeriesIns)
    {
        double[] timeSeries = timeSeriesIns.toDoubleArray();
        return onlineSubsequenceDistance(candidate, sortedIndices, timeSeries);
    }

    /**
     * Calculate the distance between a shapelet candidate and a full time
     * series (both double[]).
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param sortedIndices
     * @param timeSeries a double[] representation of a whole time series (inc.
     * class value)
     * @return the distance between a candidate and a time series
     *
     *
     * NOTE: it seems that the reordering is repeated for each new time series.
     * This could be avoided, but not sure how to structure the code to do it
     */
    public static double onlineSubsequenceDistance(double[] candidate, double[][] sortedIndices, double[] timeSeries)
    {

        DoubleWrapper sumPointer = new DoubleWrapper();
        DoubleWrapper sum2Pointer = new DoubleWrapper();

        //Generate initial subsequence 
        double[] subseq = new double[candidate.length];
        System.arraycopy(timeSeries, 0, subseq, 0, subseq.length);
        subseq = optimizedZNormalise(subseq, false, sumPointer, sum2Pointer);

        //Keep count of fundamental ops for experiment
        subseqDistOpCount += subseq.length;

        double sum = sumPointer.get();
        double sum2 = sum2Pointer.get();

        double bestDist = 0.0;

        double mean;
        double stdv;

        double temp;
        //Compute initial distance
        for (int i = 0; i < candidate.length; i++)
        {
            temp = candidate[i] - subseq[i];
            bestDist += temp * temp;
        }
        
        //Keep count of fundamental ops for experiment
        subseqDistOpCount+= candidate.length;

        // Scan through all possible subsequences of two
        for (int i = 1; i < timeSeries.length - candidate.length; i++)
        {
            //Update the running sums
            sum = sum - timeSeries[i - 1] + timeSeries[i - 1 + candidate.length];
            sum2 = sum2 - (timeSeries[i - 1] * timeSeries[i - 1]) + (timeSeries[i - 1 + candidate.length] * timeSeries[i - 1 + candidate.length]);

            //Compute the stats for new series
            mean = sum / candidate.length;

            //Get rid of rounding errors
            double stdv2 = (sum2 - (mean * mean * candidate.length)) / candidate.length;
            
            stdv = (stdv2 < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv2);

            int j = 0;
            double currentDist = 0.0;
            double toAdd;
            int reordedIndex;
            while (j < candidate.length && currentDist < bestDist)
            {
                reordedIndex = (int) sortedIndices[j][0];
                toAdd = candidate[reordedIndex] - (stdv == 0.0 ? 0.0 : ((timeSeries[i + reordedIndex] - mean) / stdv));
                currentDist += (toAdd * toAdd);
                j++;

                //Keep count of fundamental ops for experiment
                subseqDistOpCount++;
            }

            if (currentDist < bestDist)
            {
                bestDist = currentDist;
            }
        }

        return (bestDist == 0.0) ? 0.0 : (1.0 / candidate.length * bestDist);
    }

    /**
     * A method to sort the array indeces according to their corresponding
     * values
     *
     * @param series a time series, which indeces need to be sorted
     * @return
     */
    public static double[][] sortIndexes(double[] series)
    {
        //Create an boxed array of values with corresponding indexes
        double[][] sortedSeries = new double[series.length][2];
        for (int i = 0; i < series.length; i++)
        {
            sortedSeries[i][0] = i;
            sortedSeries[i][1] = Math.abs(series[i]);
        }

        Arrays.sort(sortedSeries, new Comparator<double[]>()
        {
            @Override
            public int compare(double[] o1, double[] o2)
            {
                return Double.compare(o1[1], o2[1]);
            }
        });

        return sortedSeries;
    }

    /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value
     * (e.g. an full instance might, a candidate shapelet wouldn't)
     * @return a z-normalised version of input
     */
    public static double[] optimizedZNormalise(double[] input, boolean classValOn)
    {
        return optimizedZNormalise(input, classValOn, null, null);
    }

    /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value
     * (e.g. an full instance might, a candidate shapelet wouldn't)
     * @param storeGlobally specify whether the sum and sum of squares should be
     * stored globally - this is used in subsequence distance method
     * @return a z-normalised version of input
     */
    private static double[] optimizedZNormalise(double[] input, boolean classValOn, DoubleWrapper sum, DoubleWrapper sum2)
    {
        double mean;
        double stdv;

        double classValPenalty = classValOn ? 1:0;

        double[] output = new double[input.length];
        double seriesTotal = 0;
        double seriesTotal2 = 0;

        for (int i = 0; i < input.length - classValPenalty; i++)
        {
            seriesTotal += input[i];
            seriesTotal2 += (input[i] * input[i]);
        }

        if (sum != null && sum2 != null)
        {
            sum.set(seriesTotal);
            sum2.set(seriesTotal2);
        }

        mean = seriesTotal / (input.length - classValPenalty);
        double num = (seriesTotal2 - (mean * mean * (input.length - classValPenalty))) / (input.length - classValPenalty);
        stdv = (num <= ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(num);

        for (int i = 0; i < input.length - classValPenalty; i++)
        {
            output[i] = (stdv == 0.0) ? 0.0 : (input[i] - mean) / stdv;
        }

        if (classValOn)
        {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    private static class DoubleWrapper
    {

        private double d;

        public DoubleWrapper()
        {
            d = 0.0;
        }

        public DoubleWrapper(double d)
        {
            this.d = d;
        }

        public void set(double d)
        {
            this.d = d;
        }

        public double get()
        {
            return d;
        }
    }

    @Override
    public long opCountForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) throws Exception
    {
        data = FullShapeletTransform2.roundRobinData(data, null);
        subseqDistOpCount = 0;
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        return subseqDistOpCount;
    }

    /**
     *
     * @param args
     */
    public static void main(String[] args)
    {

        //################ Test 1 ################
        System.out.println("1) Testing index sorter: ");
        double[] series = new double[10];
        double[] subseq = new double[series.length / 2];

        int min = -5;
        int max = 5;
        for (int i = 0; i < series.length; i++)
        {
            series[i] = min + (int) (Math.random() * ((max - min) + 1));
            if (i < series.length / 2)
            {
                subseq[i] = min + (int) (Math.random() * ((max - min) + 1));
            }
        }

        printSeries(series);
        double[][] indices = sortIndexes(series);
        for (int i = 0; i < series.length; i++)
        {
            System.out.print(series[(int) indices[i][0]] + ((i == series.length - 1) ? "\n" : ", "));
        }

        //################ Test 2 ################
        System.out.println("\n 2) Testing normalization: ");
        double[] normSeries;
        normSeries = FullShapeletTransform2.zNormalise(series, false);
        System.out.print("Original: ");
        printSeries(normSeries);
        normSeries = optimizedZNormalise(series, false);
        System.out.print("Optimized: ");
        printSeries(normSeries);

        //################ Test 3 ################
        System.out.println("\n 2) Testing subsequence distance: ");
        System.out.println("Original dist: " + FullShapeletTransform2.subsequenceDistance(subseq, normSeries));
        double[][] sortedIndexes = sortIndexes(subseq);
        System.out.println("Optimized dist: " + onlineSubsequenceDistance(subseq, sortedIndexes, normSeries));
    }
    /* Method to estimate the range 
    
    
     */

    private static int[] estimateMinMax(Instances data, int runs, int sampleSize)
    {
        int[] minMax = new int[2];
        ArrayList<Integer> lengths = new ArrayList<>();
//        System.out.println("Performing length estimation");

        for (int i = 0; i < runs; i++)
        {
//            System.out.println("Sample "+i);
//  jacknife: sample without replacement
            Instances copy = new Instances(data);
            Random ran = new Random();
            copy.randomize(ran);
            Instances sample = new Instances(copy, 10);
            for (int j = 0; j < sampleSize; j++)
            {
                sample.add(copy.instance(j));
            }

            ShapeletTransformDistCaching shp = new ShapeletTransformDistCaching(sampleSize, 3, data.numAttributes() - 1);
            shp.setCandidatePruning(false);
            shp.setUseSeparationGap(true);
            shp.setRoundRobin(true);
            shp.supressOutput();
            try
            {
                shp.process(sample);
            }
            catch (Exception ex)
            {
                Logger.getLogger(ShapeletTransform.class.getName()).log(Level.SEVERE, null, ex);
            }
            ArrayList<Integer> sampleLengths = shp.getShapeletLengths();
//            print(sampleLengths);
            lengths.addAll(sampleLengths);
            System.out.println("Completed sample " + i);
        }

        Collections.sort(lengths);
        int numShapelets = sampleSize * runs;
        int lowerQuartile = numShapelets / 4 - 1;
        int upperQuartile = (numShapelets / 4) * 3 - 1;

        minMax[0] = lengths.get(lowerQuartile);
        minMax[1] = lengths.get(upperQuartile);
//        System.out.println("Minimum shapelet length = "+minMax[0]+" maximum"
//                + " shapelet length = "+minMax[1]);

        return minMax;
    }

    /**
     *
     * @param series
     */
    public static void printSeries(double[] series)
    {
        for (int i = 0; i < series.length; i++)
        {
            System.out.print(series[i] + ((i == series.length - 1) ? "\n" : ", "));
        }
    }
}
