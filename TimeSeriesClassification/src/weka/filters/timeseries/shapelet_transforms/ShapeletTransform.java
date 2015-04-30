package weka.filters.timeseries.shapelet_transforms;

import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;

/**
 * An optimised filter to transform a dataset by k shapelets.
 *
 * This method uses the distance calculation early abandons described in the
 * "Trillions" KDD paper from Eamonn's group
 *
 * @author Edgaras Baranauskas
 */
public class ShapeletTransform extends FullShapeletTransform
{

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletTransform()
    {
        super();
        this.subseqDistance = new OnlineSubSeqDistance();
    }

    /**
     * Single param constructor: filter is unusable until min/max params are
     * initialised. Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     */
    public ShapeletTransform(int k)
    {
        super(k);
        this.subseqDistance = new OnlineSubSeqDistance();
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletTransform(int k, int minShapeletLength, int maxShapeletLength)
    {
        super(k, minShapeletLength, maxShapeletLength);
        this.subseqDistance = new OnlineSubSeqDistance();
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
    public ShapeletTransform(int k, int minShapeletLength, int maxShapeletLength, QualityMeasures.ShapeletQualityChoice qualityChoice)
    {
        super(k, minShapeletLength, maxShapeletLength, qualityChoice);
        this.subseqDistance = new OnlineSubSeqDistance();
    }

    /**
     *
     * @param args
     */
    public static void main(String[] args)
    {

       /* //################ Test 1 ################
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
        normSeries = FullShapeletTransform.zNormalise(series, false);
        System.out.print("Original: ");
        printSeries(normSeries);
        normSeries = optimizedZNormalise(series, false);
        System.out.print("Optimized: ");
        printSeries(normSeries);

        //################ Test 3 ################
        System.out.println("\n 2) Testing subsequence distance: ");
        FullShapeletTransform fst = new FullShapeletTransform();
        System.out.println("Original dist: " + fst.subsequenceDistance(subseq, normSeries));
        double[][] sortedIndexes = sortIndexes(subseq);
        //System.out.println("Optimized dist: " + onlineSubsequenceDistance(subseq, sortedIndexes, normSeries));*/
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
