/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.Shapelet;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.ROUNDING_ERROR_CORRECTION;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.getClassDistributions;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.removeSelfSimilar;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransform.onlineSubsequenceDistance;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransform.sortIndexes;

/**
 *
 * @author raj09hxu
 */
public class BinarisedShapeletTransform extends ShapeletTransform
{

    Map<Double, Map<Double, Integer>> binaryClassDistribution;
    
    //default.
    public BinarisedShapeletTransform()
    {
        super();
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data)
    {
        //TODO: update this to have a list for each class instead of a mega list.
        
        classDistributions = getClassDistributions(data);                       // used to calc info gain
        binaryClassDistribution = buildBinaryDistributions(classDistributions); //used for binary info gain.

        ArrayList<Shapelet> kShapelets;
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        //construct a map for our K-shapelets lists, on for each classVal.
        Map<Double, ArrayList<Shapelet>> kShapeletsMap = new TreeMap();
        for (Double classVal : classDistributions.keySet())
        {
            kShapeletsMap.put(classVal, new ArrayList<Shapelet>());
        }
        
        //found out how many we want in each sub list.
        int proportion = numShapelets/kShapeletsMap.keySet().size();
        System.out.println("proportion: " + proportion);
        
        //for all time series
        outputPrint("Processing data: ");

        int dataSize = data.numInstances();
        //for all possible time series.
        for (int i = 0; i < dataSize; i++)
        {
            outputPrint("data : " + i);
            
            //get the Shapelets list based on the classValue of our current time series.
            kShapelets = kShapeletsMap.get(data.get(i).classValue());

            double[] wholeCandidate = getToDoubleArrayOfInstance(data, i);
            
            //we only want to pass in the worstKShapelet if we've found K shapelets. but we only care about this class values worst one.
            //this is due to the way we represent each classes shapelets in the map.
            Shapelet kWorst = kShapelets.size() == proportion ? kShapelets.get(kShapelets.size()-1) : null;
            
            seriesShapelets = findShapeletCandidates(data, i, wholeCandidate, kWorst);

            Comparator comp = useSeparationGap ? new Shapelet.ReverseSeparationGap() : new Shapelet.ReverseOrder();
            Collections.sort(seriesShapelets, comp);

            seriesShapelets = removeSelfSimilar(seriesShapelets);

            kShapelets = combine(proportion, kShapelets, seriesShapelets);
            
            //re-update the list because it's changed now. 
            kShapeletsMap.put(data.get(i).classValue(), kShapelets);
        }

        kShapelets = buildKShapeletsFromMap(kShapeletsMap);
        
        this.numShapelets = kShapelets.size();

        recordShapelets(kShapelets);
        printShapelets(kShapelets);

        return kShapelets;
    }
       
    
    private ArrayList<Shapelet> buildKShapeletsFromMap(Map<Double, ArrayList<Shapelet>> kShapeletsMap)
    {
       ArrayList<Shapelet> kShapelets = new ArrayList<>();
       
       int numberOfClassVals = kShapeletsMap.keySet().size();
       int proportion = numShapelets/numberOfClassVals;
       
       
       Iterator<Shapelet> it = null;
       
       //all lists should be sorted.
       //go through the map and get the sub portion of best shapelets for the final list.
       for(ArrayList<Shapelet> list : kShapeletsMap.values())
       {
           int i=0;
           it = list.iterator();
           
           while(it.hasNext() && i++ <= proportion)
           {
               kShapelets.add(it.next());
           }
       }
        
       
       return kShapelets;
    }
    
    @Override
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
                QualityBound.ShapeletQualityBound qualityBound = initializeQualityBound(getBinaryDistribution(data.get(i).classValue()));

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
    
    

    @Override
    protected Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, QualityBound.ShapeletQualityBound qualityBound)
    {
        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();
        
        //we want to build a differentClassDistributions one which represents a binary split of the data, and a binary orderline.
        double shapeletClassVal = data.get(seriesId).classValue();
        int dataSize = data.numInstances();
        
        
        double[][] sortedIndexes = sortIndexes(candidate);
        
        long currentOp = subseqDistOpCount;
        
        for (int i = 0; i < dataSize; i++)
        {
            //Check if it is possible to prune the candidate. if it is possible, we can just return null.
            if (qualityBound != null && qualityBound.pruneCandidate())
            {
                prunes+= (dataSize - i); //how many we're skipping.
                writeToLogFile(pruneLogFile,"%d,%d,%d,%d,%d\n",seriesId, startPos, candidate.length, (dataSize - i), prunes);
                return null;
            }

            double distance = 0.0;
            //don't compare the shapelet to the the time series it came from.
            if (i != seriesId)
            {
                distance = onlineSubsequenceDistance(candidate, sortedIndexes, getToDoubleArrayOfInstance(data, i), startPos);
            }

            //binarise instead of copying the value.
            double classVal = binariseClassVal(data.instance(i).classValue(), shapeletClassVal);

            // without early abandon, it is faster to just add and sort at the end. 
            orderline.add(new OrderLineObj(distance, classVal));

            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            if (qualityBound != null)
            {
                qualityBound.updateOrderLine(orderline.get(orderline.size() - 1));
            }
        }
        
        writeToLogFile("%d,%d,%d,%d,%d\n", seriesId, startPos, candidate.length, (subseqDistOpCount-currentOp), subseqDistOpCount);

        // note: early abandon entropy pruning would appear here, but has been ommitted
        // in favour of a clear multi-class information gain calculation. Could be added in
        // this method in the future for speed up, but distance early abandon is more important
        // create a shapelet object to store all necessary info, i.e.
        Shapelet shapelet = new Shapelet(candidate, dataSourceIDs[seriesId], startPos, this.qualityMeasure);
        shapelet.calculateQuality(orderline, getBinaryDistribution(shapeletClassVal));
        shapelet.classValue = shapeletClassVal; //set classValue of shapelet. (interesing to know).
        return shapelet;
    }
    
    private static double binariseClassVal(double classVal, double shapeletClassVal)
    {
        return classVal == shapeletClassVal ? 0.0 : 1.0;
    }
    
    private static Map<Double, Integer> binariseDistributions(Map<Double, Integer> classDistributions, double shapeletClassVal)
    {
        Map<Double, Integer> binaryDistribution = new TreeMap<>();

        Integer shapeletClassCount = classDistributions.get(shapeletClassVal);
        binaryDistribution.put(0.0, shapeletClassCount);
        
        int sum = 0;
        for(Integer count : classDistributions.values())
        {
            sum += count;
        }
        
        //remove the shapeletsClass count. Rest should be all the other classes.
        sum -= shapeletClassCount; 
        binaryDistribution.put(1.0, sum);
        return binaryDistribution;
    }
    
    public static Map<Double, Map<Double, Integer>> buildBinaryDistributions(Map<Double, Integer> classDistributions)
    {
        Map<Double, Map<Double, Integer>> binaryMapping = new TreeMap<>();
        
        //for each classVal build a binary distribution map.
        for(Double cVal : classDistributions.keySet())
        {
            binaryMapping.put(cVal, binariseDistributions(classDistributions, cVal));
        }
        return binaryMapping;
    }
    
    public Map<Double, Integer> getBinaryDistribution(double classVal)
    {
        return this.binaryClassDistribution.get(classVal);
    }
    
       @Override
    protected Instances buildTansformedDataset(Instances data)
    {
         //logFile
        writeToLogFile("\nTRAIN\n");
        
        //Reorder the training data and reset the shapelet indexes
        Instances output = determineOutputFormat(data);

        Shapelet s;
        double[][] sortedIndexes;
        // for each data, get distance to each shapelet and create new instance
        int size = shapelets.size();
        int dataSize = data.numInstances();

        //create our data instances
        for (int j = 0; j < dataSize; j++)
        {
            output.add(new DenseInstance(size + 1));
        }

        double dist;
        for (int i = 0; i < size; i++)
        {
            s = shapelets.get(i);
            sortedIndexes = sortIndexes(s.content);
            long currentOp = subseqDistOpCount;
            for (int j = 0; j < dataSize; j++)
            {
                dist = onlineSubsequenceDistance(s.content, sortedIndexes, getToDoubleArrayOfInstance(data, j), s.startPos);
                output.instance(j).setValue(i, dist);
            }
            writeToLogFile("%d,%d,%d,%d,%d\n",s.seriesId, s.startPos, s.content.length, (subseqDistOpCount-currentOp), subseqDistOpCount);
        }

        //do the classValues.
        for (int j = 0; j < dataSize; j++)
        {
            output.instance(j).setValue(size, data.instance(j).classValue());
        }

        return output;
    }
    
    
     /**
     * Calculate the distance between a shapelet candidate and a full time
     * series (both double[]).
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param sortedIndices
     * @param timeSeries a double[] representation of a whole time series (inc.
     * class value)
     * @param startPos
     * @return the distance between a candidate and a time series
     *
     *
     * NOTE: it seems that the reordering is repeated for each new time series.
     * This could be avoided, but not sure how to structure the code to do it
     */
    public static double onlineSubsequenceDistance(double[] candidate, double[][] sortedIndices, double[] timeSeries, int startPos)
    {
        DoubleWrapper sumPointer = new DoubleWrapper();
        DoubleWrapper sum2Pointer = new DoubleWrapper();

        //Generate initial subsequence that starts at the same position our candidate does.
        double[] subseq = new double[candidate.length];
        System.arraycopy(timeSeries, startPos, subseq, 0, subseq.length);
        subseq = optimizedZNormalise(subseq, false, sumPointer, sum2Pointer);
        

        //Keep count of fundamental ops for experiment
        subseqDistOpCount += subseq.length;

        //copy for left and right.
        double sumR = sumPointer.get(),    sumL = sumPointer.get();
        double sum2R = sum2Pointer.get(),  sum2L = sum2Pointer.get();
        
        double bestDist = 0.0;
        double temp;
        
        //Compute initial distance. from the startPosition the candidate was found.
        for (int i = 0; i < candidate.length; i++)
        {
            temp = candidate[i] - subseq[i];
            bestDist = bestDist + (temp * temp);
        }

        
                
        System.out.println("bestDist: " + bestDist);
        
        
        //Keep count of fundamental ops for experiment
        subseqDistOpCount += candidate.length;

        double startL, endL, startR, endR;

        // Scan through all possible subsequences of two
        boolean traverseRight = true, traverseLeft = true;
        int i=1;
        int posL, posR;
        double currentDist;
        while(traverseRight || traverseLeft)
        {
            posL = startPos - i;
            posR = startPos + i;
            
            traverseRight = posR < timeSeries.length - candidate.length;
            traverseLeft = posL >= 0;

            if(traverseRight)     
            {     
                startR  = timeSeries[posR-1];
                endR    = timeSeries[posR-1 + candidate.length];
            
                //Update the running sums - get the begining and remove, get the end and add. going right.
                sumR = sumR - startR + endR; 
                sum2R = sum2R -(startR * startR) + (endR * endR);

                currentDist = calculateBestDistance(posR, timeSeries, candidate, sortedIndices, bestDist, sumR, sum2R);  
                
                if (currentDist < bestDist)
                {
                    bestDist = currentDist;
                    System.out.println("best dist pos: " + posR);
                }
            }
            
            if(traverseLeft)
            {
                startL  = timeSeries[posL];
                endL    = timeSeries[posL + candidate.length];
                
                //Update the running sums - get the begining and add, get the end and remove. going left.
                sumL = sumL + startL - endL; 
                sum2L = sum2L + (startL * startL) - (endL * endL);
                
                currentDist = calculateBestDistance(posL, timeSeries, candidate, sortedIndices, bestDist, sumL, sum2L);

                if (currentDist < bestDist)
                {
                    bestDist = currentDist;
                    System.out.println("best dist pos: " + posL);
                }
            }
            
            i++;
        }

        return (bestDist == 0.0) ? 0.0 : (1.0 / candidate.length * bestDist);
    }
    
    private static double calculateBestDistance(int i, double[] timeSeries, double[] candidate, double[][] sortedIndices, double bestDist, double sum, double sum2)
    {
        //Compute the stats for new series
        double mean = sum / candidate.length;

        //Get rid of rounding errors
        double stdv2 = (sum2 - (mean * mean * candidate.length)) / candidate.length;

        double stdv = (stdv2 < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv2);
                  
        
        //calculate the normalised distance between the series
        int j = 0;
        double currentDist = 0.0;
        double toAdd;
        int reordedIndex;
        double normalisedVal = 0.0;
        boolean dontStdv = (stdv == 0.0);

        while (j < candidate.length  && currentDist < bestDist)
        {
            reordedIndex = (int) sortedIndices[j][0];
            //if our stdv isn't done then make it 0.
            normalisedVal = dontStdv ? 0.0 : ((timeSeries[i + reordedIndex] - mean) / stdv);
            toAdd = candidate[reordedIndex] - normalisedVal;
            currentDist += (toAdd * toAdd);
            j++;

            //Keep count of fundamental ops for experiment
            subseqDistOpCount++;
        }
        
                    
        //System.out.println("binarised performed calculations: " + j + " of " + candidate.length);
            


        return currentDist;
    }
    
    
    public static void main(String[] args)
    {
        test();
    }
    
    public static void test()
    {
        double[] series = {1.598006,1.599439,1.570529,1.550474,1.507371,1.434341,1.368986,1.305294,1.210305,1.116653,1.023976,0.925977,0.828107,0.739222,0.643051,0.556432,0.462948,0.369575,0.278426,0.18583,0.095532,0.010646,-0.080856,-0.165549,-0.243492,-0.33543,-0.426109,-0.497486,-0.56685,-0.649856,-0.732247,-0.7778,-0.842984,-0.917304,-0.987973,-1.055011,-1.096775,-1.147065,-1.198429,-1.243188,-1.28163,-1.312638,-1.335745,-1.351122,-1.37692,-1.387935,-1.376892,-1.357099,-1.329577,-1.293692,-1.257696,-1.230419,-1.17425,-1.114199,-1.072004,-1.006755,-0.948929,-0.870171,-0.79017,-0.724507,-0.642835,-0.591634,-0.521392,-0.43063,-0.350848,-0.269856,-0.173638,-0.089113,-0.00392,0.085453,0.175176,0.267055,0.362971,0.452381,0.549112,0.649817,0.746244,0.845659,0.946333,1.045681,1.146307,1.24236,1.343994,1.441798,1.540663,1.624094,1.669585,1.685201,1.69666,1.697462,1.687431,1.638525,1.573703,1.478584,1.376902,1.276597,1.177031,1.0777,0.979203,0.880533,0.777926,0.681645,0.590432,0.492751,0.404343,0.308589,0.213516,0.122677,0.037603,-0.052602,-0.13751,-0.220402,-0.315558,-0.394614,-0.446076,-0.533439,-0.60286,-0.673407,-0.737606,-0.81903,-0.885258,-0.950083,-1.002563,-1.063003,-1.125195,-1.165769,-1.204262,-1.248452,-1.27377,-1.28438,-1.273337,-1.286739,-1.291872,-1.288647,-1.277739,-1.258145,-1.231352,-1.197268,-1.156312,-1.110456,-1.084524,-1.025831,-0.962551,-0.895071,-0.847815,-0.772537,-0.694573,-0.631998,-0.551219,-0.484587,-0.408677,-0.324424,-0.247454,-0.163651,-0.076913,0.008801,0.097274,0.185896,0.284005,0.381205,0.463025,0.556933,0.65097,0.7382,0.834742,0.929939,1.02486,1.120906,1.217175,1.31253,1.40292,1.481043,1.521012,1.564154,1.570854,1.59289};
        double[] subseq = {-1.097483,-1.158599,-1.22042,-1.277007,-1.32718,-1.369576,-1.395475,-1.416281,-1.427878,-1.427286,-1.422008};
        double[][] sortedIndexes = sortIndexes(subseq);
        
        double [] normSeries = optimizedZNormalise(series, false);
        
        System.out.println("BinarisedShapeletTransform.subseqDistOpCount: " + BinarisedShapeletTransform.subseqDistOpCount);
        BinarisedShapeletTransform.onlineSubsequenceDistance(subseq, sortedIndexes, normSeries, 124);
        System.out.println("BinarisedShapeletTransform.subseqDistOpCount: " + BinarisedShapeletTransform.subseqDistOpCount);
        ShapeletTransform.subseqDistOpCount=0;
        System.out.println("ShapeletTransform.subseqDistOpCount: " + ShapeletTransform.subseqDistOpCount);
        ShapeletTransform.onlineSubsequenceDistance(subseq, sortedIndexes, normSeries);
        System.out.println("ShapeletTransform.subseqDistOpCount: " + ShapeletTransform.subseqDistOpCount);
    
    }

}
