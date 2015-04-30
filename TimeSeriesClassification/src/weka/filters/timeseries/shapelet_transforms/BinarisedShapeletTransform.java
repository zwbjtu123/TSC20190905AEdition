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
import static utilities.InstanceTools.getClassDistributions;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.ROUNDING_ERROR_CORRECTION;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.removeSelfSimilar;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;

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
        subseqDistance = new ImprovedOnlineSubSeqDistance();
    }

    public BinarisedShapeletTransform(int i, int min, int max, QualityMeasures.ShapeletQualityChoice shapeletQualityChoice) {
        super(i,min,max,shapeletQualityChoice);
        subseqDistance = new ImprovedOnlineSubSeqDistance();
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

            double[] wholeCandidate = data.get(i).toDoubleArray();
            
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
       
       
       Iterator<Shapelet> it;
       
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
        
        //set the candidate. This will also sort the indexes.
        subseqDistance.setCandidate(candidate);
        
        for (int i = 0; i < dataSize; i++)
        {
            //Check if it is possible to prune the candidate. if it is possible, we can just return null.
            if (qualityBound != null && qualityBound.pruneCandidate())
            {
                return null;
            }

            double distance = 0.0;
            //don't compare the shapelet to the the time series it came from.
            if (i != seriesId)
            {
                distance = subseqDistance.calculate(data.get(i).toDoubleArray(), startPos);
            }

            //binarise instead of copying the value. TODO: make this generic.
            double classVal = binariseClassVal(data.instance(i).classValue(), shapeletClassVal);

            // without early abandon, it is faster to just add and sort at the end. 
            orderline.add(new OrderLineObj(distance, classVal));

            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            if (qualityBound != null)
            {
                qualityBound.updateOrderLine(orderline.get(orderline.size() - 1));
            }
        }

        //TODO: Change this into a create Shapelet function.
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
    
    
    public static void main(String[] args)
    {
        test();
    }
    
    public static void test()
    {
        /*
        double[] series1 = {-2.35883,-2.35559,-2.35454,-2.34338,-2.31736,-2.30061,-2.28114,-2.26463,-2.21722,-2.17778,-2.14637,-2.12312,-2.09685,-2.04244,-1.99365,-1.93976,-1.89255,-1.82506,-1.73738,-1.6517,-1.60714,-1.50672,-1.42336,-1.32344,-1.27373,-1.18805,-1.10609,-0.986926,-0.889897,-0.872755,-0.809654,-0.740608,-0.677319,-0.661043,-0.644438,-0.607948,-0.576844,-0.557063,-0.548549,-0.517287,-0.475963,-0.425143,-0.388282,-0.342208,-0.320914,-0.248243,-0.256056,-0.229242,-0.202671,-0.228913,-0.228913,-0.217255,-0.217255,-0.230399,-0.205525,-0.180597,-0.168731,-0.183076,-0.197396,-0.210728,-0.226991,-0.282901,-0.339088,-0.369761,-0.409539,-0.453912,-0.520932,-0.540419,-0.550083,-0.629803,-0.673184,-0.71053,-0.745705,-0.825118,-0.87774,-0.918719,-0.94649,-0.962059,-0.98006,-0.974878,-0.966522,-0.954408,-0.962079,-0.982794,-1.03389,-1.07145,-1.16734,-1.23652,-1.30709,-1.34664,-1.39151,-1.42981,-1.46628,-1.50096,-1.48963,-1.52224,-1.53894,-1.55475,-1.55603,-1.58377,-1.6362,-1.64837,-1.64773,-1.65881,-1.67234,-1.65998,-1.64605,-1.65811,-1.62909,-1.5969,-1.55926,-1.54559,-1.52193,-1.53264,-1.48983,-1.46406,-1.45693,-1.43192,-1.41055,-1.36412,-1.34024,-1.32582,-1.38311,-1.43392,-1.46834,-1.52228,-1.55677,-1.53852,-1.52125,-1.47655,-1.44466,-1.41865,-1.42403,-1.42253,-1.44458,-1.43527,-1.43042,-1.43727,-1.45044,-1.46462,-1.49365,-1.51564,-1.53976,-1.54639,-1.55239,-1.5708,-1.58798,-1.59122,-1.59561,-1.56971,-1.55544,-1.55304,-1.52571,-1.49402,-1.49165,-1.51285,-1.47432,-1.45199,-1.43318,-1.40902,-1.39751,-1.36967,-1.33461,-1.29077,-1.22598,-1.21902,-1.1612,-1.10525,-0.990312,-0.86719,-0.784717,-0.781386,-0.772976,-0.794816,-0.815942,-0.822881,-0.821923,-0.772969,-0.725666,-0.649244,-0.65261,-0.598976,-0.590117,-0.552959,-0.541044,-0.466437,-0.433354,-0.415578,-0.390976,-0.385024,-0.325496,-0.306833,-0.289861,-0.262262,-0.233975,-0.244979,-0.230901,-0.230901,-0.227787,-0.213816,-0.175148,-0.210728,-0.210728,-0.232092,-0.257803,-0.259303,-0.271433,-0.282124,-0.323476,-0.390974,-0.406311,-0.45172,-0.487394,-0.558741,-0.633816,-0.653986,-0.70906,-0.719124,-0.734661,-0.761652,-0.777301,-0.807103,-0.856996,-0.873969,-0.973638,-1.04752,-1.1376,-1.20518,-1.24089,-1.3293,-1.43616,-1.51857,-1.55937,-1.64382,-1.72967,-1.80862,-1.84535,-1.91542,-1.97986,-2.03545,-2.08808,-2.12186,-2.16653,-2.20642,-2.22625,-2.25462,-2.28279,-2.30963,-2.32282,-2.33439,-2.32517};
        double[] series = {-2.42362,-2.42037,-2.40958,-2.38444,-2.34403,-2.31221,-2.2892,-2.26257,-2.20698,-2.17146,-2.11826,-2.06019,-2.00828,-1.93929,-1.91255,-1.8356,-1.75096,-1.6679,-1.57946,-1.49402,-1.39886,-1.30113,-1.20721,-1.10977,-1.04827,-0.957522,-0.883125,-0.827823,-0.766507,-0.711107,-0.685959,-0.628028,-0.574652,-0.525584,-0.480532,-0.439185,-0.401229,-0.37882,-0.339959,-0.343965,-0.312258,-0.293985,-0.278957,-0.243371,-0.231714,-0.244979,-0.23318,-0.208815,-0.196886,-0.209385,-0.197396,-0.185348,-0.185348,-0.173246,-0.161092,-0.161092,-0.161389,-0.173885,-0.161638,-0.174126,-0.186602,-0.174313,-0.174313,-0.174313,-0.186774,-0.149627,-0.137236,-0.174529,-0.162138,-0.162138,-0.162138,-0.149697,-0.177295,-0.206157,-0.251899,-0.28713,-0.32119,-0.365304,-0.390937,-0.442299,-0.451622,-0.503908,-0.531645,-0.608071,-0.657744,-0.657889,-0.708626,-0.759941,-0.797898,-0.834918,-0.858439,-0.89341,-0.940762,-0.97362,-1.00531,-1.03584,-1.05895,-1.10343,-1.13883,-1.20385,-1.2482,-1.30396,-1.36946,-1.43629,-1.50389,-1.56006,-1.64095,-1.70827,-1.75326,-1.81769,-1.87946,-1.93278,-1.97697,-1.9933,-1.98434,-1.98565,-1.97273,-1.96788,-1.9614,-1.95312,-1.94286,-1.92576,-1.93088,-1.92907,-1.90104,-1.90063,-1.93433,-1.93088,-1.94543,-1.95767,-1.96773,-1.97576,-1.99609,-1.98372,-1.98688,-1.97552,-1.9766,-1.96922,-1.95075,-1.90502,-1.84486,-1.79476,-1.73013,-1.66389,-1.6091,-1.54106,-1.4723,-1.40435,-1.3378,-1.26845,-1.20531,-1.14497,-1.07307,-1.02905,-1.00664,-1.00899,-0.963046,-0.95229,-0.918405,-0.883434,-0.859901,-0.822881,-0.797245,-0.746109,-0.719001,-0.695168,-0.645495,-0.588003,-0.546026,-0.49474,-0.45239,-0.418832,-0.349027,-0.309276,-0.272255,-0.263512,-0.243407,-0.214545,-0.199307,-0.149627,-0.137151,-0.149511,-0.137024,-0.137024,-0.14935,-0.124528,-0.136854,-0.136854,-0.149143,-0.161389,-0.161389,-0.173591,-0.185745,-0.185745,-0.197847,-0.209895,-0.184896,-0.196886,-0.184391,-0.19632,-0.171903,-0.195697,-0.207496,-0.219226,-0.20675,-0.217531,-0.229038,-0.255529,-0.282229,-0.289666,-0.305647,-0.338269,-0.353253,-0.412735,-0.451207,-0.492984,-0.52593,-0.550202,-0.603562,-0.636349,-0.698716,-0.766634,-0.815266,-0.882392,-0.943598,-1.03275,-1.11739,-1.2127,-1.28884,-1.38657,-1.46949,-1.55501,-1.64894,-1.73423,-1.8083,-1.88701,-1.9614,-2.01711,-2.04854,-2.10683,-2.16052,-2.20698,-2.25171,-2.29107,-2.31355,-2.35681,-2.38557,-2.41029,-2.41197,-2.42345,-2.42408};
        
        int startPos = 60;
        int length = 17; 
                
        double[] subseq = Arrays.copyOfRange(series1, startPos, startPos+length);
        double[][] sortedIndexes = sortIndexes(subseq);
        
        System.out.println("Shapelet: " + Arrays.toString(subseq));
        System.out.println("Series: "+ Arrays.toString(series));
        
        double [] normSeries = optimizedZNormalise(series, false);
        
        BinarisedShapeletTransform bst = new BinarisedShapeletTransform();
        System.out.println("BinarisedShapeletTransform.subseqDistOpCount: " + BinarisedShapeletTransform.subseqDistOpCount);
        bst.subsequenceDistance(subseq, normSeries, startPos, sortedIndexes);
        System.out.println("BinarisedShapeletTransform.subseqDistOpCount: " + BinarisedShapeletTransform.subseqDistOpCount);
        ShapeletTransform.subseqDistOpCount=0;
        System.out.println("ShapeletTransform.subseqDistOpCount: " + ShapeletTransform.subseqDistOpCount);
        ShapeletTransform st = new ShapeletTransform();
        st.subsequenceDistance(subseq, normSeries, 0, sortedIndexes);
        System.out.println("ShapeletTransform.subseqDistOpCount: " + ShapeletTransform.subseqDistOpCount);
        */
    }

}
