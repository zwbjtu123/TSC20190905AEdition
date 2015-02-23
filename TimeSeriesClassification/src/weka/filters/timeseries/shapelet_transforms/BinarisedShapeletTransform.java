/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.Shapelet;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.getClassDistributions;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.removeSelfSimilar;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.subsequenceDistance;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.subsequenceDistance;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.subsequenceDistance;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.subsequenceDistance;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.subsequenceDistance;

/**
 *
 * @author raj09hxu
 */
public class BinarisedShapeletTransform extends FullShapeletTransform
{

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
        ArrayList<Shapelet> kShapelets;
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        classDistributions = getClassDistributions(data);                       // used to calc info gain

        //construct a map for our K-shapelets lists, on for each classVal.
        Map<Double, ArrayList<Shapelet>> kShapeletsMap = new TreeMap();
        for (Double classVal : classDistributions.keySet())
        {
            kShapeletsMap.put(classVal, new ArrayList<Shapelet>());
        }

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

            seriesShapelets = findShapeletCandidates(data, i, wholeCandidate, kShapelets);

            Comparator comp = useSeparationGap ? new Shapelet.ReverseSeparationGap() : new Shapelet.ReverseOrder();
            Collections.sort(seriesShapelets, comp);

            seriesShapelets = removeSelfSimilar(seriesShapelets);

            kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
            
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
    protected Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, QualityBound.ShapeletQualityBound qualityBound)
    {

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        int dataSize = data.numInstances();

        //TODO: 
        //the way we calculate our orderline needs to be different. 
        //We need to build the orderline as a representation of whether the class value is the same as ours or not. BINARY!
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
                distance = subsequenceDistance(candidate, getToDoubleArrayOfInstance(data, i));
            }

            //TODO: change this line to binarise instead of copying the value.
            double classVal = data.instance(i).classValue();

            // without early abandon, it is faster to just add and sort at the end. 
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
        return shapelet;
    }

}
