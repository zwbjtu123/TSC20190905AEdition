/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.searchFuntions;

import java.util.ArrayList;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.Shapelet;

/**
 *
 * @author raj09hxu
 */
public class ShapeletSearch {
    
    public interface ProcessCandidate{
        public Shapelet process(double[] candidate);
    }
    
    
    int minShapeletLength;
    int maxShapeletLength;
    
    public ShapeletSearch(int min, int max){
        minShapeletLength = min;
        maxShapeletLength = max;
    }
    
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        double[] series = timeSeries.toDoubleArray();

        for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
            double[] candidate = new double[length];
            //for all possible starting positions of that length
            for (int start = 0; start <= series.length - length - 1; start++) {
                //-1 = avoid classVal - handle later for series with no class val
                // CANDIDATE ESTABLISHED - got original series, length and starting position
                // extract relevant part into a double[] for processing
                System.arraycopy(series, start, candidate, 0, length);

                Shapelet shapelet = checkCandidate.process(candidate);

                if (shapelet != null) {
                    seriesShapelets.add(shapelet);
                }
            }
        }
        
        return seriesShapelets;
    }
    
}
