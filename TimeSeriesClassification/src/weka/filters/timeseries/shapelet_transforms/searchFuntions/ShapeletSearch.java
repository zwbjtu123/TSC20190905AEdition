/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.searchFuntions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.function.Function;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.Shapelet;

/**
 *
 * @author raj09hxu
 */
public class ShapeletSearch implements Serializable{
    
    public interface ProcessCandidate{
        public Shapelet process(double[] candidate, int start, int length);
    }
    
    protected int minShapeletLength;
    protected int maxShapeletLength;
    
    public ShapeletSearch(int min, int max){
        minShapeletLength = min;
        maxShapeletLength = max;
    }
    
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

        double[] series = timeSeries.toDoubleArray();
        
        for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
            //for all possible starting positions of that length
            for (int start = 0; start <= timeSeries.numAttributes() - length - 1; start++) {
                Shapelet shapelet = checkCandidate.process(series, start, length);

                if (shapelet != null) {
                    seriesShapelets.add(shapelet);
                }
            }
        }
        
        return seriesShapelets;
    }
}
