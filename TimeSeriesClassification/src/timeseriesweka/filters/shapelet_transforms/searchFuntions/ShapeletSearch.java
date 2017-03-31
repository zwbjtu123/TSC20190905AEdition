/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.searchFuntions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
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
    
    ArrayList<String> shapeletsVisited = new ArrayList<>();
    int seriesCount;
    
    public ArrayList<String> getShapeletsVisited() {
        return shapeletsVisited;
    }
    
    protected Comparator<Shapelet> comparator;
    
    public void setComparator(Comparator<Shapelet> comp){
        comparator = comp;
    }
    
    
    protected int minShapeletLength;
    protected int maxShapeletLength;
    
    protected int lengthIncrement = 1;
    protected int positionIncrement = 1;
    
    protected Instances inputData;

    public ShapeletSearch(int min, int max){
        minShapeletLength = min;
        maxShapeletLength = max;
    }
    
    public ShapeletSearch(int min, int max, int lengthInc, int posInc){
        this(min, max);
        lengthIncrement = lengthInc;
        positionIncrement = posInc;
    }
    
    public void setMinAndMax(int min, int max){
        minShapeletLength = min;
        maxShapeletLength = max;
    }
    
    public void init(Instances input){
        inputData = input;
    }
    
    
    //given a series and a function to find a shapelet 
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

        double[] series = timeSeries.toDoubleArray();

        
        for (int length = minShapeletLength; length <= maxShapeletLength; length+=lengthIncrement) {
            //for all possible starting positions of that length. -1 to remove classValue
            for (int start = 0; start <= timeSeries.numAttributes() - length - 1; start+=positionIncrement) {
                Shapelet shapelet = checkCandidate.process(series, start, length);
                if (shapelet != null) {
                    seriesShapelets.add(shapelet);
                }
                
                shapeletsVisited.add(seriesCount+","+length+","+start);
            }
        }
        
        seriesCount++;
        return seriesShapelets;
    }
}
