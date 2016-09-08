/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.searchFuntions;

import java.util.ArrayList;
import java.util.Random;
import weka.core.Instance;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 *
 * @author raj09hxu
 */
public class RandomSearch extends ShapeletSearch{
        
    protected Random random;
    private long numShapelets;
    
    protected boolean[][] visited;

    protected RandomSearch(int min, int max){
        super(min,max);
    }
    
    public RandomSearch(int min, int max, long shapelets, long seed) {
        super(min, max);    
        
        numShapelets = shapelets;
        random = new Random(seed);
    }
    
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        double[] series = timeSeries.toDoubleArray();

        int numLengths = maxShapeletLength - minShapeletLength /*+ 1*/; //want max value to be inclusive.
        
        visited = new boolean[numLengths][];
        
        //Only consider a fixed amount of shapelets.
        for(int i=0; i<numShapelets; i++ ){
            int lengthIndex = random.nextInt(numLengths);
            int length = lengthIndex + minShapeletLength; //offset the index by the min value.
            
            int maxPositions = series.length - length ;
            int start  = random.nextInt(maxPositions); // can only have valid start positions based on the length.

            //we haven't constructed the memory for this length yet.
            initVisitedMemory(series, length);
            
            Shapelet shape = visitCandidate(series, start, length, checkCandidate);
            if(shape != null)
                seriesShapelets.add(shape);           
        }

        for(int i=0; i<visited.length; i++){
            if(visited[i] == null) continue;
            for(int j=0; j<visited[i].length; j++){
                if(visited[i][j])
                    shapeletsVisited.add(seriesCount+","+(i+minShapeletLength)+","+j);
            }
        }
        
        seriesCount++; //keep track of the series.
        
        
        return seriesShapelets;
    }
    
        
    protected void initVisitedMemory(double[] series, int length){
        int lengthIndex = getLenghtIndex(length);
        if(visited[lengthIndex] == null){
            int maxPositions = series.length - length;
            visited[lengthIndex] = new boolean[maxPositions];
        }  
    }
    
        
    protected int getLenghtIndex(int length){
        return length - minShapeletLength;
    }
      
        
    protected Shapelet visitCandidate(double[] series, int start, int length, ProcessCandidate checkCandidate){
        initVisitedMemory(series, length);
        int lengthIndex = getLenghtIndex(length);
        Shapelet shape = null;     
        if(!visited[lengthIndex][start]){
            shape = checkCandidate.process(series, start, length);
            visited[lengthIndex][start] = true;
        }
        return shape;
    }

}
