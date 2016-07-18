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

/**
 *
 * @author raj09hxu
 */
public class RandomTimedSearch extends ShapeletSearch{
        
    long timeLimit;
    Random random;
    
    boolean[][] visited;
    
    protected RandomTimedSearch(int min, int max){
        super(min,max);
    }
    
    public RandomTimedSearch(int min, int max, long time, long seed) {
        super(min, max);    
        
        timeLimit = time;
        
        random = new Random(seed);
    }
    
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        
        long currentTime =0;
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        double[] series = timeSeries.toDoubleArray();

        int numLengths = maxShapeletLength - minShapeletLength /*+ 1*/; //want max value to be inclusive.
        
        visited = new boolean[numLengths][];
        
        //you only get a 1/nth of the time.
        while((timeLimit/inputData.numInstances()) > currentTime){
            int lengthIndex = random.nextInt(numLengths);
            int length = lengthIndex + minShapeletLength; //offset the index by the min value.
            
            int maxPositions = series.length - length ;
            int start  = random.nextInt(maxPositions); // can only have valid start positions based on the length.

            //we haven't constructed the memory for this length yet.
            initVisitedMemory(series, length);
            
            Shapelet shape = visitCandidate(series, start, length, checkCandidate);
            if(shape != null)
                seriesShapelets.add(shape);

            
            //we add time o, even if we've visited it, this is just incase we end up stuck in some improbable recursive loop.
            currentTime += calculateTimeToRun(inputData.numInstances(), series.length-1, length); //n,m,l            
        }

        
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
    
    protected long calculateTimeToRun(int n, int m, int length){
        long time = (m - length + 1) * length; //number of subsequeneces in the seuquenece, and we do euclidean comparison length times for each.
        return time * (n-1); //we calculate this for n-1 series.
    }
    
        
    private Shapelet visitCandidate(double[] series, int start, int length, ProcessCandidate checkCandidate){
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
