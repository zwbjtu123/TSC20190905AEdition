/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.searchFuntions;

import java.util.ArrayList;
import utilities.generic_storage.Pair;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 *
 * @author raj09hxu
 */
public class RefinedRandomSearch extends ImpRandomSearch{

    float shapeletToSeriesRatio;
    
    public RefinedRandomSearch(int min, int max, long shapelets, long seed, float prop) {
        super(min, max, shapelets, seed);
        
        shapeletToSeriesRatio = prop;
    }
       
    @Override
    public void init(Instances input){
         int numInstances = input.numInstances();
         int numAttributes = input.numAttributes() - 1;
        
        System.out.println("numIntances: " + numInstances);
         
         float currentRatio;
         do{
            long totalShapelets = ShapeletTransformFactory.calculateNumberOfShapelets(--numInstances, numAttributes, minShapeletLength, maxShapeletLength);
            currentRatio = (float) numShapelets / (float) totalShapelets;
            
            if(numInstances == 25) break; // any less than 25 and we've sampled too far (Subject to change and discussion).
            
        }while(currentRatio < shapeletToSeriesRatio);
         
        System.out.println("numIntances: " + numInstances);
        System.out.println(currentRatio);
         
        
        inputData = input;
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        
        
        //generate the random shapelets we're going to visit.
        for(int i=0; i<numShapelets; i++){
            //randomly generate values.
            int series = random.nextInt(numInstances);
            int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
            int position  = random.nextInt(numAttributes - length + 1); // can only have valid start positions based on the length. the upper bound is exclusive. 
            //so for the m-m+1 case it always resolves to 0.
            
            //find the shapelets for that series.
            ArrayList<Pair<Integer,Integer>> shapeletList = shapeletsToFind.get(series);
            if(shapeletList == null)
                shapeletList = new ArrayList<>();
            
            //add the random shapelet to the length
            shapeletList.add(new Pair(length, position));
            //put back the updated version.
            shapeletsToFind.put(series, shapeletList);
        }          
    }
    
    
}