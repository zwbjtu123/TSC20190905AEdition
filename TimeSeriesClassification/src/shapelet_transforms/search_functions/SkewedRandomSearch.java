/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package shapelet_transforms.search_functions;

import java.util.ArrayList;
import utilities.generic_storage.Pair;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class SkewedRandomSearch extends ImpRandomSearch{
    
    int[] lengthDistribution;
    
    protected SkewedRandomSearch(ShapeletSearchOptions sops){
        super(sops);
        
        lengthDistribution = sops.getLengthDistribution();
    }
    
    @Override
    public void init(Instances input){
        inputData = input;

        //generate the random shapelets we're going to visit.
        for(int i=0; i<numShapelets; i++){
            //randomly generate values.
            int series = random.nextInt(input.numInstances());
            int length = lengthDistribution[random.nextInt(lengthDistribution.length)]; //select the random length from the distribution of lengths.
            int position  = random.nextInt(input.numAttributes() - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
            
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
