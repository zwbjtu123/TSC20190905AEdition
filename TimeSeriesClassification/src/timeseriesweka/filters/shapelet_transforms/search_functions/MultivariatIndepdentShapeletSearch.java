/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.util.ArrayList;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class MultivariatIndepdentShapeletSearch extends ShapeletSearch {
    
    
    
    protected MultivariatIndepdentShapeletSearch(ShapeletSearchOptions ops) {
        super(ops);
    }
    
    @Override
    public void init(Instances input){
        super.init(input);
        
    }
    
    //given a series and a function to find a shapelet 
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        //split the time series up into sections.
        
        
        Instance[] dimension = utilities.MultivariateInstanceTools.splitMultivariateInstance(timeSeries);
        
        for (int dim = 0; dim < dimension.length; dim++) {
            Instance series = dimension[dim];
            
            for (int length = minShapeletLength; length <= maxShapeletLength; length+=lengthIncrement) {
                //for all possible starting positions of that length. -1 to remove classValue but would be +1 (m-l+1) so cancel.
                for (int start = 0; start < seriesLength - length; start+=positionIncrement) {
                    Shapelet shapelet = checkCandidate.process(series, start, length);
                    
                    if (shapelet != null) {
                        seriesShapelets.add(shapelet);
                        shapeletsVisited.add(seriesCount+","+length+","+start+","+shapelet.qualityValue);
                    }
                }
            }
        }
        seriesCount++;
        return seriesShapelets;
    }
    
}
