/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package shapelet_transforms.search_functions;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import utilities.generic_storage.Pair;
import weka.core.Instance;
import weka.core.Instances;
import shapelet_transforms.Shapelet;
/**
 *
 * @author raj09hxu
 */
public class TabuSearch extends ImpRandomSearch{
    int neighbourhoodWidth = 3;     //3x3 
    
    int maxTabuSize = 50;

    int numShapeletsPerSeries;
    
    Shapelet bsf_shapelet;

    @Deprecated
    public TabuSearch(ShapeletSearchOptions ops) {
        super(ops);
    }    
    
    @Override
    public void init(Instances input){
        inputData = input;
        numShapeletsPerSeries = (int) (numShapelets / inputData.numInstances());  
        
        // we might need to reduce the number of series. could do 10% subsampling.
        if(numShapeletsPerSeries < 1)
            System.err.println("Too Few Starting shapelets");
    }
    
    
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ShapeletSearch.ProcessCandidate checkCandidate){
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        double[] series = timeSeries.toDoubleArray();

        currentSeries++;

        Queue<Pair<Integer, Integer>> tabuList = new LinkedList<>();
        
        Pair<Integer, Integer> shapelet;
        
        int numShapeletsEvaluated = 0;

        
        //Only consider a fixed amount of shapelets.
        while(numShapeletsPerSeries > numShapeletsEvaluated){

            //create the random shapelet.
            //if it's the first iteration and we've got a previous best shapelet.
            if(numShapeletsEvaluated == 0 && bsf_shapelet != null){
               shapelet = new Pair<Integer,Integer>(bsf_shapelet.length, bsf_shapelet.startPos) ;
               bsf_shapelet = null; //reset the best one for this series.
            }else{
                shapelet = createRandomShapelet(timeSeries);
            }
            
            ArrayList<Pair<Integer, Integer>> candidateList = new ArrayList<>();
            //find it's local neighbours that are not in the tabuList.
            ArrayList<Pair<Integer, Integer>> neighbourhood = createNeighbourhood(shapelet);
            boolean inList = false;
            for(Pair<Integer, Integer> neighbour : neighbourhood){
                //i think if we collide with the tabuList we should abandon the neighbourhood.
                if(!tabuList.contains(neighbour))
                    candidateList.add(neighbour);
                else{
                    inList = true;
                    break;
                }
            }
            //if inList is true we want to abandon this whole search area.
            //reckon we need to increase evaluated counter by one, 
            //as a safety measure or we need to keep the tabuList smallish.
            if(inList){
                //numShapeletsEvaluated++;
                continue;
            }
            
            
            
            //find the best local candidate
            Pair<Integer, Integer> bestLocal = null;
            Shapelet local_bsf_shapelet = null;
            for(Pair<Integer, Integer> shape : candidateList ){
                Shapelet sh = checkCandidate.process(series, shape.var1, shape.var2);
                numShapeletsEvaluated++;

                //we've abandoned this shapelet, and therefore it is null.
                if(sh == null) continue;
                
                if(local_bsf_shapelet == null){
                    bestLocal = shape;
                    local_bsf_shapelet = sh;
                }
                
                
                //System.out.println(sh.length +" " + sh.startPos + " " +sh.qualityValue);
                
                //if the comparator says it's better.
                if(comparator.compare(local_bsf_shapelet, sh) < 0){
                    bestLocal = shape;
                    local_bsf_shapelet = sh;
                }
            }
            
            if(local_bsf_shapelet == null) continue;
            
            
            //update bsf shapelet if the local one is better.
            if(bsf_shapelet == null) bsf_shapelet = local_bsf_shapelet;
            
            if(comparator.compare(bsf_shapelet, local_bsf_shapelet) <= 0){
                bsf_shapelet = local_bsf_shapelet;
                seriesShapelets.add(local_bsf_shapelet); //stick the local best ones in the list.
            }          
            
            //add the best local to the TabuList
            tabuList.add(bestLocal);
            
            if(tabuList.size() > maxTabuSize){
                tabuList.remove();
            }
        }
        
        return seriesShapelets;
    }
    
    ArrayList<Pair<Integer,Integer>> createNeighbourhood(Pair<Integer,Integer> shapelet){
        return createNeighbourhood(shapelet, inputData.numAttributes()-1);
    }
    
    ArrayList<Pair<Integer,Integer>> createNeighbourhood(Pair<Integer,Integer> shapelet, int m){
        ArrayList<Pair<Integer,Integer>> neighbourhood = new ArrayList<>();
        neighbourhood.add(shapelet); //add the shapelet to the neighbourhood.
        
        int halfWidth = (int)((double)neighbourhoodWidth / 2.0);
                
        for(int pos= -halfWidth; pos <= halfWidth; pos++){
            for(int len= -halfWidth; len <= halfWidth; len++){
                if(len == 0 && pos == 0) continue;
                
                //need to prune impossible shapelets.
                int newLen = shapelet.var1 + len;
                int newPos = shapelet.var2 + pos;
                
                if(newLen < minShapeletLength || //don't allow length to be less than minShapeletLength. 
                   newLen > maxShapeletLength || //don't allow length to be more than maxShapeletLength.
                   newPos < 0                 || //don't allow position to be less than 0.               
                   newPos >= (m-newLen+1))       //don't allow position to be greater than m-l+1.
                    continue;
                
                neighbourhood.add(new Pair<>(newLen, newPos));
            } 
        }
        
        return neighbourhood;
    }
    
    private Pair<Integer, Integer> createRandomShapelet(Instance series){
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
        int position  = random.nextInt(series.numAttributes() - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
        return new Pair(length, position);   
    }
    
    
    public static void main(String[] args){
        //3 -> 100 series. 100 shapelets.
        
        //will aim to make a searchFactory so you dont hand build a searchFunction.
        ShapeletSearchOptions tabuOptions = new ShapeletSearchOptions.Builder().setMin(3).setMax(100).setNumShapelets(1000).setSeed(0).build();
        
        TabuSearch tb = new  TabuSearch(tabuOptions);
        //edge case neighbour hood testing
        //length of m, and 
        for (int len = 3; len < 100; len++) {
            for(int pos=0; pos< 100-len+1; pos++){
                ArrayList<Pair<Integer, Integer>> createNeighbourhood = tb.createNeighbourhood(new Pair<>(len, pos), 100);
                System.out.println(createNeighbourhood);
            }
        }

    }
}
