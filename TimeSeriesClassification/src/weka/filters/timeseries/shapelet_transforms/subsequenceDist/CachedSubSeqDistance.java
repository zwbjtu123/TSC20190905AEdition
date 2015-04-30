/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.subsequenceDist;

/**
 *
 * @author raj09hxu
 */
public class CachedSubSeqDistance implements SubSequenceDistance{

    private double[] candidate;

    @Override
    public void setCandidate(double[] cnd) {
        candidate = cnd;
    }
    
    @Override
    public double calculate(double[] timeSeries) 
    {
        return calculate(timeSeries, 0); 
    }

    @Override
    public double calculate(double[] timeSeries, int startPos) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
