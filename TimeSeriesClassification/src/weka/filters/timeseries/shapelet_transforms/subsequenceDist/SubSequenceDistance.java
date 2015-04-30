/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.subsequenceDist;

/**
 *
 * @author raj09hxu
 * @param <T>
 */
public interface SubSequenceDistance {
    public static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
    public void setCandidate(double[] candidate);
    public double calculate(double[] timeSeries);
    public double calculate(double[] timeSeries, int startPos);
};

