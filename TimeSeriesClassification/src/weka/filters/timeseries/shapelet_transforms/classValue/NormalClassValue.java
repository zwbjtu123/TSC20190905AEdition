/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.classValue;

import java.util.Map;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class NormalClassValue {
    
    double shapeletValue;
    Map<Double, Integer> classDistributions;
    
    public void init(Instances inst)
    {
        classDistributions = InstanceTools.createClassDistributions(inst);
    }
    
    public Map<Double, Integer> getClassDistributions()
    {
        return classDistributions;
    }
    
    //this will get updated as and when we work with a new shapelet.
    public void setShapeletValue(Instance shapeletSeries)
    {
        shapeletValue = shapeletSeries.classValue();
    }
    
    public double getClassValue(Instance in){
        return in.classValue();
    }
    
    public final double getUnAlteredClassValue(Instance in)
    {
        return in.classValue();
    }

    public double getShapeletValue() {
        return shapeletValue;
    }
    
}
