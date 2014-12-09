/*
Shapelet Factory: determines shapelet type and parameters based on the input
* 1. distance caching: used if there is enough memory. 
*           Distance caching requires O(nm^2)  
* 2. number of shapelets:  
*           Set to n*m/10, with a max size max(1000,2*m,2*n) . Assume we will post process cluster
* 3. shapelet length range: 3 to train.numAttributes()-1
* 
 */
package weka.filters.timeseries.shapelet_transforms;
import weka.core.Instances;
import weka.core.shapelet.*;

/**
 *
 * @author ajb
 */
public class ShapeletTransformFactory {
    public static final double MEM_CUTOFF=0.5;
    public static final int MAX_NOS_SHAPELETS=1000;
    public FullShapeletTransform createTransform(Instances train){       
 //Memory in bytes available        
//1. distance caching or not 
        long mem=getAvailableMemory();
        System.out.println(" Memory asvailable ="+(mem/1000000)+" MB");
// Memory in bytes required by the distance Cache
        long distCache=train.numInstances()*(train.numAttributes()-1)*(train.numAttributes()-1);
//Currently the cache is in doubles, will convert to floats.
        distCache*=32;   //8 bytes per double, 4 for floats, but be conservative
        FullShapeletTransform s;
            s=new ShapeletTransform();
            System.out.println(" No Caching");
        if(distCache<MEM_CUTOFF*mem){   //Use caching
            s=new ShapeletTransformDistCaching();
            System.out.println(" Using Caching");
        }
        else{
            s=new ShapeletTransform();
            System.out.println(" No Caching");
        }
//2. Number of shapelets to retain
        int m=train.numAttributes()-1;
        int n=train.numInstances();
        
        int nosShapelets=n*m/10;
        if(nosShapelets<m)
            nosShapelets=m;
        else if(nosShapelets<n)
            nosShapelets=n;
        else if(nosShapelets>MAX_NOS_SHAPELETS)
            nosShapelets=MAX_NOS_SHAPELETS;
        
        long spareMem=mem-distCache;
        long memPerShapelet=16*(train.numAttributes()-1);
        System.out.println("Spare memory ="+(spareMem/1000000)+" shapelet memory required ="+(nosShapelets*memPerShapelet)/1000000);
        
        System.out.println("Generating "+nosShapelets+" Shapelets");
        s.setNumberOfShapelets(nosShapelets);        
//3. Shapelet length range, 
        int minLength=3;
        int maxLength=train.numAttributes()-1;
        s.setShapeletMinAndMax(minLength, maxLength);
        
//4. Shapelet quality measure 
        s.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);
        s.setCandidatePruning(true);

        
        return s;
    }
    long getAvailableMemory() {
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        long usedMemory = totalMemory - freeMemory;
        long availableMemory = maxMemory - usedMemory;
        return availableMemory;
}   
}
