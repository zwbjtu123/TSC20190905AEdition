/*

Interval model for simulators
 */
package statistics.simulators;

/**
 *
 * @author ajb
 */
public class IntervalModel extends Model{
    int[][] intervals=new int[2][];  //areas of the series where therer are discriminatory features  
    int nosIntervals=1; //
    public IntervalModel(){
    }
    public IntervalModel(int n){
        nosIntervals=n;
        createIntervals();
    }
    public final void createIntervals(){
        intervals[0]=new int[nosIntervals];
        intervals[1]=new int[nosIntervals];
    }
    public final void createModels(){
        
    }

    @Override
    public void setParameters(double[] p) {
        nosIntervals=(int)p[0];
        
    }
}
