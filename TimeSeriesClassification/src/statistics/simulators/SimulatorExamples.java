/*
Class demonstrating how to use the data simulators to generate weka instances
 */
package statistics.simulators;

import java.util.ArrayList;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulatorExamples {
    
    public static Instances generateARDataSet(){
        return null;
    }
    
    public static void main(String[] args){
/**DataSimulator: All the simulators inherit from abstract DataSimulator
 * a DataSimulator contains an ArrayList of Models, one for each class
 * To create a data simulator, you can either pass it a 2D array of parameters
 * (one array for each class) or pass it an ArrayList of models 
 * (again, one for each class).
*/
        double[][] paras={{0.1,0.5,-0.6},{0.2,0.4,-0.5}};
// Creates a two class simulator for AR(3) models         
        DataSimulator arma=new SimulateAR(paras);
        
/* Model: All models inherit from the base Model class. Model has three abstract 
 * methods. generate: returns the next observation in the series, generate(t) 
 * generates the observation at time t (if possible) and generateSeries(int n), 
 * which calls generate n times and returns an array        */
        ArrayList<Model> m=new ArrayList<>();
        m.add(new ArmaModel(paras[0]));
        m.add(new ArmaModel(paras[1]));
   
/** Once you have created the simulator and/or the models, you can create sets 
 * of instances thus */
        int seriesLength=100;
        int[] casesPerClass={100,100};
        Instances data = arma.generateDataSet(seriesLength, casesPerClass);
        
    }
}

