/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package statistics.simulators;

import fileIO.OutFile;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulateDictionaryData {
    int[] shapeletsPerClass={1,1};
    
    
    /**
     * This method creates and returns a set of Instances representing a
     * simulated two-class time-series problem.
     * 

     * @param seriesLength The length of the series. All time series in the
     * dataset are the same length.
     * @param casesPerClass An array of two integers indicating the number of 
     * instances of class 0 and class 1.
     * @return Instances representing the time-series dataset. The Instances
     * returned will be empty if the casesPerClass parameter does not contain
     * exactly two values.
     */
    public static Instances generateDictionaryData(int seriesLength, int []casesPerClass)
    {
        
        if( casesPerClass.length != 2)
        {
            System.err.println("Incorrect parameters, dataset will not be co"
                    + "rrect.");
            int[] tmp = {0,0};
            casesPerClass = tmp;
            
        }
        ShapeletModel[] shapeMod = new ShapeletModel[casesPerClass.length];
        populateRepeatedShapeletArray(shapeMod, seriesLength);
//        for(ShapeletModel s:shapeMod)
//            System.out.println("Shapel Model "+s);
        DataSimulator sim = new DataSimulator(shapeMod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;          
    }
   /**
     * This is a support method for generateShapeletData
     * 
     * @param array An array of two ShapeletModel2 models, representing the 
     * simulated shapes inserted into the respective classes.
     * @param seriesLength The length of the series.
     */
    private static void populateRepeatedShapeletArray(ShapeletModel [] s, int seriesLength)
    {
        
        double[] p1={seriesLength,2};//CHANGE TO AVOID HARD CODING!
        double[] p2={seriesLength,3};
//Create two ShapeleModels with the same base Shapelet        
//        ShapeletModel.DEFAULTSHAPELETLENGTH=13;
        s[0]=new ShapeletModel(p1);        
        s[1]=new ShapeletModel(p2);
        
        ShapeletModel.ShapeType st=s[0].getShapeType();
        s[0].setShapeType(st);
        s[1].setShapeType(st);
    }
    public static void main(String[] args) {
        Model.setDefaultSigma(0);
//seriesLength=1000;
//                casesPerClass=new int[]{20,20};        
        Instances d=generateDictionaryData(1000,new int[]{20,20});
        System.out.println(" DATA "+d);
        OutFile of = new OutFile("C:\\Temp\\dict.csv");
        of.writeString(d.toString());
    }
        
}
