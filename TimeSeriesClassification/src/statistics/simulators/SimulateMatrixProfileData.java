/*

 */
package statistics.simulators;

import development.DataSets;
import fileIO.OutFile;
import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulateMatrixProfileData {
    static DataSimulator sim;
    public static Instances generateMatrixProfileData(int seriesLength, int []casesPerClass)
    {
        MatrixProfileModel.setGlobalSeriesLength(seriesLength);
       
        MatrixProfileModel[] MP_Mod = new MatrixProfileModel[casesPerClass.length];
        populateMatrixProfileModels(MP_Mod,seriesLength); 
        sim = new DataSimulator(MP_Mod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;
    }
    private static void populateMatrixProfileModels(MatrixProfileModel[] m, int seriesLength){
        if(m.length!=2)
            System.out.println("ONLY IMPLEMENTED FOR TWO CLASSES");
//Create two models with same interval but different shape. 
        MatrixProfileModel m1=new MatrixProfileModel();
        MatrixProfileModel m2=new MatrixProfileModel();

        m[0]=m1;
        m[1]=m2;
        
    }
    public static void main(String[] args) {
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
        int seriesLength=500;
        int[] casesPerClass=new int[]{2,2};        
        Instances d=generateMatrixProfileData(seriesLength,casesPerClass);
        Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
        System.out.println(" DATA "+d);
        OutFile of = new OutFile("C:\\Temp\\intervalSimulationTest.csv");
//        of.writeLine(""+sim.generateHeader());
        of.writeString(split[0].toString());
        of = new OutFile("C:\\Temp\\intervalSimulationTrain.csv");
        of.writeString(split[1].toString());
    }
    
}
