/*

 */
package statistics.simulators;

import development.DataSets;
import fileIO.OutFile;
import timeseriesweka.classifiers.DTW_1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.filters.MatrixProfile;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.filters.NormalizeCase;

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
        populateMatrixProfileModels(MP_Mod); 
        sim = new DataSimulator(MP_Mod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;
    }
    private static void populateMatrixProfileModels(MatrixProfileModel[] m){
        if(m.length!=2)
            System.out.println("ONLY IMPLEMENTED FOR TWO CLASSES");
//Create two models with same interval but different shape. 
        MatrixProfileModel m1=new MatrixProfileModel();
        MatrixProfileModel m2=new MatrixProfileModel();

        m[0]=m1;
        m[1]=m2;
        
    }
    
    private static void test1NNClassifiers(){
        for(double sig=0;sig<=1;sig+=1){
            Model.setDefaultSigma(sig);
            double meanAcc=0;
            double meanAcc2=0;
            int r=20;
            for(int i=0;i<r;i++){
                Model.setGlobalRandomSeed(i);
                int seriesLength=200;
                int[] casesPerClass=new int[]{100,100};        
                Instances d=generateMatrixProfileData(seriesLength,casesPerClass);
                Instances[] split=InstanceTools.resampleInstances(d, 0,0.1);
                kNN knn= new kNN();
                knn.setKNN(1);
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(knn, split[0], split[1]);
                DTW1NN dtw=new DTW1NN();
                double acc2=ClassifierTools.singleTrainTestSplitAccuracy(dtw, split[0], split[1]);
                meanAcc+=acc;
                meanAcc2+=acc2;
                System.out.println("Train Size ="+split[0].numInstances()+" 1NN acc = "+acc+" DTW acc ="+acc2);
            }
            System.out.println(" Sig ="+sig+" Mean 1NN Acc ="+meanAcc/r+" Mean 1NN Acc ="+meanAcc2/r);
        }
        
    }
    public static void main(String[] args) throws Exception {
//        test1NNClassifiers();
//        System.exit(0);
        
        Model.setDefaultSigma(0.1);
        Model.setGlobalRandomSeed(0);
        int seriesLength=500;
        int[] casesPerClass=new int[]{10,10};        
        NormalizeCase nc=new NormalizeCase();
        Instances d=generateMatrixProfileData(seriesLength,casesPerClass);
        Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
        MatrixProfile mp=new MatrixProfile(29);
        Instances m1=mp.process(split[0]);
//        m1=nc.process(m1);
        OutFile of = new OutFile("C:\\Temp\\MPSimulationTrain.csv");
//        of.writeLine(""+sim.generateHeader());
        of.writeString(split[0].toString()+"\n\n");
        of.writeString(m1.toString());
        of = new OutFile("C:\\Temp\\MPSimulationTrain.arff");
        of.writeString(split[0]+"");
        Instances m2=mp.process(split[1]);
 //       m2=nc.process(m2);
        of = new OutFile("C:\\Temp\\MPSimulationTest.csv");
        of.writeString(split[1].toString()+"\n\n");
        of.writeString(m2.toString());
    }
    
}