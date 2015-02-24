/*
1. Ties?
2. Normalization
3. 
*/
package development;

import fileIO.OutFile;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.*;
import weka.core.*;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class ElasticDistanceMeasures {
    public static void testUCR() throws Exception{   //EuclideanDistance d
        OutFile of=new OutFile(DataSets.ucrPath+"UCRResults.csv");//d.getClass().getName()+
 //       for(String s:DataSets.ucrNames){
            String s="NonInvasiveFatalECG_Thorax1";
            Instances train = ClassifierTools.loadData(DataSets.ucrPath+s+"\\"+s+"V2_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.ucrPath+s+"\\"+s+"V2_TEST");
            NormalizeCase nc = new NormalizeCase();
    //        train=nc.process(train);
    //        test=nc.process(test);
            kNN knn= new kNN(1);
            knn.normalise(false);
            Classifier ib1= new IBk();
            double a1=ClassifierTools.singleTrainTestSplitAccuracy(knn, train, test);
            double a2=ClassifierTools.singleTrainTestSplitAccuracy(ib1, train, test);
            double a3=nearestNeighbourAcc(train,test);
            System.out.println(s+" train size"+train.numInstances()+" test size ="+test.numInstances()+" series length "+(test.numAttributes()-1)+" 1nn ="+a1+" 1bk = "+a2+" strippedNN ="+a3);
            of.writeLine(s+","+(1-a1)+","+(1-a2)+","+(1-a3));
 //       }
    }
    static double distance(Instance a, Instance b){
        double[] d1=a.toDoubleArray();
        double[] d2=b.toDoubleArray();
        double dist=0;
        for(int i=0;i<d1.length-1;i++)
            dist+=(d1[i]-d2[i])*(d1[i]-d2[i]);
        return Math.sqrt(dist);
    }
    static int nearestNeighbour(Instances train, Instance test){
        double d;
        double minDist=Double.MAX_VALUE;
        int predClass=0;
        for(Instance tr:train){
            d=distance(tr,test);
            if(d<=minDist){
                minDist=d;
                predClass=(int)tr.classValue();
            }
        }
        return predClass;
    }
    static double nearestNeighbourAcc(Instances train, Instances test){
        double acc=0;
        int correct=0;
        for(Instance ins:test){
            int pred=nearestNeighbour(train,ins);
          // System.out.println(" pred ="+pred);
            if(pred==ins.classValue())
                correct++;
        }
        acc=correct/(double)test.numInstances();
        return acc;
    }
    
    public static void main(String[] args) throws Exception{
        testUCR();//new EuclideanDistance()
    }
}
