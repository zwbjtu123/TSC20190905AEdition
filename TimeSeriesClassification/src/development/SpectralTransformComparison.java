/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package development;


import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import statistics.simulators.SimulateAR;
import statistics.simulators.SimulatePowerSpectrum;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.FFT;
import weka.filters.timeseries.PowerCepstrum;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class SpectralTransformComparison extends Thread{
    int m;
    Classifier[] c;
    public static String path ="C:\\Users\\ajb\\Dropbox\\Results\\SpectralDomain\\Experiment1\\";

//Does the whole experiment for a certain length of series    
    public SpectralTransformComparison(int length){
        m=length;
    }
    public static int MIN_PARAS_EXP1=2;
    public static int MAX_PARAS_EXP1=10;
    
    
    public void run(){
//   Set up the       
            int nosCases=400;
            int[] nosCasesPerClass={nosCases/2,nosCases/2};
            int runs=50;
            int minParas=2;
            int maxParas=10;
             ArrayList<String> names=new ArrayList<>();
          Random rand = new Random();  
          c=ACFDomainClassification.setSingleClassifiers(names);           
          
           int length=m;
           try{
               int nosTrans=3;
                Instances[] train=new Instances[nosTrans];
                Instances[] test=new Instances[nosTrans];
                double[][] sum=new double[train.length][c.length];
                double[][] sumSq=new double[train.length][c.length];
                PowerSpectrum ps = new PowerSpectrum();
                PowerCepstrum pc= new PowerCepstrum();
                pc.useFFT();
                FFT fft=new FFT();
                
                OutFile of = new OutFile(path+"mean_"+m+".csv");
                OutFile of2 = new OutFile(path+"sd_"+m+".csv");
                System.out.println(" Running length ="+m);
                of.writeLine("classifier,PS,PC,FFT");
                of2.writeLine("classifier,PS,PC,FFT");
                
                
                for(int i=0;i<runs;i++){
                    //Generate data AND SET NOISE LEVEL
                    c=ACFDomainClassification.setSingleClassifiers(names);           
                    if(i%10==0)
                        System.out.println(" m ="+m+" performing run ="+i);
                    train = new Instances[nosTrans];
                    test = new Instances[nosTrans];
//Change to simulate sin waves.
                    Instances rawTrain=SimulatePowerSpectrum.generateFFTDataSet(minParas,maxParas,length,nosCasesPerClass,true);
                    rawTrain.randomize(rand);
                    Instances rawTest = new Instances(rawTrain,0);
                    for(int k=0;k<nosCases/2;k++){
                        Instance r=rawTrain.remove(0);
                        rawTest.add(r);
                    }
//Generate transforms                        
                   train[0]=ps.process(rawTrain);
                   train[1]=pc.process(rawTrain);
                   train[2]=fft.process(rawTrain);

                   test[0]=ps.process(rawTest);
                   test[1]=pc.process(rawTest);
                   test[2]=fft.process(rawTest);
//Measure classification accuracy
                    for(int j=0;j<test.length;j++){
                        for(int k=0;k<c.length;k++){
                            double a=ClassifierTools.singleTrainTestSplitAccuracy(c[k], train[j], test[j]);
                            sum[j][k]+=a;
                            sumSq[j][k]+=a*a;    
                        }
                    }
                }
                DecimalFormat df= new DecimalFormat("###.###");
                 System.out.print("\n m="+length);
                for(int j=0;j<c.length;j++){
                    of.writeString(names.get(j)+",");
                    of2.writeString(names.get(j)+",");
                    for(int i=0;i<test.length;i++){
                        sum[i][j]/=runs;
                        sumSq[i][j]=sumSq[i][j]/runs-sum[i][j]*sum[i][j];
                        System.out.print(","+df.format(sum[i][j])+" ("+df.format(sumSq[i][j])+")");
                        of.writeString(df.format(sum[i][j])+",");                    
                        of2.writeString(df.format(sumSq[i][j])+",");
                    }
                    of.writeString("\n");                        
                    of2.writeString("\n");  
                }
       }catch(Exception e){
               System.out.println(" Error ="+e);
               e.printStackTrace();
               System.exit(0);
           }        
                        

    }
    public static void main(String[] args){
        SpectralTransformComparison[] c= new SpectralTransformComparison[10];
        for(int i=0;i<10;i++){
            c[i]=new SpectralTransformComparison((i+1)*100);
            c[i].start();
        }
        try{
            for(int i=0;i<c.length;i++)
                c[i].join();
        
        }catch(Exception e){
            System.out.println(" Error "+e);
        }
    }

}
