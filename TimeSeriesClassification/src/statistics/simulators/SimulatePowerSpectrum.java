/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package statistics.simulators;

import java.util.Random;
import weka.core.Instances;
import weka.filters.NormalizeCase;


/**
 *
 * @author ajb
 */
public class SimulatePowerSpectrum extends DataSimulator {
    //Models 

    int minWaves=1;
    int maxWaves=5;

    public SimulatePowerSpectrum(double[][] paras){
        super(paras);
        for(int i=0;i<nosClasses;i++)
            models.add(new SinusoidalModel(paras[i]));
    }

    public static Instances generateFFTDataSet(int minParas, int maxParas, int seriesLength, int[] nosCases, boolean normalize){
        double[][] paras=new double[nosCases.length][];
//Generate random parameters for the first FFT        
        Random rand= new Random();
        SinusoidalModel[] sm=new SinusoidalModel[nosCases.length];
        int modelSize=minParas+rand.nextInt(maxParas-minParas);
        paras[0]=new double[3*modelSize];
        for(int j=0;j<paras.length;j++)
             paras[0][j]=rand.nextDouble();
        for(int i=1;i<sm.length;i++){
            paras[i]=new double[3*modelSize];
            for(int j=0;j<paras.length;j++){
                paras[i][j]=paras[0][j];
//Perturb it 10%
                paras[i][j]+=-0.1+0.2*rand.nextDouble();
                if(paras[i][j]<0 || paras[i][j]>1)
                    paras[i][j]=paras[0][j];
            }
        }
        for(int i=0;i<sm.length;i++){
            sm[i]=new SinusoidalModel(paras[i]);
            sm[i].setFixedOffset(false);
        }
            
        
//        for(int i=0;i<paras.length;i++)
//            paras[i]=generateStationaryParameters(minParas,maxParas);
        DataSimulator ds = new DataSimulator(sm);
        ds.setSeriesLength(seriesLength);
        ds.setCasesPerClass(nosCases);
        Instances d=ds.generateDataSet();
        if(normalize){
        try{
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
            d=norm.process(d);
            }catch(Exception e){
                System.out.println("Exception e"+e);
                e.printStackTrace();
                System.exit(0);
            }
        }
        return d;
    }
    
    
    @Override
    public double[] generate(int length, int modelNos) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

  
    
    
}
