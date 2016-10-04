/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package statistics.simulators;

/**
 *
 * @author ajb
 */
public class SimulateWholeSeriesElasticData extends DataSimulator{
    public SimulateWholeSeriesElasticData(double[][] paras){
        super(paras);
        for(int i=0;i<nosClasses;i++)
            models.add(new SinusoidalModel(paras[i]));
    }
      public void setWarping(){
          for(Model m:models){
              ((SinusoidalModel)m).setWarp(true);
          }
      }

}
