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
public class WhiteNoiseModel extends  Model{

    public WhiteNoiseModel(){
        super();
    }

    @Override
    public void setParameters(double[] p) {//Mean and variance of the noise
        setVariance(p[0]);
        
    }
    
    
}
