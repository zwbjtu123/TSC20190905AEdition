/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import multivariate_timeseriesweka.classifiers.NN_DTW_D;
import multivariate_timeseriesweka.classifiers.NN_DTW_I;
import weka.classifiers.Classifier;

/**
 *
 * @author raj09hxu
 */
public class DefaultClassifiers {
    
        
    public static final Map<String, Supplier<Classifier>> CLASSIFIERS;
    static {
        Map<String, Supplier<Classifier>> map = new HashMap();
        map.put("DTW_A", DefaultClassifiers::createDTW_A);
        map.put("DTW_D", DefaultClassifiers::createDTW_D);
        map.put("DTW_I", DefaultClassifiers::createDTW_I);
        CLASSIFIERS = Collections.unmodifiableMap(map);
    }
    
    public static Classifier createDTW_A(){
        NN_DTW_A A = new NN_DTW_A();
        A.setR(0.2); //20%
        return A;
    }
    
    public static Classifier createDTW_I(){
       NN_DTW_I nn = new NN_DTW_I();
       nn.setR(0.2);
       return nn;
    }
    
    public static Classifier createDTW_D(){
       NN_DTW_D nn = new NN_DTW_D();
       nn.setR(0.2);
       return nn;
    }
    
}
