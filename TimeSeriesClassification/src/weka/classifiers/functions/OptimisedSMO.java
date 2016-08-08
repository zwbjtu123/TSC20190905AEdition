/*
Tony's attempt to see the effect of parameter setting on SVM.

Two parameters: 
kernel para: for polynomial this is the weighting given to lower order terms
    k(x,x')=(<x'.x>+a)^d
regularisation parameter, used in the SMO 

m_C

*/
package weka.classifiers.functions;

import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class OptimisedSMO extends SMO{
    
    
    @Override   
    public void buildClassifier(Instances data){
//If the current kernel is polynomial, switch to parameterised one
        if(m_kernel instanceof PolyKernel){
            double exp=((PolyKernel)m_kernel).getExponent();
            PolynomialKernel pk= new PolynomialKernel();
            pk.setExponent(exp);
            m_kernel=pk;
        }
//Very niave implementation for grid search.         
        
    }
    protected class PolynomialKernel extends PolyKernel{
        double a=0; //Parameter
    @Override   
    protected double evaluate(int id1, int id2, Instance inst1)
        throws Exception {

        double result;
        if (id1 == id2) {
          result = dotProd(inst1, inst1);
        } else {
          result = dotProd(inst1, m_data.instance(id2));
        }
//Only change from base class        
        result += a;
        if (m_exponent != 1.0) {
          result = Math.pow(result, m_exponent);
        }
        return result;
      }

    
}
    
}
