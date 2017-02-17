/*
 Adjusted Rotation Forest. To do:

1. Limit the max number of attributes per tree
    Test 1: make sure it still does the same thing when maxNumAttributes> numAtts in all cases
10/2/17: Run TunedRotationForest and EnhancedRotationForest with maxNumAttributes=10000 
should be no difference: 

    Test 2: check it still runs with problems where maxNumAttributes> numAtts
    Test 3: Compare accuracy on problems where maxNumAttributes> numAtts
    Test 4: Perform timing experiment on problems where maxNumAttributes> numAtts

2. Impose bagging and work out OOB Error
 */
package development;

import java.util.Random;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.TunedRotationForest;

/**
 *
 * @author ajb
 */
public class EnhancedRotationForest extends TunedRotationForest{
    private int maxNumAttributes=100;
    
    public void setMaxNumAttributes(int m){
        maxNumAttributes=m;
    }
    @Override
    protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                         Random random) {

    int [] permutation = new int[numAttributes-1];
    int i = 0;
//This is a bit weird, not sure what it is doing here.ClassAttribute    
    for(; i < classAttribute; i++){
      permutation[i] = i;
    }
    for(; i < permutation.length; i++){
      permutation[i] = i + 1;
    }

    permute( permutation, random );
    if(numAttributes>maxNumAttributes){
//TRUNCTATE THE PERMATION TO CONSIDER maxNumAttributes. This is not done in official version
       int[] temp = new int[maxNumAttributes];
       System.arraycopy(permutation, 0, temp, 0, maxNumAttributes);
       permutation=temp;
    }
    
    return permutation;
  }    
    public static void main(String[] args){
        
    }
}
