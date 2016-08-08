/*
Bespoke split creator for Ethanol, enforces a leave-one-bottle out classifier

By default it does NOT remove the first attribute

*/
package applications;

import tsc_algorithms.FoldCreator;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class EthanolFoldCreator extends FoldCreator {
    
    @Override
    public Instances[] createSplit(Instances all, int rep) throws Exception{
        int nosDist=all.attribute(0).numValues();
        if(rep>=nosDist)
            throw new Exception(" ERROR, distillery index out of range");
        Instances[] split=new Instances[2];
        split[0]=new Instances(all);
        split[1]=new Instances(all,0);
    //Take all with position rep out of train, put into test
        int pos=0;
        while(pos<split[0].numInstances()){
            if(split[0].instance(pos).value(0)==rep)
                split[1].add(split[0].remove(pos));
            else
                pos++;
        }
        if(deleteFirstAttribute){
            split[0].deleteAttributeAt(0);
            split[1].deleteAttributeAt(0);
        }
        return split;
    }
}
