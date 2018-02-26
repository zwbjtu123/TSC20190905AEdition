/*
Multivariate data can be stored in Wekas "multi instance" format
https://weka.wikispaces.com/Multi-instance+classification

for TSC, the basic univariate syntax is 

 */
package development;

import fileIO.OutFile;
import utilities.ClassifierTools;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Sorting out the new archive
 * @author ajb
 */
public class MultiVariateProcessing {
    
    
    public static void main(String[] args) throws Exception {
        //gettingStarted();
        mergeEpilepsy();
    }
    public static void mergeEpilepsy(){
        Instances x,y,z;
        Instances all;
        String sourcePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\EpilepsyX\\";
        String destPath="C:\\Users\\ajb\\Dropbox\\Multivariate TSC Problems\\HAR\\Epilepsy\\";
        x=ClassifierTools.loadData(sourcePath+"EpilepsyX_ALL");
        y=ClassifierTools.loadData(sourcePath+"EpilepsyY_ALL");
        z=ClassifierTools.loadData(sourcePath+"EpilepsyZ_ALL");
//Delete the use ID, will reinsert manually after        
        x.deleteAttributeAt(0);
        y.deleteAttributeAt(0);
        z.deleteAttributeAt(0);
        all=utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(new Instances[]{x,y,z});
//        OutFile out=new OutFile(destPath+"EpilepsyNew.arff");
//        out.writeString(all.toString());
 //Create train test splits so participant 1,2,3 in train and 4,5,6 in test       
        int trainSize=149;
        int testSize=126;
        Instances train= new Instances(all,0);
        Instances test= new Instances(all);
        for(int i=0;i<trainSize;i++){
            Instance t= test.remove(0);
            train.add(t);
        }
        OutFile tr=new OutFile(destPath+"Epilepsy_TRAIN.arff");
        OutFile te=new OutFile(destPath+"Epilepsy_TEST.arff");
        tr.writeString(train.toString());
        te.writeString(test.toString());
        
        
    }
    
/**A getting started with relational attributes in Weka. Once you have the basics
 * there are a range of tools for manipulating them in 
 * package utilities.multivariate_tools 
 * 
 * See https://weka.wikispaces.com/Multi-instance+classification
 * for more     
 * */
    public static void gettingStarted(){
//Load a multivariate data set
        String path="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate\\univariateConcatExample";
        Instances train = ClassifierTools.loadData(path);
        System.out.println(" univariate data = "+train);
        path="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate\\multivariateConcatExample";
        train = ClassifierTools.loadData(path);
        System.out.println(" multivariate data = "+train);
//Recover the first instance
        Instance first=train.instance(0);
//Split into separate dimensions
        Instances split=first.relationalValue(0);
        System.out.println(" A single multivariate case split into 3 instances with no class values= "+split);
        for(Instance ins:split)
            System.out.println("Dimension of first case =" +ins);
//Extract as arrays
        double[][] d = new double[split.numInstances()][];
        for(int i=0;i<split.numInstances();i++)
           d[i]=split.instance(i).toDoubleArray();

    
    }
}
