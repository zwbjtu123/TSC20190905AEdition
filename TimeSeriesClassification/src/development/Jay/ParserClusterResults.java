
package development.Jay;

import development.DataSets;
import development.ElasticEnsembleCluster;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

/**
 *
 * @author jay
 * 
 * Temporary file for parsing the results of the EE on the cluster. Merge the code later with
 * ElasticEnsembleCluster (or create a single class for the work on the cluster, and tidy up EE)
 */
public class ParserClusterResults {
    
    
    
    public static void main(String[] args) throws Exception{
        
        String clusterDir = "/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/03_classification_loocvDatasets_testResults/eeClusterOutput_testResults/";
        
        // we want results for a dataset/classifier where 100 runs have been completed. Average accuracy and stdev
        //row: dataset; column: classifier - avgAcc, stdvAcc;
        ElasticEnsembleCluster.ClassifierVariants[] classifiers = ElasticEnsembleCluster.ClassifierVariants.values();
        
        StringBuilder output = new StringBuilder();
        output.append(",");
        for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
            output.append(classifier+",,");
        }
        output.delete(output.length()-2, output.length());
        output.append("\n");
        
        output.append("Dataset");
        for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
            output.append(",acc,stdev");
        }
        output.append("\n");
        
        double[] accs;
        File result;
        Scanner scan;
        boolean writeResult;
        double sum;
        double avg;
        double stdev;
        for(String dataset:DataSets.fileNames){
            output.append(dataset);
            for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
                writeResult = true;
                accs = new double[100];
                sum = 0;
                // only write to output if all 100 exist
                for(int f = 1; f <=100 && writeResult; f++){
                    
                    result = new File(clusterDir+dataset+"/"+dataset+"_"+f+"/"+dataset+"_"+f+"_"+classifier+".txt");
                    if(!result.exists()){
                        writeResult=false;
                        continue;
                    }
                    // file has accuracy (0-1) on the first row, every subsequent row is prediction/actual[prob_c1, prob_c2,...,prob_cn]
                    // we don't care about individual predictions at the minute, so just store the double in the first row
//                    scan = new Scanner()
                    scan = new Scanner(result);
                    scan.useDelimiter("\n");
                    accs[f-1] = Double.parseDouble(scan.next().trim());
                    sum+=accs[f-1];
                    scan.close();
                }
                if(writeResult){
                    avg = sum/100;
                    stdev = 0;
                    for(int i = 0; i < accs.length; i++){
                        stdev += (accs[i]-avg)*(accs[i]-avg);
                    }
                    stdev/=100;
                    stdev = Math.sqrt(stdev);
                    output.append(","+avg+","+stdev);
                }else{
                    // append a dash or something
                    output.append(",-,-");
                }
                
            }    
            output.append("\n");
        }
        
        FileWriter outFile = new FileWriter("test.csv");
        outFile.append(output);
        outFile.close();
        
    }
    
    
}
