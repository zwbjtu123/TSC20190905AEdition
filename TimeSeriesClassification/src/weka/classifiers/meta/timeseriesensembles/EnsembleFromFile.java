package weka.classifiers.meta.timeseriesensembles;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import utilities.DebugPrinting;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instances;

/**
 * Given a results directory and classifiers to use (the results for which lie in that directory),
 * will ensemble those classifier's classifications to form a collective classification
 * 
 * By default, will use each member's train accuracy as it's vote weighting, and simple 
 * majority voting to form an ensemble classification
 * 
 * Unless a specific name for the ensemble is provided, will attempt to construct a name
 * by concatenating the names of its constituents. ensembleIdentifier is (as of writing) 
 * only used for saving the cv accuracy of the ensemble (for SaveCVAccuracy)
 * 
 * @author James Large
 */
public abstract class EnsembleFromFile extends AbstractClassifier implements DebugPrinting {
    
    //results file reading/writing
    protected boolean resultsFilesParametersInitialised;
    protected String individualResultsFilesDirectory;
    protected String ensembleIdentifier = "EnsembleFromFile";
    protected int resampleIdentifier;

    protected String datasetName;
    protected String[] classifierNames;

    /**
     * must be called in order to build ensemble from results files or to write individual's 
     * results files
     * 
     * exitOnFilesNotFound defines whether the ensemble will simply throw exception/exit if results files 
     * arnt found, or will try to carry on (e.g train the classifiers normally)
     */
    public void setResultsFileLocationParameters(String individualResultsFilesDirectory, String datasetName, int resampleIdentifier) {
        resultsFilesParametersInitialised = true;
        
        this.individualResultsFilesDirectory = individualResultsFilesDirectory;
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
    }
    
    
    protected File findResultsFile(String classifierName, String trainOrTest) {
        File file = new File(individualResultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"/"+trainOrTest+"Fold"+resampleIdentifier+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else 
            return file;
    }
    
    protected static ModuleResults loadResultsFile(File file, int numClasses) throws Exception {    
                
        ArrayList<Double> alpreds = new ArrayList<>();
        ArrayList<Double> alclassvals = new ArrayList<>();
        ArrayList<ArrayList<Double>> aldists = new ArrayList<>();
        
        Scanner scan = new Scanner(file);
        scan.useDelimiter("\n");
        scan.next();
        scan.next();
        double acc = Double.parseDouble(scan.next().trim());
        
        String [] lineParts = null;
        while(scan.hasNext()){
            lineParts = scan.next().split(",");

            if (lineParts == null || lineParts.length < 2)
                continue;
            
            alclassvals.add(Double.parseDouble(lineParts[0].trim()));
            alpreds.add(Double.parseDouble(lineParts[1].trim()));
            
            if (lineParts.length > 3) {//dist for inst is present
                ArrayList<Double> dist = new ArrayList<>();
                for (int i = 3; i < lineParts.length; ++i)  //act,pred,[empty],firstclassprob.... therefore 3 start
                    if (lineParts[i] != null && !lineParts[i].equals("")) //may have an extra comma on the end...
                        dist.add(Double.parseDouble(lineParts[i].trim()));
                aldists.add(dist);
            }
        }
        
        scan.close();
        
        validateResultsFile(file, acc, alclassvals, alpreds, aldists, numClasses);
        
        double [] classVals = new double[alclassvals.size()];
        for (int i = 0; i < alclassvals.size(); ++i)
            classVals[i]= alclassvals.get(i);
        
        double [] preds = new double[alpreds.size()];
        for (int i = 0; i < alpreds.size(); ++i)
            preds[i]= alpreds.get(i);
        
        double[][] distsForInsts = null;
        if (!aldists.isEmpty()) {
            distsForInsts = new double[aldists.size()][aldists.get(0).size()];
            for (int i = 0; i < aldists.size(); ++i)
                for (int j = 0; j < aldists.get(i).size(); ++j) 
                    distsForInsts[i][j] = aldists.get(i).get(j);
        }
        
        return new ModuleResults(acc, classVals, preds, distsForInsts, numClasses);
    }
    
    private static void validateResultsFile(File file, double acc, ArrayList<Double> alclassvals, ArrayList<Double> alpreds, ArrayList<ArrayList<Double>> aldists, int numClasses) throws Exception {
        if (!aldists.isEmpty()) {
            if (alpreds.size() != aldists.size())
                throw new Exception("validateResultsFile Exception: "
                        + "somehow have different number of predictions and distForInstances: " + file.getAbsolutePath());

            for (ArrayList<Double> dist : aldists)
                if (dist.size() != numClasses)
                    throw new Exception("validateResultsFile Exception: "
                            + "instance reports different numbers of classes: " + file.getAbsolutePath());
        }
        
        double count = 0.0;
        for (int i = 0; i < alpreds.size(); ++i)
            if (alpreds.get(i).equals(alclassvals.get(i)))
                count++;
        
        double a = count/alpreds.size();
        if (a != acc)
            throw new Exception("validateResultsFile Exception: "
                    + "incorrect accuracy (" + acc + "reported vs" +a +"actual) reported in: " + file.getAbsolutePath());
    }
    
    protected void writeResultsFile(String classifierName, String parameters, ModuleResults results, String trainOrTest) throws IOException {
        printlnDebug(classifierName + " " + trainOrTest + " writing...");
                
        StringBuilder st = new StringBuilder();
        st.append(this.datasetName).append(",").append(this.ensembleIdentifier).append(classifierName).append(","+trainOrTest+"\n");
        st.append(parameters + "\n"); //st.append("internalHesca\n");
        st.append(results.acc).append("\n");
        
        for(int i = 0; i < results.predClassVals.length;i++) {
            st.append(results.trueClassVals[i]).append(",").append(results.predClassVals[i]).append(","); //pred
            
            if (results.distsForInsts != null && results.distsForInsts[i] != null)
                for (int j = 0; j < results.distsForInsts[i].length; j++)
                    st.append("," + results.distsForInsts[i][j]);
            
            st.append("\n");
        }
        
        String fullPath = this.individualResultsFilesDirectory+classifierName+"/Predictions/"+datasetName;
        new File(fullPath).mkdirs();
        FileWriter out = new FileWriter(fullPath+"/" + trainOrTest + "Fold"+this.resampleIdentifier+".csv");
        out.append(st);
        out.close();
    }

    
    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
