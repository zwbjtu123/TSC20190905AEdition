package timeseries_classifiers.cote;


/**
 *
 * @author sjx07ngu
 */
public interface HiveCoteModule{ 
        
    public double getEnsembleCvAcc();
    public double[] getEnsembleCvPreds();
    public String getParameters();
    
    
}
