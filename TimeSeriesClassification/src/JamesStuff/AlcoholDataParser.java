
package JamesStuff;

import fileIO.InFile;
import fileIO.OutFile;

/**
 * Just some code to parse the alcohol data to csv and weka formats 
 * 
 * @author James
 */
public class AlcoholDataParser {
    
    public static final String masterPath = "C:\\\\Temp\\\\TESTDATA\\\\AlcoholData\\\\";
    public static final String ethPath = "Ethanol\\\\";
    public static final String methPath = "Methanol\\\\";
    
    public static final String[] ethExpConcentrations = { 
        ethPath+"E35", ethPath+"E38", ethPath+"E40", ethPath+"E45"
    };
    
    public static final String[] methExpConcentrations = { 
        ethPath+"E40", methPath+"M05", methPath+"M1", methPath+"M2", methPath+"M5"
    };
    
    public static final String[] ethExpClassValues = {
        "35%", "38%", "40%", "45%"
    };
    
    public static final String[] methExpClassValues = {
        "0%", "0.5%", "1%", "2%", "5%"
    };
    
    public static final String[] bottles = {
        
    };
    
    public static void main(String[] args) {
        //target output: ethanol classification
        //all ethanol readings, paired with class values, in single file
        //e35, e38, e40, e45
        
        
        
        OutFile ethOut = new OutFile("ethanol4class");
        
        ethOut.writeLine("@Relation Ethanol4Class");
        ethOut.writeLine("");
        
        //todo build attributes, may need to do dummy read or something just to loop through waelength values 
        for (int i = 0; i < numAtts; i++) {
            ethOut.writeLine("@attribute ")
        }
        
        for (int con = 0; con < ethExpConcentrations.length; con++) { 
            for (int batch = 1; batch <= 3; batch++) {
                for (int rep = 1; rep <= 3; rep++) {
                    String path = masterPath + ethExpConcentrations[con] + "_" + batch + "_" + rep + ".SSM";
                    SSMFileReader reading = new SSMFileReader(path);
                    
                    while (!reading.eof()) {
                        ethOut.writeDouble(reading.nextValue());
                        ethOut.writeString(",");
                    }
                    
                    ethOut.writeInt(con); //class value
                    ethOut.newLine();
                }
            }
        }
        
        
        
        
        //target output: methanol classification
        //all meth readings, paired with calss values, in single file
        //e40 (aka m0), m05 (aka m0.5), m1, m2, m5
    }
    
    
    public static class SSMFileReader {
        String file;
        String metaData;
        double[] data;
        
        InFile in;
        
        public SSMFileReader(String file) {
            this.file = file;
            
            //add error checking blabla
            //correct for actual file structure blabla
            in = new InFile(file);
            
            in.readLine(); //file path/name again
            metaData = in.readLine(); //actual meta data, expand this later
        }
        
        public double nextValue() {
            in.readDouble(); //wavelength val
            return in.readDouble();
        }
        
        public double[] nextReading() {
            double r[] = new double[2];
            r[0] = in.readDouble();
            r[1] = in.readDouble();
            return r;
        }
        
        public boolean eof() {
            return in.isEmpty();
        }
    }
    
}
