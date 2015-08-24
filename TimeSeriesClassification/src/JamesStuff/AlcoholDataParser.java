
package JamesStuff;

import fileIO.InFile;
import fileIO.OutFile;

/**
 * UNTESTED as of 24/08
 * 
 * Code to parse the ifr alcohol data into arff formats.
 * 
 * Currently, it simply does the conversion based on the criteria/class labels provided ('Experiment') into a single dataset. i.e
 * does not split data into train and test datasets, only a single dump file
 * 
 * todo could make experiment etc into full class if wanted, could be useful elsewhere as a general raw data -> arff file parser
 * different experiments(i.e ways of grouping the data into files/different class values)? 
 *          -binary class problem of normal vs odd bottles regardless of contents
 *          -classifying based on individual bottle regardless of contents
 *          -binary classification of 'legal' vs 'illegal' regardless of specific concentration
 * etc
 * 
 * @author James
 */
public class AlcoholDataParser {
    
    public static final String masterPath = "C:\\\\Temp\\\\TESTDATA\\\\AlcoholData\\\\";
    public static final String ethPath = "Ethanol\\\\";
    public static final String methPath = "Methanol\\\\";
    
    public enum Experiment {
        ETHANOL("Ethanol4Class"), METHANOL("Methanol5Class");
        
        private final String experimentName;
        private Experiment (String en) {
            experimentName=en;
        }
        public String getName() { return experimentName; }
    }
    
    public static final String[/*Experiment*/][] expConcentrations = {  
        { ethPath+"E35", ethPath+"E38", ethPath+"E40", ethPath+"E45" },                //Ethanol
        { ethPath+"E40", methPath+"M05", methPath+"M1", methPath+"M2", methPath+"M5" } //Methanol
    };

    public static final String[/*Experiment*/][] expClassValues = {
        { "35%", "38%", "40%", "45%" },   //Ethanol
        { "0%", "0.5%", "1%", "2%", "5%"} //Methanol
    };

    public static final String[] bottles = {
        //"Normal" bottles
        "aberfeldy",
        "aberlour",
        "amrut",
        "ancnoc",
        "armorik",
        "arran10",
        "arran14",
        "asyla",
        "benromach",
        "bladnoch",
        "blairathol",
        "exhibition",
        "glencadam",
        "glendeveron",
        "glenfarclas",
        "glenfiddich",
        "glengoyne",
        "glenlivet15",
        "glenmorangie",
        "glenmoray",
        "glenscotia",
        "oakcross",
        "organic",
        "peatmonster",
        "scapa",
        "smokehead",
        "speyburn",
        "spicetree",
        "taliskerenglishwhisky9",
            
        //"Odd" bottles
        "allmalt",
        "auchentoshan",
        "balblair",
        "cardhu",
        "elijahcraig",
        "englishwhisky15",
        "greatkingst",
        "highlandpark",
        "mackmyra",
        "nikka",
        "bernheim",
        "dutchsinglemalt",
        "finlaggan",
        "glenlivet12",
        "laphroaig"
    };
    
    public static void main(String[] args) {
        for (Experiment exp : Experiment.values()) {
            OutFile out = new OutFile(exp.getName() + ".arff");

            writeMetaData(out, exp.getName(), expClassValues[exp.ordinal()]);

            for (int con = 0; con < expConcentrations[exp.ordinal()].length; con++)
                for (int bottle = 0; bottle < bottles.length; ++bottle)
                    for (int batch = 1; batch <= 3; batch++)
                        for (int rep = 1; rep <= 3; rep++)
                            SSMFileToARFF(out, expConcentrations[exp.ordinal()][con], con, bottle, batch, rep);

            out.closeFile();
        }
    }
    
    public static void writeMetaData(OutFile out, String relationName, String[] classValues) {
        out.writeLine("@Relation " + relationName);
        out.writeLine("");
        
        writeBottleAttribute(out);
        writeWaveLengthAttributes(out);
        writeClassAttribute(out, classValues);
        
        out.writeLine("");
        out.writeLine("@Data");
    }
    
    public static void writeBottleAttribute(OutFile out) {
        out.writeLine("@attribute bottleName {");
        for (String bottle : bottles) 
            out.writeString(bottle + ",");
        out.writeLine("}");
    }
    
    public static void writeWaveLengthAttributes(OutFile out) {
        SSMFileReader testFile = new SSMFileReader(masterPath + "testFile.SSM");
        while (!testFile.eof())
            out.writeLine("@attribute wavelength_" + testFile.nextWavelength());
        testFile.closeFile();
    }
    
    public static void writeClassAttribute(OutFile out, String[] classValues) {
        out.writeLine("@attribute classValues {");
        for (String val : classValues) 
            out.writeString(val + ",");
        out.writeLine("}");
    }
    
    public static void SSMFileToARFF(OutFile out, String concentration, int classValue, int bottle, int batch, int rep) {
        String path = masterPath + concentration + "_" + bottles[bottle] + "_" + batch + "_" + rep + ".SSM";
        SSMFileReader reading;

        try {
            reading = new SSMFileReader(path);
        } catch (Exception e) {
            System.out.println("Error opening file: " + path + "\n" + e);
            return; //reading may not exist (may have missed / overwritten it etc), not necessarily incorrect logic 
        }

        try {
            out.writeString(bottle + ",");

            while (!reading.eof()) {
                out.writeDouble(reading.nextValue());
                out.writeString(",");
            }

            out.writeInt(classValue); 
            out.newLine();
        } catch (Exception e) {
            System.out.println("Error writing to file: " + path);
            throw e;
        }

        reading.closeFile();
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
        
        public double nextWavelength() {
            double wl = in.readDouble();
            in.readDouble();
            return wl;
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
        
        public void closeFile() {
            in.closeFile();
        }
    }
    
}
