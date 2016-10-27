/*
Written by Jon Hills

Model to simulate data where shapelet approach should be optimal.

*/
package statistics.simulators;
import fileIO.OutFile;
import java.util.*;
import java.io.*;
import statistics.distributions.NormalDistribution;
public class ShapeletModel extends Model {
    
    public enum ShapeType {TRIANGLE,HEADSHOULDERS,SINE, STEP, SPIKE};
    protected ArrayList<Shape> shapes;
    
    private static int DEFAULTNUMSHAPELETS=1;
    private static int DEFAULTSERIESLENGTH=500;
    private static int DEFAULTSHAPELETLENGTH=29;
    private static int DEFAULTBASE=-2;
    private static int DEFAULTAMP=4;
   
    protected int numShapelets;
    protected int seriesLength; 
    protected int shapeletLength;
    
    
    
    
    //Default Constructor, max start should be at least 29 less than the length
    // of the series if using the default shapelet length of 29
    public ShapeletModel()
    {
        this(new double[]{DEFAULTNUMSHAPELETS,DEFAULTSERIESLENGTH,DEFAULTSHAPELETLENGTH});
    }
    public final void setDefaults(){
       seriesLength=DEFAULTSERIESLENGTH; 
       numShapelets=DEFAULTNUMSHAPELETS;
       shapeletLength=DEFAULTSHAPELETLENGTH;
    }
    public ShapeletModel(double[] param)
    {
        super();
        setDefaults();
//PARAMETER LIST: seriesLength,  numShapelets, shapeletLength, maxStart
//Using the fall through for switching, I should be shot!        
        if(param!=null){
            switch(param.length){
                default:
                case 3:             shapeletLength=(int)param[2];
                case 2:             numShapelets=(int)param[1];
                case 1:             seriesLength=(int)param[0];
            }
        }
        shapes=new ArrayList<>();
        // Shapes are randomised for type and location; the other characteristics, such as length
        // must be changed in the inner class.
        for(int i=0;i<numShapelets;i++)
        {
           Shape sh = new Shape();
           sh.randomiseShape();
           shapes.add(sh); 
        }
//        error=new NormalDistribution(0,1);    
    }
    //This constructor is used for data of a given length
    public ShapeletModel(int s)
    {
        this(new double[]{(double)s});
    }
    
    //This constructor is used for data of a given length in a two class problem
    //where the shape distinguishing the first class is known
    public ShapeletModel(int seriesLength,Shape shape)
    {
        setDefaults();
        shapes=new ArrayList<Shape>();
        
        for(int i=0;i<numShapelets;i++)
        {
           Shape sh = new Shape();
           sh.randomiseShape();
           shapes.add(sh); 
        }       

    }
    // This constructor accepts an ArrayList of shapes for the shapelet model,
    // rather than determining the shapes randomly.
    public ShapeletModel(ArrayList<Shape> s)
    {
        shapes=new ArrayList<Shape>(s);
    }

     @Override   
	public	double[] generateSeries(int n)
	{
           t=0;
//Randomize the starting locations each time this is called
           for(Shape s:shapes)
               s.randomiseLocation();
           double[] d = new double[n];
           for(int i=0;i<n;i++)
              d[i]=generate();
           return d;
        }
    
    
    
//Fix all shapelets to a single type
    
    /*Generate a single data
//Assumes a model independent of previous observations. As
//such will not be relevant for ARMA or HMM models, which just return -1.
* Should probably remove. 
*/
    @Override
	public double generate(double x)
        {
            double value=error.simulate();
            //Slightly inefficient for non overlapping shapes, but worth it for clarity and generality
            for(Shape s:shapes)
                value+=s.generate((int)x);
                
            return value;
        }

//This will generate the next sequence after currently stored t value
    @Override
	public double generate()
        {
//            System.out.println("t ="+t);
            double value=generate(t);
            t++;
            return value;
        }
    
    /**
 * Subclasses must implement this, how they take them out of the array is their business.
 * @param p 
 */ 
    @Override
    public void setParameters(double[] p)
    {
    }
    
    // The implementation of the reset sets t back to zero.
       public void reset(){
        t=0;
    }
    
    public ShapeType getShapeType(){
        return shapes.get(0).type;
    }
    public void setShapeType(ShapeType st){
        for(Shape s:shapes){
            s.setType(st);
        }
    }
    
    // The toString() method has not been changed.
    @Override
    public String toString(){
        String str= "nos shapes = "+shapes.size()+"\n";
        for(Shape s:shapes)
            str+=s.toString()+"\n";
        return str;
    }
    @Override
    public String getModelType(){ return "ShapeletSimulator";}
    @Override
        public String getAttributeName(){return "Shape";} 
    @Override
        public String getHeader(){
            String header=super.getHeader();
            header+="% Shapelet Length ="+shapeletLength;
            header+="% Series Length ="+seriesLength;
            header+="% Number of Shapelets ="+numShapelets;
            for(int i=0;i<shapes.size();i++)
                header+="%\t Shape "+i+" "+shapes.get(i).type+"\n";
            return header;
        }
   // Inner class determining the shape inserted into the shapelet model
    public class Shape{
        // Type: head and shoulders, spike, step, triangle, or sine wave.
        private ShapeType type;
        //Length of shape
        private int length;
        //Position of shape on axis determined by base (lowest point) and amp(litude).
        private double base;
        private double amp;
        //The position in the series at which the shape begins.
        private int location;
        
        
        //Default constructor, call randomise shape to get a random instance
        // The default length is 29, the shape extends from -2 to +2, is of 
        // type head and shoulders, and is located at index 0.
        private Shape()
        {
            this(ShapeType.HEADSHOULDERS,DEFAULTSHAPELETLENGTH,DEFAULTBASE,DEFAULTAMP); 
        }  
        //Set length only, default for the others
         private Shape(int length){
            this(ShapeType.HEADSHOULDERS,length,DEFAULTBASE,DEFAULTAMP);      
             
         }       
        // This constructor produces a completely specified shape
        private Shape(ShapeType t,int l, double b, double a){
            type=t;
            length=l;
            base=b;
            amp=a;
        }
        
        //Checks the location against the value t, and outputs part of the shape
        // if appropriate.
        private double generate(int t){
             if(t<location || t>location+length-1)
                return 0;
            
            int offset=t-location;            
            double value=0;
            
            switch(type){
             case TRIANGLE:
                 if(offset<=length/2) {
                    if(offset==0)
                       value=base;
                    else
                       value=((offset/(double)(length/2))*(amp))+base;
                 }
                 else
                 {
                     if(offset+1==length)
                         value=base;
                     else
                         value=((length-offset-1)/(double)(length/2)*(amp))+base;
                 }
                   break;
                case HEADSHOULDERS:
                    
                if(offset<length/3)
                    value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*offset))+base;
                else
                {
                    if(offset+1>=(2*length)/3){
                        if(length%3>0&&offset>=(length/3)*3)
                            value = base;
                        else
                            value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset+1-(2*length)/3)))+base;
                    }
                    else
                        value = ((amp)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset-length/3)))+base;
                }
                
                   break;
                case SINE:
                     value=amp*Math.sin(((2*Math.PI)/(length-1))*offset)/2;
                    break;
                case STEP:
                if(offset<length/2)
                    value=base;
                else
                    value=base+amp;
                    break;
                case SPIKE:
                if(offset<=length/4)
                 {
                    if(offset==0)
                       value=0;
                    else
                       value=offset/(double)(length/4)*(-amp/2);
                 }
                if(offset>length/4 && offset<=length/2)
                {
                    if(offset == length/2)
                        value=0;
                    else
                        value=(-amp/2)+((length/4-offset-1)/(double)(length/4)*(-amp/2));
                }                               
                if(offset>length/2&&offset<=length/4*3)
                    value=(offset-length/2)/(double)(length/4)*(amp/2);
                              
                if(offset>length/4*3)
                 {
                     if(offset+1==length)
                         value=0;
                     else
                         value=(length-offset-1)/(double)(length/4)*(amp/2);
                 }
                break;
            }
            return value;
            
        }
        
        private void setLocation(int newLoc){
            this.location=newLoc;
        }
        private void setType(ShapeType newType){
            this.type=newType;
        }
        
        // Randomises the starting location of a shape. Returns false when
        // there is insufficient space to fit all shapes within the value
        // of maxStart. This is resolved in the constructor.
        private boolean randomiseLocation(){
            int start = Model.rand.nextInt(seriesLength-shapeletLength);       
           setLocation(start);
           return true; 
        }
        
        @Override
        public String toString(){
            return ""+this.type+" start = "+location+" length ="+length;
        }
        
        //gives a shape a random type
        private boolean randomiseShape(){
            ShapeType [] types = ShapeType.values();            
            int ranType = Model.rand.nextInt(types.length);
            setType(types[ranType]);
                       
            return true; 
                
        }
    
}
    
    
    //Test harness
   
    public static void main (String[] args) throws IOException
    {
        OutFile out=new OutFile("C:\\temp\\shapeletExamples.csv");
        ShapeletModel model=new ShapeletModel();
        Shape s = model.new Shape();
        for(ShapeType st: ShapeType.values()){
            out.writeString(st.name());
            s.setType(st);
            s.setLocation(0);
            double[] series=new double[29];
            for(int i=0;i<29;i++){
                series[i]=s.generate(i);
                System.out.print(series[i]+" ");
                out.writeString(","+series[i]);
            }
            out.writeString("\n");
            System.out.print("\n");
        }
    }
    
}
