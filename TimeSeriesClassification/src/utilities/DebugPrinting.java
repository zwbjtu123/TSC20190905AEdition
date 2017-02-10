/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

/**
 * Just saves some time/copied code when making new classes
 * 
 * The codebase needed a bit more hacky-ness to it 
 * 
 * TODO: implement using lambda's or something to skip the overhead of boolean casting
 * 
 * @author James Large
 */
public interface DebugPrinting {
    
    Object[] debug = new Object[] { false };
    
    default void setDebugPrinting(boolean b) {
        debug[0] = b;
    }
    
    default boolean getDebugPrinting() {
        return (Boolean)debug[0];
    }
    
    default void printDebug(String str) {
        if ((Boolean)debug[0])
            System.out.print(str);
    }
    
    default void printlnDebug(String str) {
        if ((Boolean)debug[0])
            System.out.println(str);
    }
    
}
