/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vector_classifiers;

/**
 *
 * @author ajb
 */
public interface SaveEachParameter {
    void setPathToSaveParameters(String r);
    default void setSaveEachParaAcc(){setSaveEachParaAcc(true);}
    void setSaveEachParaAcc(boolean b);
}
