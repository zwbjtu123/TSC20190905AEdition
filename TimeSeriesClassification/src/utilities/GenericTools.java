/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 *
 * @author raj09hxu
 */
public class GenericTools {
    
    public static <E> ArrayList<E> cloneArrayList(ArrayList<E> list){
        ArrayList<E> temp = new ArrayList<>();
        for (E el : list) {
            temp.add(el);
        }
        return temp;              
    }

    public static <E> ArrayList<E> twoDArrayToList(E[][] twoDArray) {
        ArrayList<E> list = new ArrayList<>();
        for (E[] array : twoDArray){
            if(array == null) continue;
            
            for(E elm : array){
                if(elm == null) continue;
                
                list.add(elm);
            }
        }
        return list;
    }
    
    //this is inclusive of the top value.
    public static int randomRange(Random rand, int min, int max){
        return rand.nextInt((max - min) + 1) + min;
    }
}
