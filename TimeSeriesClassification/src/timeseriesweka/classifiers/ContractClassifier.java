/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.classifiers;

/**
 *
 * @author raj09hxu
 */
public interface ContractClassifier {

    public enum TimeLimit {MINUTE, HOUR, DAY};

    public default void setOneDayLimit(){
        setTimeLimit(TimeLimit.DAY, 1);
    }
    
    public default void setOneHourLimit(){
        setTimeLimit(TimeLimit.HOUR, 1);
    }

    public default void setOneMinuteLimit(){
        setTimeLimit(TimeLimit.MINUTE, 1);
    }
    
    public default void setDayLimit(int t){
        setTimeLimit(TimeLimit.DAY, t);
    }

    public default void setHourLimit(int t){
        setTimeLimit(TimeLimit.HOUR, t);
    }
    
    public default void setMinuteLimit(int t){
        setTimeLimit(TimeLimit.MINUTE, t);
    }

    //set any value in nanoseconds you like.
    void setTimeLimit(long time);

    //pass in an enum of hour, minut, day, and the amount of them.
    void setTimeLimit(TimeLimit time, int amount);
    
}
