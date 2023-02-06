
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Salahuddin
 */
public class SyncObject {
    public double[] data; 
    public int label;
    public int label_flg;
       
    
    public SyncObject() {
    	
    }
    public SyncObject(double[] inst,int ins_label,int ins_label_flg) {
    	this.data=inst;
        this.label=ins_label;
        this.label_flg=ins_label_flg;
    }
    public SyncObject(SyncObject obj) {
    	this.data=obj.data;
        this.label=obj.label;
        this.label_flg=obj.label_flg;
        
    }
    public void setLabel_flg(int label_flg) {
        this.label_flg = label_flg;
    }
    
    
}
