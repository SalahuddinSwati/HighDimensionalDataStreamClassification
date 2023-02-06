/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Salahuddin
 */
public class MicroCluster extends SyncObject{
    public double[] LS;
    public double[] SS;
    public int npts;
    public double radius;
    public int tr_pse_flg;
    public int time;
    public double weigth;
     public MicroCluster() {
    	
    }
      public MicroCluster(double[] LS, double [] SS, int pt,int label, int label_flg,double[] center, double r, int tpf,int time, double w ) {
    	this.LS=LS;
        this.SS=SS;
        this.npts=pt;
        super.label=label;
        super.label_flg=label_flg;
        super.data=center;
        this.radius=r;
        this.tr_pse_flg=tpf;
        this.time=time;
        this.weigth=w;
        
    }

    public void setCenter(double[] center) {
        super.data = center;
    }
    
    public int getlabel_flg(){
        return super.label_flg;
    }
     public int getlabel(){
        return super.label;
    }
    
    public double[] getCenter(){
        return super.data;
    }

    public void setLS(double[] LS) {
        this.LS = LS;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public void setLabel_flg(int label_flg) {
        this.label_flg = label_flg;
    }

    public void setNpts(int npts) {
        this.npts = npts;
    }

    public void setRadius(double radius) {
        this.radius = radius;
    }

    public void setSS(double[] SS) {
        this.SS = SS;
    }

    public void setTime(int time) {
        this.time = time;
    }

    public void setTr_pse_flg(int tr_pse_flg) {
        this.tr_pse_flg = tr_pse_flg;
    }

    public void setWeigth(double weigth) {
        this.weigth = weigth;
    }
      
    
}
