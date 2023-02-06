
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Salahuddin
 */
public class main {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String[] filePath = new String[23];
        String[] ds_name = new String[23];
        int dataset_total_ins[] = {58000, 45312, 18159, 10299, 13910, 9324, 105000, 50000, 663795, 494021, 581012, 245057, 58509, 20560
                ,144400,183000,200000,200000,200000};
        filePath[1] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\GSDReduced13";
        ds_name[1] = "GSDReduced13";//58
        
        String fn = filePath[1]+".csv";//"F:\\PhD_Study\\Datasets\\weather\\weather_test.csv";//"spam.txt" ;//"elecNormNew.data";
        String sp = ",";
        SyncSemiMC sync = new SyncSemiMC(fn, sp, new int[]{5,10,20}, 3);
        try {
            sync.startProcess();
        } catch (IOException ex) {
//            Logger.getLogger(main.class.getName()).log(Level.SEVERE, null, ex);
            ex.printStackTrace();
        }
    
    }
    
    
}
