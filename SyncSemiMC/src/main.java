
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
        //String result_path = "F:\\PhD_Study\\Results_for_semi_papers\\semi_supervised\\ReSSL_other\\";
        //spamReduced13_New, spamReduced13, GSDReduced01, HARReduced13_New, HARReduced13, KDDCupReduce13_New, KDDCupReduced13
        // FCTReduce13_New, FCTReduced13, IoTBotnetReduce13_New, IoTBotnetReduced13, MNISTReduced13_New, MNISTReduced13
        filePath[1] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\spamReduce13_L3";//58000
        filePath[2] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\GSDReduce13_L2";//58000
        filePath[3] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\HARReduce13_L3";//58000
        filePath[4] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\RBF0001Reduced";//58000
        filePath[5] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\K9Reduced";//58000
       filePath[6] = "F:\\PhD_Study\\Datasets\\Dataset-MOA\\MOA_normal";//58000
        filePath[7] = "F:\\PhD_Study\\code\\denoising_autoencoder_v2\\20NG-nonsparse-filtered-single5Reduced";//58000
        filePath[8] = "F:\\PhD_Study\\Datasets\\RBFHYP\\new\\Hyp2C10A2D";//58000
        filePath[9] = "F:\\PhD_Study\\Datasets\\hyperplaneDataset\\ForPaper\\hyperplane_gradual_drift";//58000
          filePath[10] ="F:\\PhD_Study\\Datasets\\syntheticEvdatasets\\forpaper\\4CRE-V2_183K";
          filePath[11] = "F:\\PhD_Study\\Datasets\\RBFHYP\\new\\Hyp2C10A2D";//58000
          filePath[12] = "F:\\PhD_Study\\Datasets\\RBFHYP\\new\\Hyp2C10A10D";//58000
        ds_name[1] = "spamReduce13_L3";//58000
        ds_name[2]="GSDReduce13_L2";
        ds_name[3]="HARReduce13_L3";
        ds_name[4]="RBF0001Reduced";
        ds_name[5]="K9Reduced";
        ds_name[6]="MOA_normal";
        ds_name[7]="20NG-nonsparse-filtered-single5Reduced";
        ds_name[8]="HyperplaneFast";
        
       ds_name[9]="hyperplane_gradual_drift";
       ds_name[10]="4CRE-V2_183K";
       ds_name[11]="Hyp2C10A2D";
       ds_name[12]="Hyp2C10A10D";
        for(int i=9;i<=9;i++)
        {
        String fn = filePath[i]+".csv";//"F:\\PhD_Study\\Datasets\\weather\\weather_test.csv";//"spam.txt" ;//"elecNormNew.data";
        String sp = ",";
        SyncSemiMC sync = new SyncSemiMC(fn, sp, new int[]{5,10,20}, 3);
        try {
            sync.startProcess(ds_name[i]);
        } catch (IOException ex) {
//            Logger.getLogger(main.class.getName()).log(Level.SEVERE, null, ex);
            ex.printStackTrace();
        }
        }
    }
    
    
}
