
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Salahuddin
 */
public class SyncSemiMC {

    public int dim = 0;
    public int eK;
    public int[] labelPercentage;
    public static int Dinit = 1000;
    public static int maxMC = 1000;
    public static int perofsample = 10;
    private String datasetFile;
    private String sp;
    public float lamba = 0.000002f;
    public float wT = 0.06f;
    public int Plocal = 0;
    public int PGlobal = 0;

    public SyncSemiMC(String fn, String sp, int[] labelPercentage, int eK) {
        this.datasetFile = fn;
        this.sp = sp;
        this.labelPercentage = labelPercentage;
        this.eK = eK;
        // TODO code application logic here
    }

    public void startProcess() throws IOException {
        Utils.Data data = Utils.readData(datasetFile, sp, Dinit);
        HashSet<Integer> partialLabelStream = new HashSet<Integer>();
        HashSet<Integer> partialLabelTrain = new HashSet<Integer>();
        double Acc;
        for (int i = 0; i < labelPercentage.length; i++) {
            // clear all prvious label flags to reset partial label process
            clearLabels(data.getStream_buffer(), partialLabelStream, 0);
            clearLabels(data.getTrain_buffer(), partialLabelTrain, 0);
            // partial label the data
            partialLabelStream = Utils.assignPartialLabel(data.getStream_buffer(), labelPercentage[i]);
            partialLabelTrain = Utils.assignPartialLabel(data.getTrain_buffer(), 90);

            double eRL = eRangData(data.getTrain_buffer(), eK);

            System.out.println("Local e-Rang =" + eRL);

            ArrayList<MicroCluster> LocalM = clustering(data.getTrain_buffer(), eRL);

            System.out.println("Local Model Size(MCs) =" + LocalM.size());
            //label propogate
            ArrayList<Integer> listofUnlabeled = new ArrayList<Integer>();
            for (int j = 0; j < LocalM.size(); j++) {
                if (LocalM.get(j).label_flg == 0) {
                    listofUnlabeled.add(j);
                }
            }
            double eRG = eRangMC(LocalM, eK);
            labelPropogate(LocalM, listofUnlabeled, eRG);
            //label propogation count
            int proLabelCount = labelPorgCount(LocalM);

            System.out.println("Total label Propogated= " + proLabelCount);
            //Strat of stream
            int currTime = 0;
            int TotalnewUnLabdeld = 0;
            Iterator<SyncObject> iterator1 = data.getStream_buffer().iterator();
            int correct = 0;
            Acc = 0;
            while (iterator1.hasNext()) {
                SyncObject ins_obj = iterator1.next();
                currTime++;
                ArrayList<Integer> p_label_idx = classify(ins_obj, LocalM, eRG);
                int p_label = p_label_idx.get(0);
                int mc_idx = p_label_idx.get(1);
                if (p_label == ins_obj.label) {
                    correct++;
                }
                //error driven represenation
                if (ins_obj.label_flg == 1) 
                {
                    if (ins_obj.label == p_label) {
                        double w = LocalM.get(mc_idx).weigth + 1;
                        LocalM.get(mc_idx).setWeigth(w);
                    } else {
                        double w = LocalM.get(mc_idx).weigth - 1;
                        LocalM.get(mc_idx).setWeigth(w);
                    }
                }
                //weight over time and del outdated mcs
//                int mcDelonTime = LocalM.size();
//                int mcDelonTimeflg = 0;
                updateModel(LocalM, currTime);
//                if (mcDelonTime - LocalM.size() != 0) {
//                    mcDelonTimeflg = 1;
//                }
                //incrementally add or create a new MC
                nnDistIndex nearest_MC = nearestSearchModel(ins_obj, LocalM, 1);
                int MCindex = nearest_MC.index[0];
                double MCdist = nearest_MC.dist[0];
                double MCradius = LocalM.get(MCindex).radius;
                int mcCreateflg = 0;
                if (MCdist <= MCradius) {
                    updateMC(ins_obj, MCindex, LocalM, currTime);
                } else {
                    createMC(ins_obj, LocalM, currTime, MCradius);
                    mcCreateflg = 1;
                }

                Acc = ((double) correct / currTime) * 100;
                if (currTime % 1000 == 0) {
                    //eRG = eRangMC(LocalM, eK);
                    System.out.println("Accuracy = " + Acc);
                   // System.out.println("Local Prediciton = " + Plocal);
                    System.out.println("Example= " + currTime);
                }
//                if (mcDelonTimeflg == 1 || mcCreateflg == 1) {
//                    //label propogate
//                    ArrayList<Integer> listofUnlabelednew = new ArrayList<Integer>();
//                    for (int j = 0; j < LocalM.size(); j++) {
//                        if (LocalM.get(j).label_flg == 0) {
//                            listofUnlabelednew.add(j);
//                        }
//                    }
//                    eRG = eRangMC(LocalM, eK);
//                    int proLableOld = labelPorgCount(LocalM);
//                    labelPropogate(LocalM, listofUnlabelednew, eRG);
//                    int proLablenew = labelPorgCount(LocalM);
//                    // System.out.println("Total label Propogated= " + (proLablenew-proLableOld));
//                }
                //int sum1 = labelPorgCount(LocalM);

                //int sum2 = labelPorgCount(LocalM);
//                if (sum1 < sum2) {
//                    System.out.println("new Propog labels =" + (sum2 - sum1));
//                }
            }//while loop over stream
            System.out.println("Label Per = " + labelPercentage[i]);
            System.out.println("Accuracy = " + Acc);
            // System.out.println("Total new MC unlabeled=" + TotalnewUnLabdeld);
            //System.out.println("Total new MC after Lbprob=" + proLabelCount);
        }//main for loop end
    }//functionend

    public void createMC(SyncObject x, ArrayList<MicroCluster> LocalM, int currTime, double radius) {

        if (LocalM.size() == maxMC) {

            if (mergUnLb(LocalM) == 0) {

                mergMaxLb(LocalM);
            }
        } //end if maxMC
        double[] LS = x.data.clone();
        double[] SS = new double[x.data.length];
        int label = 0;
        int label_flg = 0;
        double[] center = x.data.clone();;
        int tr_pse_flg = 0;
        double weigth = 1;
        int npts = 1;
        for (int d = 0; d < x.data.length; d++) {
            SS[d] += x.data[d] * x.data[d];
        }
        if (x.label_flg == 1) {
            label = x.label;
            label_flg = 1;
            tr_pse_flg = 1;

        }
        MicroCluster ms = new MicroCluster(LS, SS, npts, label, label_flg, center, radius, tr_pse_flg, currTime, weigth);
        LocalM.add(ms);
    }//func end

    public int mergUnLb(ArrayList<MicroCluster> LocalM) {
        double minDist = Double.MAX_VALUE;
        int minI = 0;
        int minJ = 0;
        int flg = 0;
        for (int i = 0; i < LocalM.size(); i++) {
            for (int j = 0; j < LocalM.size(); j++) {
                if (LocalM.get(i).label_flg == 0 && LocalM.get(j).label_flg == 1 && i != j) {
                    double pdist = computeDistance(LocalM.get(i), LocalM.get(j));
                    if (minDist > pdist) {
                        minDist = pdist;
                        minI = i;
                        minJ = j;
                        flg = 1;
                    }
                }
            }//for j end
        }// for i end
        if (flg == 1) {
            mergMC(LocalM, minI, minJ);
        }
        return flg;
    }

    public void mergMaxLb(ArrayList<MicroCluster> LocalM) {

        ArrayList<Integer> label_list = new ArrayList<Integer>();
        for (int i = 0; i < LocalM.size(); i++) {
            if (LocalM.get(i).label_flg == 1) {
                label_list.add(LocalM.get(i).label);
            }
        }
        int mmxlb = countFrequencies(label_list);
        double minDist = Double.MAX_VALUE;
        int minI = 0;
        int minJ = 0;
        for (int i = 0; i < LocalM.size(); i++) {
            for (int j = 0; j < LocalM.size(); j++) {
                if (LocalM.get(i).label == mmxlb && LocalM.get(j).label == mmxlb && i != j) {
                    double pdist = computeDistance(LocalM.get(i), LocalM.get(j));
                    if (minDist > pdist) {
                        minDist = pdist;
                        minI = i;
                        minJ = j;
                    }

                }
            }//for j end
        }// for i end
        mergMC(LocalM, minI, minJ);
    }

    public void mergMC(ArrayList<MicroCluster> LocalM, int minI, int minJ) {
        int dim = LocalM.get(minI).data.length;
        for (int d = 0; d < dim; d++) {
            LocalM.get(minJ).LS[d] += LocalM.get(minI).LS[d];
            LocalM.get(minJ).SS[d] += LocalM.get(minI).SS[d];
        }
        LocalM.get(minJ).npts += LocalM.get(minI).npts;

        double ls_sum = 0.0;
        double ss_sum = 0.0;
        double[] center = new double[dim];
        double radius = 0.0;
        for (int d = 0; d < dim; d++) {
            center[d] = LocalM.get(minJ).LS[d] / LocalM.get(minJ).npts;
            ls_sum += (center[d] * center[d]);
            ss_sum += (LocalM.get(minJ).SS[d] / LocalM.get(minJ).npts);
//                sum += ( (SS[d] / npts) - (center[d]*center[d]) );
        }

        radius = Math.sqrt(ss_sum - ls_sum);
        LocalM.get(minJ).setCenter(center);
        LocalM.get(minJ).radius = radius;
        if (LocalM.get(minJ).tr_pse_flg == 2 && LocalM.get(minI).tr_pse_flg == 1) {
            LocalM.get(minJ).tr_pse_flg = 1;
        }
        LocalM.remove(minI);
        //LocalM.get(minJ).time = currTime;
    }

    public void updateMC(SyncObject x, int index, ArrayList<MicroCluster> LocalM, int currTime) {
        int dim = x.data.length;
        for (int d = 0; d < dim; d++) {
            LocalM.get(index).LS[d] += x.data[d];
            LocalM.get(index).SS[d] += (x.data[d] * x.data[d]);
        }
        LocalM.get(index).npts++;
        if (x.label_flg == 1 && LocalM.get(index).label_flg == 0) {
            LocalM.get(index).label = x.label;
            LocalM.get(index).label_flg = 1;
            LocalM.get(index).tr_pse_flg = 1;
        }
        double ls_sum = 0.0;
        double ss_sum = 0.0;
        double[] center = new double[dim];
        double radius = 0.0;
        for (int d = 0; d < dim; d++) {
            center[d] = LocalM.get(index).LS[d] / LocalM.get(index).npts;
            ls_sum += (center[d] * center[d]);
            ss_sum += (LocalM.get(index).SS[d] / LocalM.get(index).npts);
//                sum += ( (SS[d] / npts) - (center[d]*center[d]) );
        }

        radius = Math.sqrt(ss_sum - ls_sum);
        LocalM.get(index).setCenter(center);
        LocalM.get(index).radius = radius;
        LocalM.get(index).time = currTime;

    }

    public void updateModel(ArrayList<MicroCluster> LocalM, int currTime) {
        ArrayList<MicroCluster> mcToDel = new ArrayList<MicroCluster>();
        for (int i = 0; i < LocalM.size(); i++) {
            double w = LocalM.get(i).weigth * Math.pow(2, -lamba * (currTime - LocalM.get(i).time));
            LocalM.get(i).setWeigth(w);
            if (w < wT) {
                mcToDel.add(LocalM.get(i));
            }
        }
        LocalM.remove(mcToDel);

    }

    public ArrayList<Integer> classify(SyncObject inst, ArrayList<MicroCluster> LocalM, double eRG) {
        ArrayList<Integer> p_label_idx = new ArrayList<Integer>();
        double minDistL = Double.MAX_VALUE;
        int idxL = 0;
        for (int i = 0; i < LocalM.size(); i++) {
            if (LocalM.get(i).label_flg == 1) {
                double dist = computeDistance(inst, LocalM.get(i));
                if (minDistL > dist) {
                    minDistL = dist;
                    idxL = i;
                }
            }
        }
//        if (minDistL <= eRG) {
        p_label_idx.add(LocalM.get(idxL).label);
//            Plocal++;
//        } else {
//            //For global model
//            ArrayList<SyncObject> global_data = new ArrayList<SyncObject>();
//            Iterator<MicroCluster> iterator = LocalM.iterator();
//            while (iterator.hasNext()) {
//                MicroCluster mcs = iterator.next();
//                global_data.add(new SyncObject(mcs.getCenter().clone(), mcs.label, mcs.label_flg));
//            }
//            //double eRG1 = eRangMC(LocalM,eK);
//            //double eRG2 = eRang(global_data, eK);
//            // System.out.println("Gobal e-Rang using offset =" + eRG1);
//            //System.out.println("Gobal e-Rang using Local =" + eRG2);
//            ArrayList<MicroCluster> GlobalM = clustering(global_data, eRG);
//            //System.out.println("Gobal Model Size(MCs) =" + GlobalM.size());
//            double minDistG = Double.MAX_VALUE;
//            int idxG = 0;
//            for (int i = 0; i < GlobalM.size(); i++) {
//                if (GlobalM.get(i).label_flg == 1) {
//                    double dist = computeDistance(inst, GlobalM.get(i));
//                    if (minDistG > dist) {
//                        minDistG = dist;
//                        idxG = i;
//                    }
//                }
//            }
//            p_label_idx.add(GlobalM.get(idxG).label);
//            PGlobal++;
//        }
        p_label_idx.add(idxL);
        return p_label_idx;
    }

    /**
     * synchronization-based dynamic clustering
     */
    public ArrayList<MicroCluster> clustering(ArrayList<SyncObject> data_orig, double eRange) {

        int num = data_orig.size();
        int dim = data_orig.get(0).data.length;
        //System.out.println(num + "," + dim);
        boolean loop = true;
        int loopNum = 0;
        double localOrder = 0.0;
        double prelocalOrder = 0.0;
        double allorder = 0.0;
        double minDist = 0.0;
        int cc = 0;

        ArrayList<SyncObject> data_copy = new ArrayList<SyncObject>();
        ArrayList<SyncObject> prex = new ArrayList<SyncObject>();
        Iterator<SyncObject> data_iterator = data_orig.iterator();
        while (data_iterator.hasNext()) {
            SyncObject next = data_iterator.next();
            data_copy.add(new SyncObject(next.data.clone(), next.label, next.label_flg));
            prex.add(new SyncObject(next.data.clone(), next.label, next.label_flg));
        }

        //Dynamic clustering
        while (loop) {

            double[] order = new double[num];
            localOrder = 0.0;
            minDist = 0.0;
            cc = 0;
            allorder = 0.0;

            loopNum = loopNum + 1;

            for (int i = 0; i < num; i++) {
                double[] sinValue = new double[dim];
                double[] diss = new double[dim];
                double[] temp = new double[dim];

                double dis = 0.0;
                int n = 0;
                double sita = 0.0;

                for (int j = 0; j < num; j++) {
                    dis = 0.0;
                    for (int d = 0; d < dim; d++) {
                        diss[d] = prex.get(j).data[d] - prex.get(i).data[d];
                    }

                    dis = EuclideanDist(diss);

                    if (dis < eRange) {
                        if ((prex.get(i).label_flg == 0 || prex.get(j).label_flg == 0) || (prex.get(i).label == prex.get(j).label)) {
                            n = n + 1;
                            //Calculating the coupling strength
                            for (int d = 0; d < dim; d++) {
                                temp[d] = (diss[d] + 1e-10) / (prex.get(j).data[d] + 1e-10);
                            }

                            for (int d = 0; d < dim; d++) {
                                sinValue[d] = sinValue[d] + Math.sin(diss[d]); //attract
                            }
                            sita = sita + Math.exp(-dis);
                            minDist = minDist + dis;
                            cc++;
                        }

                    }
                }
                //System.out.println(n);

                if (n > 1) {
                    for (int d = 0; d < dim; d++) {
                        data_copy.get(i).data[d] = prex.get(i).data[d] + (float) ((1.0 / n) * sinValue[d]);
                    }
                    order[i] = sita / n;

                }
            }
            for (int k = 0; k < num; k++) {
                allorder = allorder + order[k];
            }

            minDist = minDist / cc;

            //Local order parameter
            localOrder = allorder / num;

            if (localOrder > 1 - (1e-3) || loopNum >= 20 || Math.abs(prelocalOrder - localOrder) < 10e-6) { //user's specification
                loop = false;
            }
            prelocalOrder = localOrder;

            for (int i = 0; i < num; i++) {
//	    		float[] temp1 = new float[dim];    		
//	    		for(int j=0;j<dim;j++){
//	    			temp1[j] = data.get(i).data[j];
//	    		}
                prex.get(i).data = data_copy.get(i).data.clone();
            }
        }
       // System.out.println("No of itration to converge = " + loopNum);

        //find the clusters
        ArrayList<MicroCluster> mcs = findSynCluster(data_orig, data_copy, eRange);

        return mcs;
    }

    public static double EuclideanDist(double[] dis) {

        double val = 0.0;
        for (int i = 0; i < dis.length; i++) {
            val = val + dis[i] * dis[i];
        }
        double dist = Math.sqrt(val);
        return dist;
    }

    public static double computeDistance(SyncObject x, SyncObject y) {
        double r = 0.0;
        int d = x.data.length;

        for (int i = 0; i < d; i++) {
            r = r + (x.data[i] - y.data[i]) * (x.data[i] - y.data[i]);
        }
        r = Math.sqrt(r);

        return r;

    }

    public static nnDistIndex nearestSearch(SyncObject x, ArrayList<SyncObject> y, int k) {
        double[] r = new double[y.size()];
        HashMap clusterIndex_distances = new HashMap<Integer, Double>();
        for (int index = 0; index < y.size(); index++) {
            double distance = computeDistance(x, y.get(index));
            r[index] = r[index] + distance;
            clusterIndex_distances.put(index, distance);
        }

        HashMap<Object, Object> sortedClusIndexDistances = Utils.sortByValues(clusterIndex_distances);
        int counter = 0;
        double[] kD = new double[k];
        int[] kDidx = new int[k];
        for (Map.Entry<Object, Object> entrySet : sortedClusIndexDistances.entrySet()) {

            Integer key = (Integer) entrySet.getKey();
            kDidx[counter] = key;
            Double value = (Double) entrySet.getValue();
            kD[counter] = value;
            //System.out.println("Key: "+key+"    Value: "+value);
            if (counter == k - 1) {
                break;
            }
            counter++;
        }

        return new nnDistIndex(kDidx, kD);

    }

    public static nnDistIndex nearestSearchModel(SyncObject x, ArrayList<MicroCluster> y, int k) {
        double[] r = new double[y.size()];
        HashMap clusterIndex_distances = new HashMap<Integer, Double>();
        for (int index = 0; index < y.size(); index++) {
            SyncObject z = new SyncObject(y.get(index).data, y.get(index).label, y.get(index).label_flg);
            double distance = computeDistance(x, z);
            r[index] = r[index] + distance;
            clusterIndex_distances.put(index, distance);
        }

        HashMap<Object, Object> sortedClusIndexDistances = Utils.sortByValues(clusterIndex_distances);
        int counter = 0;
        double[] kD = new double[k];
        int[] kDidx = new int[k];
        for (Map.Entry<Object, Object> entrySet : sortedClusIndexDistances.entrySet()) {

            Integer key = (Integer) entrySet.getKey();
            kDidx[counter] = key;
            Double value = (Double) entrySet.getValue();
            kD[counter] = value;
            //System.out.println("Key: "+key+"    Value: "+value);
            if (counter == k - 1) {
                break;
            }
            counter++;
        }

        return new nnDistIndex(kDidx, kD);

    }

    public static double eRangData(ArrayList<SyncObject> data, int e_Range_K) {
        int n = data.size();
        int noofsamples = (int) (n * perofsample / 100);
        HashSet<Integer> tnos = (HashSet<Integer>) Utils.generateRandomNumbers(1, n - 1, noofsamples);
        double[] eD = new double[noofsamples];
        Iterator<Integer> iterator = tnos.iterator();
        int e = 0;
        while (iterator.hasNext()) {
            Integer next = iterator.next();
            //System.out.println(next);
            nnDistIndex nearest_neighbor_distances = nearestSearch(data.get(next), data, e_Range_K + 1);
            double sum = 0;
            for (int i = 1; i < nearest_neighbor_distances.dist.length; i++) {
                sum += nearest_neighbor_distances.dist[i];
            }
            eD[e] = sum / (nearest_neighbor_distances.dist.length - 1);
            e++;

        }
        double sum = 0;
        for (int i = 0; i < eD.length; i++) {
            sum += eD[i];
        }
        double eavgD = sum / eD.length;
        return eavgD;
    }

//    -------------------------------------------------------------
    public static ArrayList<MicroCluster> findSynCluster(ArrayList<SyncObject> origdata, ArrayList<SyncObject> data, double eR) {

        int len = data.size();
        int dim = data.get(0).data.length;
        ArrayList<SyncObject> SyncCluster = new ArrayList<SyncObject>();
        ArrayList<MicroCluster> mcs = new ArrayList<MicroCluster>();
        ArrayList<Integer> comblist = new ArrayList<Integer>();
        ArrayList<ArrayList> clulist = new ArrayList<ArrayList>();
        //double [][] pdist=new double[len][len];

        for (int i = 0; i < len; i++) {
            ArrayList<Integer> templist = new ArrayList<Integer>();
            templist.add(i);
            if (comblist.contains(i)) {
                continue;
            }
            comblist.add(i);
            for (int j = 0; j < len; j++) {
                double pdist = computeDistance(data.get(i), data.get(j));
                if (i != j && pdist < 10e-4) {//or eR

                    comblist.add(j);
                    templist.add(j);
                }
            }
            clulist.add(templist);
        }
        int noc = clulist.size(); //total no of clusters
        // int sum=0;
        Iterator<ArrayList> clulist_iterator = clulist.iterator();

        while (clulist_iterator.hasNext()) {
            ArrayList cluster_items = clulist_iterator.next();
            int npts = cluster_items.size();
            //sum=sum+npts;
            double[] LS = new double[dim];
            double[] SS = new double[dim];
            int label = -1;
            int label_flg = 0;
            double[] center = new double[dim];
            double radius = 0.0;
            int tr_pse_flg = 0;
            int time = 1;
            double weigth = 1;
            for (int j = 0; j < npts; j++) {
                int instance_idx = (int) cluster_items.get(j);
                for (int d = 0; d < dim; d++) {
                    double dimension_value = origdata.get(instance_idx).data[d];
                    LS[d] += dimension_value;
                    SS[d] += (dimension_value * dimension_value);
                }
                // System.out.println(idx);
            }
            double ls_sum = 0.0;
            double ss_sum = 0.0;
            for (int d = 0; d < dim; d++) {
                center[d] = LS[d] / npts;
                ls_sum += (center[d] * center[d]);
                ss_sum += (SS[d] / npts);
//                sum += ( (SS[d] / npts) - (center[d]*center[d]) );
            }

            radius = Math.sqrt(ss_sum - ls_sum);
            label = assignLabel(cluster_items, origdata);
            if (label != -1) {
                label_flg = 1;
                tr_pse_flg = 1;
            }
            MicroCluster ms = new MicroCluster(LS, SS, npts, label, label_flg, center, radius, tr_pse_flg, time, weigth);
            mcs.add(ms);
        }
        //System.out.println("total "+sum);
        Iterator<MicroCluster> iterator = mcs.iterator();
        double rsum = 0.0;
        while (iterator.hasNext()) {
            MicroCluster mcrd = iterator.next();
            rsum += mcrd.radius;

        }
        double mnrd = rsum / mcs.size();
        Iterator<MicroCluster> iterator2 = mcs.iterator();
        while (iterator2.hasNext()) {
            MicroCluster mc_rd_up = iterator2.next();
            if (mc_rd_up.radius == 0) {
                mc_rd_up.setRadius(mnrd);
            }
        }
        return mcs;
    }

    private static int assignLabel(ArrayList cluster_items, ArrayList<SyncObject> origdata) {
        ArrayList<Integer> clu_labels = new ArrayList<Integer>();
        int truelabel = -1;
        for (int i = 0; i < cluster_items.size(); i++) {
            int idx = (int) cluster_items.get(i);
            if (origdata.get(idx).label_flg == 1) {
                clu_labels.add(origdata.get(idx).label);

            }
        }
        if (clu_labels.size() == 0) {
            truelabel = -1;
        } else if (clu_labels.size() == 1) {
            truelabel = clu_labels.get(0);
        } else {
            truelabel = countFrequencies(clu_labels);
        }

        return truelabel;
    }

    private void clearLabels(ArrayList<SyncObject> buffer, HashSet<Integer> partialLabelStream, int defined_label) {
        Iterator<Integer> iterator = partialLabelStream.iterator();
        while (iterator.hasNext()) {
            Integer index = iterator.next();
            buffer.get(index).setLabel_flg(defined_label);
        }
    }

    public static int countFrequencies(ArrayList<Integer> list) {
        // hashmap to store the frequency of element 
        Map<Integer, Integer> hm = new HashMap<Integer, Integer>();

        for (Integer i : list) {
            Integer j = hm.get(i);
            hm.put(i, (j == null) ? 1 : j + 1);
        }

        // displaying the occurrence of elements in the arraylist 
        ArrayList<Integer> labels = new ArrayList<Integer>();
        ArrayList<Integer> labels_count = new ArrayList<Integer>();
        for (Map.Entry<Integer, Integer> val : hm.entrySet()) {
            labels.add(val.getKey());
            labels_count.add(val.getValue());
        }
        int truelabel;
        if (labels.size() == 1) {
            truelabel = labels.get(0);

        } else {
            int mxidx = labels_count.indexOf(Collections.max(labels_count));
            truelabel = labels.get(mxidx);

        }
        return truelabel;
    }

    public void labelPropogate(ArrayList<MicroCluster> LocalM, ArrayList<Integer> listofUnlabeled, double eRG) {
        //double eRL1 = eRangMC(LocalM, eK);

        for (int i = 0; i < listofUnlabeled.size(); i++) {
            ArrayList<Integer> nearestLabels = new ArrayList<Integer>();
            for (int j = 0; j < LocalM.size(); j++) {
                if (LocalM.get(j).tr_pse_flg == 1) {
                    double pdist = computeDistance(LocalM.get(listofUnlabeled.get(i)), LocalM.get(j));
                    if (pdist <= eRG) {
                        nearestLabels.add(LocalM.get(j).label);
                    }

                }
            }//for j end

            HashSet<Integer> uniqueValues = new HashSet<Integer>(nearestLabels);
            if (uniqueValues.size() == 1 && nearestLabels.size() >= 3) {
                LocalM.get(listofUnlabeled.get(i)).label = nearestLabels.get(0);
                LocalM.get(listofUnlabeled.get(i)).label_flg = 1;
                LocalM.get(listofUnlabeled.get(i)).tr_pse_flg = 2;
            } else if (uniqueValues.size() == 0) {
                int aaa = 0;
            } else {
                int aaaa = 0;
            }
        }// for i end

    }

    public int labelPorgCount(ArrayList<MicroCluster> LocalM) {
        int sum = 0;
        for (int ii = 0; ii < LocalM.size(); ii++) {
            if (LocalM.get(ii).tr_pse_flg == 2) {
                sum++;
            }

        }

        return sum;
    }

    public double eRangMC(ArrayList<MicroCluster> LocalM, int eK) {
        ArrayList<SyncObject> data = new ArrayList<SyncObject>();
        Iterator<MicroCluster> iterator1 = LocalM.iterator();
        while (iterator1.hasNext()) {
            MicroCluster mcs = iterator1.next();
            data.add(new SyncObject(mcs.getCenter().clone(), mcs.label, mcs.label_flg));
        }
        int n = data.size();
        int noofsamples = (int) (n * perofsample / 100);
        HashSet<Integer> tnos = (HashSet<Integer>) Utils.generateRandomNumbers(1, n - 1, noofsamples);
        double[] eD = new double[noofsamples];
        Iterator<Integer> iterator = tnos.iterator();
        int e = 0;
        while (iterator.hasNext()) {
            Integer next = iterator.next();
            //System.out.println(next);
            nnDistIndex nearest_neighbor_distances = nearestSearch(data.get(next), data, eK + 1);
            double sum = 0;
            for (int i = 1; i < nearest_neighbor_distances.dist.length; i++) {
                sum += nearest_neighbor_distances.dist[i];
            }
            eD[e] = sum / (nearest_neighbor_distances.dist.length - 1);
            e++;

        }
        double sum = 0;
        for (int i = 0; i < eD.length; i++) {
            sum += eD[i];
        }
        double eavgD = sum / eD.length;
        return eavgD;

    }

    public static class nnDistIndex {

        private int[] index; //= new ArrayList<SyncObject>();
        private double[] dist; //= new ArrayList<SyncObject>();

        public nnDistIndex(int[] index, double[] dist) {
            this.index = index;
            this.dist = dist;
        }

        public double[] getDist() {
            return dist;
        }

        public int[] getIndex() {
            return index;
        }

    }
}
