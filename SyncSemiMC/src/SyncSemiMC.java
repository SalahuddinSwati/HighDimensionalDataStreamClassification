
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
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
    public float lamba = 0.0001f;//3->0.0009, 0.15: 8 -> 0.0001, 0.12: 20-> 0.00001, 0.26
    public float wT = 0.15f;
    public int Plocal = 0;
    public int PGlobal = 0;
    public int chunkSize = 1000;

    public SyncSemiMC(String fn, String sp, int[] labelPercentage, int eK) {
        this.datasetFile = fn;
        this.sp = sp;
        this.labelPercentage = labelPercentage;
        this.eK = eK;
        // TODO code application logic here
    }

    public void startProcess(String dname) throws IOException {
        Utils.Data data = Utils.readData(datasetFile, sp, Dinit);
        double Acc;
        StringBuffer accb = new StringBuffer();
        String result_path = "F:\\PhD_Study\\Revision_results\\RBFHYP\\semiSync\\";
        for (int i = 0; i < labelPercentage.length; i++) {
            // clear all prvious label flags to reset partial label process
            clearLabels(data.getStream_buffer());
            clearLabels(data.getTrain_buffer());
            // partial label the data
            //Utils.assignPartialLabel(data.getStream_buffer(), labelPercentage[i]);
            Utils.assignPartialLabel(data.getTrain_buffer(), 90);

            double eRL = eRangData(data.getTrain_buffer(), eK);

            System.out.println("Local e-Rang =" + eRL);

            ArrayList<MicroCluster> Model = clustering(data.getTrain_buffer(), eRL, 1);
            System.out.println("Local Model Size(MCs) =" + Model.size());
            //start stream
            int currTime = 0;
            int TotalnewUnLabdeld = 0;
            
            Iterator<SyncObject> iterator1 = data.getStream_buffer().iterator();
            int correct = 0;
            Acc = 0;
            int total_ins = 0;
            int b = 0;
            StringBuffer actual_predicted = new StringBuffer();
            ArrayList<SyncObject> chunkBuffer = new ArrayList<SyncObject>();
            while (iterator1.hasNext()) {
                SyncObject ins_obj = iterator1.next();
                currTime++;
                total_ins++;
                chunkBuffer.add(ins_obj);
                ArrayList<Integer> p_label_idx = classify(ins_obj, Model);
                int p_label = p_label_idx.get(0);
                int mc_idx = p_label_idx.get(1);
                actual_predicted.append(ins_obj.label).append("\t");
                actual_predicted.append(p_label).append("\n");
                if (p_label == ins_obj.label) {
                    correct++;
                }
                if (chunkBuffer.size() == chunkSize) {
                    System.out.println("Model size Before =" + Model.size());
                    updateModel(Model, chunkBuffer, currTime, labelPercentage[i], eRL);
                    System.out.println("Model size After =" + Model.size());
                    b++;
                    Acc = ((double) correct / total_ins) * 100;
                    //System.out.println("Label Per = " + labelPercentage[i]);
                    System.out.println("Block No = " + b);
                    System.out.println("Accuracy = " + Acc);
                }

            }//while loop over stream
            
            File dir = new File(result_path + dname + "\\");
            dir.mkdir();
            
            FileWriter fw = new FileWriter(result_path + dname + "\\" + "Actual_predictd_Label_ratio_" + labelPercentage[i] + ".txt");
            fw.write(actual_predicted.toString());
             fw.close();
            Acc = ((double) correct / total_ins) * 100;
            System.out.println("Label Per = " + labelPercentage[i]);
            System.out.println("Accuracy = " + Acc);
            accb.append(Acc).append("\n");
            
            // System.out.println("Total new MC unlabeled=" + TotalnewUnLabdeld);
            //System.out.println("Total new MC after Lbprob=" + proLabelCount);
        }//main for loop end
        FileWriter fw2 = new FileWriter(result_path + dname + "\\" + "Accu_"  + ".txt");
            fw2.write(accb.toString());
             fw2.close();
    }//functionend

    public void updateModel(ArrayList<MicroCluster> Model, ArrayList<SyncObject> chunkBuffer, int currTime, int labelPercentage, double eRG) {
        Utils.assignPartialLabel(chunkBuffer, labelPercentage);
        Iterator<SyncObject> iterator = chunkBuffer.iterator();
        while (iterator.hasNext()) {
            SyncObject chunk_ins = iterator.next();

            ArrayList<Integer> p_label_idx = classify(chunk_ins, Model);
            int p_label = p_label_idx.get(0);
            int mc_idx = p_label_idx.get(1);

            //error driven represenation
            if (chunk_ins.label_flg == 1) {
                if (p_label == chunk_ins.label) {
                    double w = Model.get(mc_idx).weigth + 1;
                    Model.get(mc_idx).setWeigth(w);
                } else {
                    double w = Model.get(mc_idx).weigth - 1;
                    Model.get(mc_idx).setWeigth(w);
                }
            }
        }
        //weight over time
        for (int i = 0; i < Model.size(); i++) {
            double w = Model.get(i).weigth * Math.pow(2, -lamba * (currTime - Model.get(i).time));
            Model.get(i).setWeigth(w);

        }
        ArrayList<MicroCluster> neg_lowWMC = new ArrayList<MicroCluster>();
        for (int i = 0; i < Model.size(); i++) {
            if (Model.get(i).weigth < wT) {
                neg_lowWMC.add(Model.get(i));
            }
        }
        Model.removeAll(neg_lowWMC);//delete negative and low weight
        // new microclusters
        double eRL = eRangData(chunkBuffer, eK);
        ArrayList<MicroCluster> newMCs = clustering(chunkBuffer, eRL, currTime);
        for (int i = 0; i < newMCs.size(); i++) {
            insertNewMC(newMCs.get(i), Model, currTime);
            //Model.add(newMCs.get(i));
        }
//        if (Model.size() > maxMC) {
//            int noMCtoMerg = Model.size() - maxMC;
//            findNearestMCPairs(Model, noMCtoMerg, currTime);
//        }
        chunkBuffer.clear();
        //return Model;
    }
public void insertNewMC(MicroCluster nwmc, ArrayList<MicroCluster> Model, int currTime) {

        if (Model.size() == maxMC) {

            if (mergUnLb(Model) == 0) {

                mergMaxLb(Model);
            }
        } //end if maxMC
        Model.add(nwmc);
    }//func end

    public int mergUnLb(ArrayList<MicroCluster> Model) {
        double minDist = Double.MAX_VALUE;
        int minI = 0;
        int minJ = 0;
        int flg = 0;
        for (int i = 0; i < Model.size(); i++) {
            for (int j = 0; j < Model.size(); j++) {
                if (Model.get(i).label_flg == 0 && Model.get(j).label_flg == 1 && i != j) {
                    double pdist = computeDistance(Model.get(i), Model.get(j));
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
            mergMC(Model, minI, minJ);
        }
        return flg;
    }

    public void mergMaxLb(ArrayList<MicroCluster> Model) {

        ArrayList<Integer> label_list = new ArrayList<Integer>();
        for (int i = 0; i < Model.size(); i++) {
            if (Model.get(i).label_flg == 1) {
                label_list.add(Model.get(i).label);
            }
        }
        int mmxlb = countFrequencies(label_list);
        double minDist = Double.MAX_VALUE;
        int minI = 0;
        int minJ = 0;
        for (int i = 0; i < Model.size(); i++) {
            for (int j = 0; j < Model.size(); j++) {
                if (Model.get(i).label == mmxlb && Model.get(j).label == mmxlb && i != j) {
                    double pdist = computeDistance(Model.get(i), Model.get(j));
                    if (minDist > pdist) {
                        minDist = pdist;
                        minI = i;
                        minJ = j;
                    }

                }
            }//for j end
        }// for i end
        mergMC(Model, minI, minJ);
    }

    public void mergMC(ArrayList<MicroCluster> Model, int minI, int minJ) {
        int dim = Model.get(minI).data.length;
        for (int d = 0; d < dim; d++) {
            Model.get(minJ).LS[d] += Model.get(minI).LS[d];
            Model.get(minJ).SS[d] += Model.get(minI).SS[d];
        }
        Model.get(minJ).npts += Model.get(minI).npts;

        double ls_sum = 0.0;
        double ss_sum = 0.0;
        double[] center = new double[dim];
        double radius = 0.0;
        for (int d = 0; d < dim; d++) {
            center[d] = Model.get(minJ).LS[d] / Model.get(minJ).npts;
            ls_sum += (center[d] * center[d]);
            ss_sum += (Model.get(minJ).SS[d] / Model.get(minJ).npts);
//                sum += ( (SS[d] / npts) - (center[d]*center[d]) );
        }

        radius = Math.sqrt(ss_sum - ls_sum);
        Model.get(minJ).setCenter(center);
        Model.get(minJ).radius = radius;
//        if (LocalM.get(minJ).tr_pse_flg == 2 && LocalM.get(minI).tr_pse_flg == 1) {
//            LocalM.get(minJ).tr_pse_flg = 1;
//        }
        Model.remove(minI);
        //LocalM.get(minJ).time = currTime;
    }

    public void findNearestMCPairs(ArrayList<MicroCluster> Model, int n, int currTime) {
        ArrayList<Integer> comPair = new ArrayList<Integer>();
        HashMap clusterIndex_distances = new HashMap<ArrayList<Integer>, Double>();
        for (int i = 0; i < Model.size(); i++) {
            for (int j = i + 1; j < Model.size(); j++) {
                if ((Model.get(i).label_flg == 0 || Model.get(j).label_flg == 0) || (Model.get(i).label == Model.get(j).label)) {
                    double pdist = computeDistance(Model.get(i), Model.get(j));
                    ArrayList<Integer> pair = new ArrayList<Integer>();
                    pair.add(i);
                    pair.add(j);
                    clusterIndex_distances.put(pair, pdist);
                }
            }//for j end
        }// for i end

        HashMap<Object, Object> sortedClusIndexDistances = Utils.sortByValues(clusterIndex_distances);
        ArrayList<ArrayList> pairlist = new ArrayList<ArrayList>();
        int f = 0;
        for (Map.Entry<Object, Object> entrySet : sortedClusIndexDistances.entrySet()) {

            ArrayList<Integer> minPairs = (ArrayList<Integer>) entrySet.getKey();
            pairlist.add(minPairs);
            if (f < 5) {
                Double value = (Double) entrySet.getValue();
                //System.out.println("values"+value);
                f++;
            }

        }
        ArrayList<MicroCluster> mcToDel = new ArrayList<MicroCluster>();
        for (int i = 0; i < n; i++) {
            mergMC(Model, (int) pairlist.get(i).get(0), (int) pairlist.get(i).get(1), currTime);
            mcToDel.add(Model.get((int) pairlist.get(i).get(0)));
        }
        Model.removeAll(mcToDel);
        int aa = 0;
    }

    public void mergMC(ArrayList<MicroCluster> Model, int minI, int minJ, int currTime) {
        int dim = Model.get(minI).data.length;
        for (int d = 0; d < dim; d++) {
            Model.get(minJ).LS[d] += Model.get(minI).LS[d];
            Model.get(minJ).SS[d] += Model.get(minI).SS[d];
        }
        Model.get(minJ).npts += Model.get(minI).npts;

        double ls_sum = 0.0;
        double ss_sum = 0.0;
        double[] center = new double[dim];
        double radius = 0.0;
        for (int d = 0; d < dim; d++) {
            center[d] = Model.get(minJ).LS[d] / Model.get(minJ).npts;
            ls_sum += (center[d] * center[d]);
            ss_sum += (Model.get(minJ).SS[d] / Model.get(minJ).npts);
//                sum += ( (SS[d] / npts) - (center[d]*center[d]) );
        }

        radius = Math.sqrt(ss_sum - ls_sum);
        Model.get(minJ).setCenter(center);
        Model.get(minJ).radius = radius;
        if (Model.get(minJ).label_flg == 0 && Model.get(minI).label_flg == 1) {
            Model.get(minJ).label_flg = 1;
            Model.get(minJ).label_flg = Model.get(minI).label;
        }
        // Model.remove(minI);
        Model.get(minJ).time = currTime;
    }

    public ArrayList<Integer> classify(SyncObject inst, ArrayList<MicroCluster> Model) {
        ArrayList<Integer> p_label_idx = new ArrayList<Integer>();
        double minDistL = Double.MAX_VALUE;
        int idxL = 0;
        for (int i = 0; i < Model.size(); i++) {
            if (Model.get(i).label_flg == 1) {
                double dist = computeDistance(inst, Model.get(i));
                if (minDistL > dist) {
                    minDistL = dist;
                    idxL = i;
                }
            }
        }
        p_label_idx.add(Model.get(idxL).label);

        p_label_idx.add(idxL);
        return p_label_idx;
    }

    /**
     * synchronization-based dynamic clustering
     */
    public ArrayList<MicroCluster> clustering(ArrayList<SyncObject> data_orig, double eRange, int currTime) {

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
        ArrayList<MicroCluster> mcs = findSynCluster(data_orig, data_copy, eRange, currTime);

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
    public static ArrayList<MicroCluster> findSynCluster(ArrayList<SyncObject> origdata, ArrayList<SyncObject> data, double eR, int currTime) {

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
            int time = currTime;
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

    private void clearLabels(ArrayList<SyncObject> buffer) {
        Iterator<SyncObject> iterator = buffer.iterator();
        while (iterator.hasNext()) {
            SyncObject next = iterator.next();
            next.setLabel_flg(0);

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
