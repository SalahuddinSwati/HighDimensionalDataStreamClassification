
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Salahuddin
 */
public class Utils {
    /**
     * This function will read the data from file and split it into training and stream data
     * @param datasetPath
     * @param sep file seperator i,e. comma, tab
     * @param initial_number_of_instance_for_training 
     * @return
     * @throws IOException 
     */
    public static Data readData(String datasetPath, String sep, int initial_number_of_instance_for_training) throws IOException {
        ArrayList<SyncObject> train_buffer = new ArrayList<SyncObject>();
        ArrayList<SyncObject> stream_buffer = new ArrayList<SyncObject>();
        System.out.println("Reading Data...");
        int label_flg = 0;
        File mFile = new File(datasetPath);
        FileReader fr = new FileReader(mFile);
        BufferedReader br = new BufferedReader(fr);
        String line;
        int dim = 0;
        int counter = 0;
        while ((line = br.readLine()) != null) {
            counter++;
            line = line.trim();
            String[] str = line.split(sep);
            dim = str.length;
            double[] inst = new double[dim - 1];
            for (int i = 0; i < dim - 1; i++) {
                inst[i] = Float.parseFloat(str[i]);
            }

            SyncObject obj = new SyncObject(inst, (Integer.parseInt(str[dim - 1])), label_flg);
            if (counter <= initial_number_of_instance_for_training) {
                train_buffer.add(obj);
            } else {
                stream_buffer.add(obj);
            }
        }
        br.close();
        fr.close();
        return new Data(train_buffer, stream_buffer, dim);
    }
    
    public static void assignPartialLabel(ArrayList<SyncObject> list, int percentage){
        int noofsamples = (int) (list.size() * percentage / 100);
        HashSet<Integer> tnos = (HashSet<Integer>) generateRandomNumbers(0, list.size()-1, noofsamples);
        //System.out.println("no of labele per chunk = "+tnos.size());
        Iterator<Integer> randomIndexIterator = tnos.iterator();
        while (randomIndexIterator.hasNext()) {
            Integer randomIndex = randomIndexIterator.next();
            list.get(randomIndex).setLabel_flg(1);
        }
        //return tnos;
    }
    
    
    public static Set<Integer> generateRandomNumbers(int from, int to, int how_many_unique_numbers) {
        int range = to - from;
        HashSet<Integer> set = new HashSet<Integer>();
        while (set.size() < how_many_unique_numbers) {
            double randomDouble = Math.random();
            randomDouble = randomDouble * range + 1;
            int randomInt = (int) randomDouble;
            set.add(randomInt + from);
        }
        return set;
    }

    public static class Data {

        private ArrayList<SyncObject> train_buffer; //= new ArrayList<SyncObject>();
        private ArrayList<SyncObject> stream_buffer; //= new ArrayList<SyncObject>();
        private int dimensions;

        public Data(ArrayList<SyncObject> train_buffer, ArrayList<SyncObject> stream_buffer, int dimesion) {
            this.train_buffer = train_buffer;
            this.stream_buffer = stream_buffer;
            this.dimensions = dimesion;
        }

        public ArrayList<SyncObject> getStream_buffer() {
            return stream_buffer;
        }

        public ArrayList<SyncObject> getTrain_buffer() {
            return train_buffer;
        }

    }
    
    
    public static HashMap<Object, Object> sortByValues(HashMap<Object, Object> map) { 
       List list = new LinkedList(map.entrySet());
       // Defined Custom Comparator here
       Collections.sort(list, new Comparator() {
            public int compare(Object o1, Object o2) {
                int i = ((Comparable) ((Map.Entry) (o1)).getValue())
                  .compareTo(((Map.Entry) (o2)).getValue());
                
               return i;
            }
       });

       // Here I am copying the sorted list in HashMap
       // using LinkedHashMap to preserve the insertion order
       HashMap<Object,Object> sortedHashMap = new LinkedHashMap<Object,Object>();
       for (Iterator it = list.iterator(); it.hasNext();) {
              Map.Entry<Object,Object> entry = (Map.Entry) it.next();
              sortedHashMap.put(entry.getKey(), entry.getValue());
       } 
       return sortedHashMap;
  }

}
