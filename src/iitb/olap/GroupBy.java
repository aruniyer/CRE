package iitb.olap;

import gnu.trove.list.array.TIntArrayList;
import iitb.data.IDataStore;
import iitb.data.IInstance;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

public class GroupBy {

    private SortedSet<Integer> attributes;
    private SortedMap<String, TIntArrayList> baseGroups;
    private Map<Integer, SortedSet<Number>> activeIDMap;
    private SortedMap<Integer, SortedMap<Number, SortedSet<String>>> idToBaseGroupMap;
    private Map<String, SortedMap<Integer, Number>> baseGroupAttrValMap;

    public GroupBy(IDataStore dataStore, Map<Integer, double[]> valueMap) {
        attributes = new TreeSet<Integer>(valueMap.keySet());
        baseGroups = new TreeMap<String, TIntArrayList>();
        activeIDMap = new HashMap<Integer, SortedSet<Number>>();
        idToBaseGroupMap = new TreeMap<Integer, SortedMap<Number, SortedSet<String>>>();
        baseGroupAttrValMap = new HashMap<String, SortedMap<Integer, Number>>();

        int i = 0;
        for (IInstance instance : dataStore) {
            StringBuilder baseGroupID = new StringBuilder();
            for (Integer attribute : attributes) {
                double[] values = valueMap.get(attribute);
                if (values == null)
                    baseGroupID.append(instance.get(attribute)).append(',');
                else {
                    double val = instance.get(attribute);
                    int id = -1;
                    for (id = 0; id < values.length; id++)
                        if (val < values[id])
                            break;
                    baseGroupID.append(id).append(',');
                }
            }
            SortedMap<Integer, Number> bg2IdMap = baseGroupAttrValMap.get(baseGroupID.toString());
            if (bg2IdMap == null) {
                bg2IdMap = new TreeMap<Integer, Number>();
                baseGroupAttrValMap.put(baseGroupID.toString(), bg2IdMap);
                for (Integer attribute : attributes)
                    bg2IdMap.put(attribute, null);
            }
            for (Integer attribute : attributes) {
                SortedSet<Number> activeIDs = activeIDMap.get(attribute);
                SortedMap<Number, SortedSet<String>> id2BgMap = idToBaseGroupMap.get(attribute);
                if (activeIDs == null) {
                    activeIDs = new TreeSet<Number>();
                    activeIDMap.put(attribute, activeIDs);
                    id2BgMap = new TreeMap<Number, SortedSet<String>>();
                    idToBaseGroupMap.put(attribute, id2BgMap);
                }
                double[] values = valueMap.get(attribute);
                if (values == null) {
                    activeIDs.add(instance.get(attribute));
                    SortedSet<String> baseGroupSet = id2BgMap.get(instance.get(attribute));
                    if (baseGroupSet == null) {
                        baseGroupSet = new TreeSet<String>();
                        id2BgMap.put(instance.get(attribute), baseGroupSet);
                        bg2IdMap.put(attribute, instance.get(attribute));
                    }
                    baseGroupSet.add(baseGroupID.toString());
                } else {
                    double val = instance.get(attribute);
                    int id = -1;
                    for (id = 0; id < values.length; id++)
                        if (val < values[id])
                            break;
                    activeIDs.add(id);
                    SortedSet<String> baseGroupSet = id2BgMap.get(id);
                    if (baseGroupSet == null) {
                        baseGroupSet = new TreeSet<String>();
                        id2BgMap.put(id, baseGroupSet);
                    }
                    bg2IdMap.put(attribute, id);
                    baseGroupSet.add(baseGroupID.toString());
                }
            }
            TIntArrayList lines = baseGroups.get(baseGroupID.toString());
            if (lines == null) {
                lines = new TIntArrayList();
                baseGroups.put(baseGroupID.toString(), lines);
            }
            lines.add(i);
            i++;
        }
    }

    public List<Integer> getAttributes() {
        return new LinkedList<Integer>(this.attributes);
    }

    public SortedMap<String, TIntArrayList> getBaseGroups() {
        return this.baseGroups;
    }

    public SortedSet<Number> getActiveIDs(Integer attribute) {
        return activeIDMap.get(attribute);
    }

    public SortedSet<String> getBaseGroups(Integer attribute, Number value) {
        return idToBaseGroupMap.get(attribute).get(value);
    }
    
    public SortedMap<Integer, Number> getAttrValForBaseGroup(String baseGroupID) {
        return this.baseGroupAttrValMap.get(baseGroupID);
    }

}
