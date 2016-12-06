import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.*;

/**
 * Ahmet Alparslan Celik
 * Student ID: A0152801R
 */
public class pos_model implements Serializable{
    private static final long serialVersionUID = 5318308582228573844L;

    private final String startTag;
    private final String endTag;
    private final String unkWordTag;
    private final Set<String> tagSpaceSet;
    private final List<String> tagSpaceList;

    // For the interpolation smoothing
    // Initially they will be zero
    private double lambda1, lambda2;

    private final Map<String, Integer> singletonCounts;
    private final int totalNumOfTokens;
    private final Map<String, HashMap<String, Double>> transitions;
    private final Map<String, HashMap<String, Double>> observations;

    public pos_model( Set<String> tagSpaceSet, List<String> tagSpaceList, String startTag, String endTag, String unkWordTag,
                        Map<String, HashMap<String, Double>> transitions,
                        Map<String, HashMap<String, Double>> observations,
                        Map<String, Integer> singletonCounts, int totalNumOfTokens,
                        double lambda1, double lambda2){
        this.tagSpaceSet = tagSpaceSet;
        this.tagSpaceList = tagSpaceList;
        this.startTag = startTag;
        this.endTag = endTag;
        this.unkWordTag = unkWordTag;
        this.transitions = transitions;
        this.observations = observations;
        this.singletonCounts = singletonCounts;
        this.totalNumOfTokens = totalNumOfTokens;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }

    // === To calculate transaction probability ===
    public Double getTransitionProb(String prev, String curr){
        // If the tag is not a valid POS tag, throw an excaption
        validateTag(prev);
        validateTag(curr);


        //if (transitions.get(prev).get(curr) == 0) return 0.0;
        //else return (double) transitions.get(prev).get(curr) / getTotalNumOfTagStatesTransactionsObserved(prev);

        // If there is no parameter update, calculate probabilities without smoothing
        if(lambda1 == 0.0 && lambda2 == 0.0)
            return (double) transitions.get(prev).get(curr) / getTotalNumOfTagStatesTransactionsObserved(prev);
        // Else, calculate the probabilities using interpolation smoothing
        else {
            if(curr.equals(endTag))
                return (double) transitions.get(prev).get(curr) / getTotalNumOfTagStatesTransactionsObserved(prev);
            else
                return lambda1 * (singletonCounts.get(curr) / totalNumOfTokens) +
                        lambda2 * ((double) transitions.get(prev).get(curr) / getTotalNumOfTagStatesTransactionsObserved(prev));
        }
    }
    // Calculates marginal count of C(t_i) for a particular tag
    public Double getTotalNumOfTagStatesTransactionsObserved(String tag){
        // get the particular tag state
        HashMap<String, Double> tagState = transitions.get(tag);

        // return total number of tag states for a particular state
        double ctr = 0;
        for(Map.Entry<String, Double> e : tagState.entrySet())
            ctr += e.getValue();
        return ctr;
    }

    // === To calculate emission probability ===
    public Double getEmissionProb(String word, String tag){
        // If the tag is not a valid POS tag, throw an excaption
        validateTag(tag);

        if(!observations.get(tag).containsKey(word) && observations.get(tag).containsKey(unkWordTag))
            return (double) observations.get(tag).get(unkWordTag) / getTotalNumOfWordsObserved(tag);
        if(!observations.get(tag).containsKey(word) && !observations.get(tag).containsKey(unkWordTag))
            return 0.0;
        else
            return (double) observations.get(tag).get(word) / getTotalNumOfWordsObserved(tag);
    }
    // Calculates marginal count of C(w_i) for a particular tag
    private Double getTotalNumOfWordsObserved(String tag){
        // get the particular state
        HashMap<String, Double> tagState = observations.get(tag);

        // return total number of words for a particular state
        double ctr = 0;
        for(Map.Entry<String, Double> e : tagState.entrySet())
            ctr += e.getValue();
        return ctr;
    }

    // === Smoothing ===
    public void laplaceSmoothingWithUnknownWordTagForObservations(String tagForUnknownWords, double B){
        for(Map.Entry<String,HashMap<String, Double>> e : observations.entrySet())
            e.getValue().put(tagForUnknownWords, B);
    }
    public void interpolationSmoothingForTransitions(double lambda1, double lambda2){
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }

    // === Calculate the best path for a sequence of words ===
    // Determine the best path by using Viberti algorithm
    public String[] viberti(String[] words){
        // In order to follow the pseudo code convention
        // we define N which is the different number of states
        // alse T which is the number of observations in the given word sequence
        final int N = tagSpaceList.size();
        final int T = words.length;

        Double[][] viberti = new Double[N + 1][T];
        Integer[][] backPointer = new Integer[N + 1][T];

        // initialization step
        for(int s = 0; s < N; s++)
            viberti[s][0] = Math.log(getTransitionProb(startTag, tagSpaceList.get(s))) +
                    Math.log(getEmissionProb(words[0], tagSpaceList.get(s)));

        // recursion step
        for(int t = 1; t < T; t++)
            for(int s = 0; s < N; s++){
                double[] logProbSoFar = getVibertiMaxLogProbSoFar(viberti, s, t, false);
                viberti[s][t] = logProbSoFar[0]+ Math.log(getEmissionProb(words[t], tagSpaceList.get(s)));
                backPointer[s][t - 1] = (int)logProbSoFar[1];
            }

        // termination step
        viberti[N][T - 1] = getVibertiMaxLogProbSoFar(viberti, -1, T - 1, true)[0];
        backPointer[N][T - 1] = (int)getVibertiMaxLogProbSoFar(viberti, -1, T - 1, true)[1];

        // backtracing the path
        return backTracing(backPointer, N, T);
    }
    private double[] getVibertiMaxLogProbSoFar(Double[][] viberti, int stateIndex, int observationIndex, boolean endState){
        int maxProbIndex = 0;
        Double maxProb = (!endState) ? viberti[maxProbIndex][observationIndex - 1] + Math.log(getTransitionProb(tagSpaceList.get(maxProbIndex), tagSpaceList.get(stateIndex))) :
                                    viberti[maxProbIndex][observationIndex] + Math.log(getTransitionProb(tagSpaceList.get(maxProbIndex), endTag));


        // -1 is for extra space due to the starting and ending states
        for(int i = 1; i < viberti.length - 1; i++){
            Double currProb = (!endState) ? viberti[i][observationIndex - 1] + Math.log(getTransitionProb(tagSpaceList.get(i), tagSpaceList.get(stateIndex))) :
                                        viberti[i][observationIndex] + Math.log(getTransitionProb(tagSpaceList.get(i), endTag));

            if (currProb > maxProb) {
                maxProbIndex = i;
                maxProb = currProb;
            }
        }

        return new double[]{maxProb, (double) maxProbIndex};
    }
    private String[] backTracing(Integer[][] backPointerArray, final int N, final int T){
        String[] tags = new String[T];
        int backPointer = backPointerArray[N][T - 1];

        for(int t = T - 1; t >= 0; t--) {
            tags[t] = tagSpaceList.get(backPointer);
            backPointer = (t >= 1) ? backPointerArray[backPointer][t - 1] : 0;
        }

        return tags;
    }

    // === Check the validation of a POS PENN Treebank tag ===
    private boolean validateTag(String tag) {
        if(!tagSpaceSet.contains(tag) && !tag.equals(startTag) && !tag.equals(endTag))
            throw new IllegalArgumentException("\'" + tag + "\' is not a valid PENN Treebank tag.");

        return true;
    }

    // === For the serialization of the pos_model object ===
    private void readObject(ObjectInputStream inputStream) throws ClassNotFoundException, IOException {
        inputStream.defaultReadObject();
    }
    private void writeObject(ObjectOutputStream outputStream) throws IOException {
        outputStream.defaultWriteObject();
    }
}
