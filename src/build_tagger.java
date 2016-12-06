import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Ahmet Alparslan Celik
 * Student ID: A0152801R
 */
public class build_tagger {

    // Some tuning parameters
    private static final boolean USE_LOWER_CASE_LETTERS = true;

    // Files
    private static File trainingFile, validationFile, modelFile;

    // Tag space
    private static final String startTag = "<s>";
    private static final String endTag = "<end>";
    private static final String tagForUnknownWords = "<UNK>";
    private static final HashSet<String> tagSpace = new HashSet<>(Arrays.asList(
            "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
            "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT",
            "WP", "WP$", "WRB", "$", "#", "-LRB-", "-RRB-", ",", ".", ":",  "''", "``", "“", "”"));

    // HMM POS model
    private static pos_model model;

    // HMM parameters for training
    private static Map<String, HashMap<String, Double>> transitions;
    private static Map<String, HashMap<String, Double>> observations;
    private static Map<String, Integer> singetonTagCountForTrainingData;
    private static int totalNumOfTokensInTrainingData;

    // Interpolation
    private static Map<String, Integer> singetonTagCountForDevData;
    private static Map<String, Integer> bigramTagCount;
    private static double lambda1, lambda2;

    // Statistics
    private static Map<String, HashMap<String, Integer>> contingencyCount;
    private static int totalNumOfPredictions;
    private static int totalNumOfMisclassifications;


    public static void main(String[] args){
        // Initialization
        initialize(args);

        // Model training
        trainModelParameters();
        println("Model is trained!");

        // Create HMM model
        createHMMModel();
        println("Model is created!");

        // Initial smoothing for unknown words
        model.laplaceSmoothingWithUnknownWordTagForObservations(tagForUnknownWords, 0.01);

        // Validate the model
        validateModel();
        println("Initial validation is completed!");

        // Smoothing for bigram tag values
        deletedInterpolation();
        model.interpolationSmoothingForTransitions(lambda1, lambda2);
        println("Necessary parameters are updated for the bigram smoothing!\n" +
                "lambda1 = " + lambda1 + " & lambda2 = " + lambda2);

        // Save HMM model
        saveHMMModel();
        println("Model is saved!");

        // Validate the model after smoothing
        validateModel();
        println("Validation is completed!");

        // Print the statistics
        printStatistics();
    }

    // == Parameter initialization ==
    public static void initialize(String[] args){
        // argument validation
        argumentCheck(args);

        // -- HMM parameter initialization --
        transitions = new HashMap<>();
        observations = new HashMap<>();
        singetonTagCountForTrainingData = new HashMap<>();

        // -- initialize transition counts --
        for(String tag1 : tagSpace) {
            singetonTagCountForTrainingData.put(tag1, 0);
            transitions.put(tag1, new HashMap<>());
            for (String tag2 : tagSpace)
                transitions.get(tag1).put(tag2, 0.0);
            transitions.get(tag1).put(endTag, 0.0);
        }
        // adding start tag in the transition counts table
        transitions.put(startTag, new HashMap<>());
        for (String tag2 : tagSpace)
            transitions.get(startTag).put(tag2, 0.0);

        // -- initialize observation count --
        for(String tag : tagSpace)
            observations.put(tag, new HashMap<>());

        // -- initialize interpolation parameters --
        lambda1 = 0.0;
        lambda2 = 0.0;
        singetonTagCountForDevData = new HashMap<>();
        bigramTagCount = new HashMap<>();
        for(String tag : tagSpace)
            singetonTagCountForDevData.put(tag, 0);

        // -- initialize contingency table/confusion matrix --
        // this part is for the validation/development set
        contingencyCount = new HashMap<>();

        for(String tag1 : tagSpace){
            contingencyCount.put(tag1, new HashMap<>());
            for(String tag2 : tagSpace)
                contingencyCount.get(tag1).put(tag2, 0);
        }
        totalNumOfPredictions = 0;
        totalNumOfMisclassifications = 0;
    }
    public static void argumentCheck(String[] args){
        if(args.length != 3) {
            System.err.println("usage: java build_tagger <sents.train> <sents.devt> <model_file>");
            System.exit(1);
        }

        trainingFile = new File(args[0]);
        validationFile = new File(args[1]);
        modelFile = new File(args[2]);
    }

    // == Model parameter training ==
    public static void trainModelParameters(){
        try {
            BufferedReader r = new BufferedReader(new FileReader(trainingFile));
            totalNumOfTokensInTrainingData = 0;

            try {
                String s;
                while ((s = r.readLine()) != null) {
                    String tagPrev = startTag;
                    for(String t : s.split(" ")) {
                        int seperator = t.lastIndexOf('/');

                        if(seperator == -1) {
                            System.err.print("'" + t + "' is not properly tagged!");
                            continue;
                        }

                        // separate the word and the POS tag
                        String[] e = new String[]{t.substring(0, seperator), t.substring(seperator + 1)};

                        // convert digits to #
                        e[0] = e[0].replaceAll("\\d+", "\\#");
                        // convert word into lowercase
                        if(USE_LOWER_CASE_LETTERS)
                            e[0] = e[0].toLowerCase();

                        // increment the current tag's counter
                        incCount(singetonTagCountForTrainingData, e[1]);
                        totalNumOfTokensInTrainingData++;

                        // insert tuple into the observation table
                        insertObservation(e[e.length - 1], e[0]);

                        // insert tag bigrams into the transition table
                        insertTransition(tagPrev, e[e.length - 1]);
                        tagPrev = e[e.length - 1];
                    }
                    // Add end of the sentences tag into transition table
                    insertTransition(tagPrev, endTag);
                }
            } finally {
                r.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // == Create the HMM model ==
    public static void createHMMModel(){
        model = new pos_model(tagSpace, new ArrayList<String>(tagSpace), startTag, endTag, tagForUnknownWords,
                            transitions, observations, singetonTagCountForTrainingData, totalNumOfTokensInTrainingData,
                lambda1, lambda2);
    }

    // == Save the HMM model
    private static void saveHMMModel() {
        try {
            FileOutputStream fileStream = new FileOutputStream(modelFile);
            ObjectOutputStream objectStream = new ObjectOutputStream(fileStream);
            try {
                objectStream.writeObject(model);
            } finally {
                if (objectStream != null) {
                    objectStream.close();
                    fileStream.close();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // == Model validation ==
    public static void validateModel(){
        try {
            BufferedReader r = new BufferedReader(new FileReader(validationFile));
            try {
                println("Development data is being processed...\nIt'll take around 1 min\nLine number being processed now");
                int lineCtr = 0;

                String s;
                while ((s = r.readLine()) != null) {
                    println(++lineCtr);

                    String[] words = s.split(" ");
                    String[] trueTags = new String[words.length];
                    String[] predictedTags;

                    for(int i = 0; i < words.length; i++){
                        int seperator = words[i].lastIndexOf('/');

                        if (seperator == -1) {
                            System.err.print("'" + words[i] + "' is not properly tagged!");
                            continue;
                        }

                        // separate the word and the POS tag
                        trueTags[i] = words[i].substring(seperator + 1);
                        words[i] = words[i].substring(0, seperator);

                        // convert digits to #
                        words[i] = words[i].replaceAll("\\d+", "\\#");
                        // convert word into lowercase
                        if (USE_LOWER_CASE_LETTERS)
                            words[i] = words[i].toLowerCase();
                    }

                    // update interpolation parameters
                    String prevTag = trueTags[0];
                    incCount(singetonTagCountForDevData, trueTags[0]);
                    for(int i = 1; i < trueTags.length; i++){
                        incCount(bigramTagCount, (prevTag + " " + trueTags[i]));
                        incCount(singetonTagCountForDevData, trueTags[i]);
                        prevTag = trueTags[i];
                    }

                    // get the predicted tags
                    predictedTags = model.viberti(words);

                    // update contingency counts
                    for(int i = 0; i < trueTags.length; i++){
                        if(!predictedTags[i].equals(trueTags[i])){
                            contingencyCount.get(trueTags[i]).
                                    put(predictedTags[i], contingencyCount.get(trueTags[i]).get(predictedTags[i]) + 1);
                            totalNumOfMisclassifications++;
                        }
                        totalNumOfPredictions++;
                    }
                }
            } finally {
                r.close();
            }
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    // == Interpolation ==
    // Deleted interpolation algorithm is used
    // to calculate the lambda value
    // The tokens are gatherd using the development/validation data
    public static void deletedInterpolation(){
        lambda1 = 0;
        lambda2 = 0;

        // val1 = [C(t_2) - 1] / [N - 1] where N is the total # of tokens
        // val2 = [C(t_1, t_2) - 1] / [C(t_1) - 1]
        double val1, val2;

        for(Map.Entry<String, Integer> bigram : bigramTagCount.entrySet()){
            String[] bigramTags = bigram.getKey().split(" ");

            // totalNumOfPredictions is equal to total # of tokens
            val1 = (double)(singetonTagCountForDevData.get(bigramTags[1]) - 1) / (totalNumOfPredictions - 1);
            val2 = (singetonTagCountForDevData.get(bigramTags[0]) - 1 == 0) ? 0.0 :
                    (double)(bigram.getValue() - 1) / (singetonTagCountForDevData.get(bigramTags[0]) - 1);

            if(val1 > val2) lambda1 += bigram.getValue();
            else lambda2 += bigram.getValue();
        }

        // Normalization for the lambda values
        double lambdaSum = lambda1 + lambda2;
        lambda1 = lambda1 / lambdaSum;
        lambda2 = lambda2 / lambdaSum;
    }

    // == Statistics ==
    public static void printStatistics(){
        //Print the contingency table
        println("Confusion Matrix: ");
        printConfusionMatrix();

        // Print the accuracy
        System.out.println("\nAccuracy of the model: " + ((double)(totalNumOfPredictions - totalNumOfMisclassifications) / totalNumOfPredictions));
    }
    public static void printConfusionMatrix(){
        final List<String> tagSpaceList = new ArrayList<>(tagSpace);
        final Double[][] prob = new Double[tagSpaceList.size()][tagSpaceList.size()];

        // Calculate the probabilities
        for(int i = 0; i < tagSpaceList.size(); i++)
            for(int j = 0; j < tagSpaceList.size(); j++)
                prob[i][j] = (contingencyCount.get(tagSpaceList.get(i)).get(tagSpaceList.get(j)) == 0) ? 0.0 :
                        (double)contingencyCount.get(tagSpaceList.get(i)).get(tagSpaceList.get(j)) / totalNumOfPredictions;


        // Print the header of the table
        System.out.format("%-7s", "");
        for(String tag : tagSpaceList)
            System.out.format("%-7s", tag);
        println("");

        // For each tag print the probabilities
        for(int i = 0; i < tagSpaceList.size(); i++){
            System.out.format("%-7s", tagSpaceList.get(i));
            for(int j = 0; j < prob[i].length; j++)
                System.out.format("%-7s", ((prob[i][j] == 0) ? "-" : new DecimalFormat("#.####").format(prob[i][j])));
            println("");
        }
        println("");
    }

    // == Insert methods ==
    public static void insertTransition(String tagPrev, String tagCurr) {
        // Validate tags
        validateTag(tagPrev);
        validateTag(tagCurr);

        transitions.get(tagPrev).put(tagCurr, transitions.get(tagPrev).get(tagCurr) + 1);
    }
    public static void insertObservation(String tag, String word){
        // Validate tags
        validateTag(tag);

        if(!observations.get(tag).containsKey(word))
            observations.get(tag).put(word, 1.0);
        else
            observations.get(tag).put(word, observations.get(tag).get(word) + 1);
    }

    // === Check the validation of a POS PENN Treebank tag ===
    private static boolean validateTag(String tag) {
        if(!tagSpace.contains(tag) && !tag.equals(startTag) && !tag.equals(endTag))
            throw new IllegalArgumentException("\'" + tag + "\' is not a valid PENN Treebank tag.");

        return true;
    }

    // === Auxiliary functions ===
    private static void println(Object o){
        System.out.println(o.toString());
    }
    private static void incCount(Map<String, Integer> map, String token){
        if(!map.containsKey(token)) map.put(token, 1);
        else map.put(token, map.get(token) + 1);
    }
}
