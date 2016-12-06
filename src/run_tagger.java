import java.io.*;

/**
 * Ahmet Alparslan Celik
 * Student ID: A0152801R
 */
public class run_tagger {
    // Files
    private static File testFile, modelFile, outputFile;

    // HMM POS model
    private static pos_model model;

    public static void main(String[] args){
        // Initialization
        initialize(args);

        // Model Loading
        loadHMMModel();

        // Run the model on the test data
        runModelOnTestData();
    }

    // == Parameter initialization ==
    public static void initialize(String[] args){
        // argument validation
        argumentCheck(args);

    }
    public static void argumentCheck(String[] args){
        if(args.length != 3) {
            System.err.println("usage: java build_tagger <sents.test> <model_file> <sents.out>");
            System.exit(1);
        }

        testFile = new File(args[0]);
        modelFile = new File(args[1]);
        outputFile = new File(args[2]);
    }

    // == Load the HMM model ==
    public static void loadHMMModel() {
        try {
            FileInputStream fs = new FileInputStream(modelFile);
            ObjectInputStream os = new ObjectInputStream(fs);
            try {
                model = (pos_model) os.readObject();
            } finally {
                os.close();
                fs.close();
            }
        } catch(IOException e) {
            e.printStackTrace();
        } catch(ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    // == Run the model on the test data ==
    public static void runModelOnTestData() {
        try {
            BufferedReader r = new BufferedReader(new FileReader(testFile));
            BufferedWriter w = new BufferedWriter(new FileWriter(outputFile));
            try {
                String line;
                while(( line = r.readLine() ) != null) {
                    String[] words = line.split(" ");
                    String[] predictedTags = model.viberti(words);

                    // write the results to the output file
                    for (int i = 0; i < words.length; i++)
                        w.write(((i == 0) ? "" : " ") + words[i] + "/" + predictedTags[i]);
                    w.newLine();
                }
            } finally {
                r.close();
                w.flush();
                w.close();
            }
        } catch(IOException e) {
            e.printStackTrace();
        }
    }
}
