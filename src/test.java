/**
 * Created by alparslan on 08/10/16.
 */
public class test {
    public static void main(String[] args){

        build_tagger bt = new build_tagger();
        String[] arr = new String[]{"sents.train", "sents.devt", "model_file"};
        bt.main(arr);


        /*
        run_tagger rt = new run_tagger();
        String[] arr2 = new String[]{"sents.test", "model_file", "sents.out"};
        rt.main(arr2);
        */
    }
}
