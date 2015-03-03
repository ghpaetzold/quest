package shef.mt.features.impl.wce;

import edu.stanford.nlp.util.ArrayUtils;
import java.util.HashMap;
import shef.mt.features.impl.WordLevelFeature;
import shef.mt.features.util.Sentence;
import shef.mt.tools.LanguageModel;

public class WordLevelFeature1015 extends WordLevelFeature {

    public WordLevelFeature1015() {
        this.setIndex("WCE1015");
        this.setIdentifier("LMBACKOFF");
        this.setDescription("Language model backoff behavior value.");
        this.addResource("ngramcount");
    }

    @Override
    public void run(Sentence source, Sentence target) {
        //Create vector of resulting values:
        String[] result = new String[target.getNoTokens()];

        //Get alignments object:
        LanguageModel lm = (LanguageModel) target.getValue("ngramcount");

        //Ge tokens from target sentence:
        String[] targetTokens = target.getTokens();

        //Output word occurrences:
        for (int i = 0; i < targetTokens.length; i++) {
            //Get ngrams:
            String[] ngrams = this.getNgrams(targetTokens, i);
            String w2w1w = ngrams[0];
            String w2w1 = ngrams[1];
            String w1w = ngrams[2];
            String w1 = ngrams[3];
            String w = ngrams[4];
            
            //Get ngram frequencies:
            Integer w2w1wi = lm.getFreq(w2w1w, 3);
            Integer w2w1i = lm.getFreq(w2w1, 2);
            Integer w1wi = lm.getFreq(w1w, 2);
            Integer w1i = lm.getFreq(w1, 1);
            Integer wi = lm.getFreq(w, 1);
            
            //Estimate backoff value:
            int backoff = -1;
            if(w2w1wi!=0){
                backoff = 7;
            }else if(w2w1i!=0 && w1wi!=0){
                backoff = 6;
            }else if(w1wi!=0){
                backoff = 5;
            }else if(w2w1i!=0 && wi!=0){
                backoff = 4;
            }else if(w1i!=0 && wi!=0){
                backoff = 3;
            }else if(wi!=0){
                backoff = 2;
            }else{
                backoff = 1;
            }
            
            //Create value:
            String value = this.getIdentifier() + "=" + backoff;

            //Save value:
            result[i] = value;
        }

        //Save values produced:
        this.setValues(result);
    }

    private String[] getNgrams(String[] tokens, int i) {
        //Create fillers:
        String[] fillers = new String[]{"<s>", "<s>", "<s>"};
        String[] allTokens = ArrayUtils.concatenate(fillers, tokens);
        
        //Ajust word index:
        int index = i + 3;
        
        //Create array of ngrams:
        String[] result = new String[5];
        result[0] = allTokens[index-2] + " " + allTokens[index-1] + " " + allTokens[index];
        result[1] = allTokens[index-2] + " " + allTokens[index-1];
        result[2] = allTokens[index-1] + " " + allTokens[index];
        result[3] = allTokens[index-1];
        result[4] = allTokens[index];
        
        //Return array of ngrams:
        return result;
    }
}
