package shef.mt.tools;

import java.util.HashMap;
import shef.mt.features.util.Sentence;

public class WordCountProcessor extends ResourceProcessor {

    @Override
    public void processNextSentence(Sentence target) {
        //Get tokens from sentence:
        String[] tokens = target.getTokens();

        //Create hash of alignments word->frequency:
        HashMap<String, Integer> counts = new HashMap<>();

        //Get word counts:
        for (String token : tokens) {
            if (counts.get(token) == null) {
                counts.put(token, 1);
            } else {
                counts.put(token, counts.get(token) + 1);
            }
        }

        //Add resource to sentence:
        target.setValue("wordcounts", counts);
    }
}
