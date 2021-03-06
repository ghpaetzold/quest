/**
 *
 */
package shef.mt.features.impl.bb;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

import shef.mt.features.impl.Feature;
import static shef.mt.features.impl.bb.Feature1052.size;
import shef.mt.features.util.Sentence;
import shef.mt.tools.LanguageModel;

/**
 * average trigram frequency in quartile 2 of frequency (lower frequency words)
 * in the corpus of the source sentence
 *
 * @author Catalina Hallett
 *
 */
public class Feature1055 extends Feature {

    static int size = 3;
    static int quart = 2;

    public Feature1055() {
        setIndex(1055);
        setDescription("average trigram frequency in quartile 2 of frequency (lower frequency words) in the corpus of the source sentence");
        HashSet res = new HashSet<String>();
        res.add("ngramcount");
//		res.add(FeatureExtractor.getPosTagger());
//		res.add(FeatureExtractor.getGiza());

        setResources(res);
    }

    /* (non-Javadoc)
     * @see wlv.mt.features.impl.Feature#run(wlv.mt.features.util.Sentence, wlv.mt.features.util.Sentence)
     */
    @Override
    public void run(Sentence source, Sentence target) {
        // TODO Auto-generated method stub
        ArrayList<String> ngrams = source.getNGrams(size);
        Iterator<String> it = ngrams.iterator();
        String ngram;
        int count = 0;
        int freq;
        int totalFreq = 0;
        LanguageModel lm = (LanguageModel) source.getValue("ngramcount");
        int cutOffLow = lm.getCutOff(size, quart - 1);
        int cutOffHigh = lm.getCutOff(size, quart);
        while (it.hasNext()) {
            ngram = it.next();
            freq = lm.getFreq(ngram, size);
            if (freq <= cutOffHigh && freq > cutOffLow) {
                count++;
            }
        }
        if (count == 0) {
            setValue(0);
        } else {
            setValue((float) count / ngrams.size());
        }
    }
}
