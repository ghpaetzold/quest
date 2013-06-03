/**
 *
 */
package shef.mt.features.impl.bb;

import java.util.HashSet;

import shef.mt.features.impl.Feature;
import shef.mt.features.util.Sentence;

/**
 * percentage of nouns in the source
 *
 * @author cat
 *
 */
public class Feature1088 extends Feature {

    public Feature1088() {
        setIndex(1088);
        setDescription("percentage of nouns in the source");
        HashSet res = new HashSet<String>();
        res.add("sourcePosTagger");
    }
    /* (non-Javadoc)
     * @see wlv.mt.features.impl.Feature#run(wlv.mt.features.util.Sentence, wlv.mt.features.util.Sentence)
     */

    @Override
    public void run(Sentence source, Sentence target) {
        // TODO Auto-generated method stub
        int noWords = source.getNoTokens();
        float noNouns = (Float) source.getValue("nouns");
        setValue(noNouns / noWords);

    }
}
