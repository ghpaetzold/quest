/**
 *
 */
package shef.mt.features.impl.bb;

import java.util.HashSet;
import java.util.StringTokenizer;

import shef.mt.features.impl.Feature;
import shef.mt.features.util.Sentence;
import shef.mt.tools.Giza;
import shef.mt.tools.Giza2;
import shef.mt.features.wce.Numerical;



/**
 * Number of stopwords between in target sentence
 *
 * @author Luong Ngoc Quang
 *
 *
 */
public class Feature5004 extends Feature {

    //final static Float probThresh = 0.10f;

    public Feature5004() {
        setIndex(5004);
        setDescription("Number of numericals in the source sentence");
        //HashSet res = new HashSet<String>();
        //res.add("Giza");
        //setResources(res);
    }

    /* (non-Javadoc)
     * @see wlv.mt.features.util.Feature#run(wlv.mt.features.util.Sentence, wlv.mt.features.util.Sentence)
     */
    @Override
    public void run(Sentence source, Sentence target) {
        // TODO Auto-generated method stub
        String text = source.getText();
        Numerical numcalculator = new Numerical(text);
        int result = numcalculator.calculate();

        setValue(result);
    }
}
