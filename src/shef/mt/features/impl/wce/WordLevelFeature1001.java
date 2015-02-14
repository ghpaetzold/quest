
package shef.mt.features.impl.wce;

import shef.mt.features.impl.WordLevelFeature;
import shef.mt.features.util.Sentence;

/**
 * Target word value feature.
 * @author GustavoH
 */
public class WordLevelFeature1001 extends WordLevelFeature{

    public WordLevelFeature1001() {
        this.setIndex("WCE1001");
        this.setIdentifier("TRGWORD");
        this.setDescription("Target word.");
    }

    @Override
    public void run(Sentence source, Sentence target) {
        String[] result = new String[target.getNoTokens()];
        String[] tokens = target.getTokens();
        for(int i=0; i<tokens.length; i++){
            String value = this.getIdentifier()+'='+tokens[i];
            result[i] = value;
        }
        this.setValues(result);
    }

}
