/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package shef.mt.tools;

import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.POSTaggerAnnotator;
import edu.stanford.nlp.pipeline.ParserAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.TokenizerAnnotator;
import edu.stanford.nlp.pipeline.WordsToSentencesAnnotator;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import shef.mt.features.util.Sentence;

public class ParsingProcessor extends ResourceProcessor {

    private StanfordCoreNLP pipeline;

    private TokenizerAnnotator tokenizer;
    private POSTaggerAnnotator tagger;
    private ParserAnnotator parser;

    public ParsingProcessor(String lang, String pm, String dm) {
        String posModel = null;
        String depModel = null;

        //Setup model paths:
        if (pm == null) {
            if (lang.equals("english")) {
                posModel = "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger";
            } else if (lang.equals("spanish")) {
                posModel = "edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger";
            } else if (lang.equals("chinese")) {
                posModel = "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger";
            } else {
                posModel = "edu/stanford.nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger";
            }
        } else {
            posModel = pm;
        }
        if (dm == null) {
            if (lang.equals("english")) {
                depModel = "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz";
            } else if (lang.equals("spanish")) {
                depModel = "edu/stanford/nlp/models/lexparser/spanishPCFG.ser.gz";
            } else if (lang.equals("chinese")) {
                depModel = "edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz";
            } else {
                depModel = "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz";
            }
        } else {
            depModel = dm;
        }

        //Create base properties:
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, parse");

        //Create base pipeline:
        pipeline = new StanfordCoreNLP(props);

        //Create pipeline objects:
        tokenizer = new TokenizerAnnotator(true, TokenizerAnnotator.TokenizerType.Whitespace);
        tagger = new POSTaggerAnnotator(posModel, false);
        parser = new ParserAnnotator(depModel, false, 300, StringUtils.EMPTY_STRING_ARRAY);

        //Add objects to the pipeline:
        pipeline.addAnnotator(tokenizer);
        pipeline.addAnnotator(tagger);
        pipeline.addAnnotator(parser);
    }

    @Override
    public void processNextSentence(Sentence s) {
        //Create resource objects:
        ArrayList<String> POSData = new ArrayList<>();
        HashMap<Integer, Integer> depData = new HashMap<>();

        //Create content object:
        Annotation document = new Annotation(s.getText());

        //Annotate content object:
        pipeline.annotate(document);

        //Initialize index shift:
        int shift = 0;

        //Get sentence fragments:
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            //Get tokens from sentence fragment:
            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);

            //Add tokens to resulting POS tag list:
            for (CoreLabel token : tokens) {
                String pos = token.get(PartOfSpeechAnnotation.class);
                POSData.add(pos);
            }

            //Get dependency relations:
            SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
            List<SemanticGraphEdge> deps = dependencies.edgeListSorted();

            //For each edge, add 1 to occurrences of source and target indexes:
            for (SemanticGraphEdge sge : deps) {
                int sourceIndex = shift + sge.getSource().index() - 1;
                int targetIndex = shift + sge.getTarget().index() - 1;
                if (depData.get(sourceIndex) == null) {
                    depData.put(sourceIndex, 1);
                } else {
                    depData.put(sourceIndex, depData.get(sourceIndex) + 1);
                }
                if (depData.get(targetIndex) == null) {
                    depData.put(targetIndex, 1);
                } else {
                    depData.put(targetIndex, depData.get(targetIndex) + 1);
                }

                //Increase shift:
                shift += tokens.size();
            }
        }

        //Add resources to sentence:
        s.setValue("postags", POSData);
        s.setValue("depcounts", depData);
    }

}
