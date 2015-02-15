package shef.mt.enes;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import shef.mt.features.util.Sentence;
import shef.mt.features.util.WordLevelFeatureManager;
import shef.mt.tools.AlignmentProcessor;
import shef.mt.tools.LanguageModel;
import shef.mt.tools.NGramExec;
import shef.mt.tools.NGramProcessor;
import shef.mt.tools.NgramCountProcessor;
import shef.mt.tools.PPLProcessor;
import shef.mt.tools.ParsingProcessor;
import shef.mt.tools.SenseProcessor;
import shef.mt.util.NGramSorter;
import shef.mt.util.PropertiesManager;
import wlv.mt.tools.ResourceManager;

/**
 * Main class for the word-level feature extraction pipeline.
 * @author GustavoH
 */
public class WordLevelFeatureExtractor {

    private String workDir;
    private String input;
    private String output;
    private String features;

    private String sourceFile;
    private String targetFile;
    private String sourceLang;
    private String targetLang;

    private PropertiesManager resourceManager;
    private WordLevelFeatureManager featureManager;
    private String configPath;
    private String mod;

    private StanfordCoreNLP sourcePipe;
    private StanfordCoreNLP targetPipe;

    public WordLevelFeatureExtractor(String[] args) {
        //Parse command line arguments:
        this.parseArguments(args);

        //Setup main folders:
        workDir = System.getProperty("user.dir");
        input = workDir + File.separator + resourceManager.getString("input");
        output = workDir + File.separator + resourceManager.getString("output");
        System.out.println("Work dir: " + workDir);
        System.out.println("Input folder: " + input);
        System.out.println("Output folder: " + output);
    }

    public static void main(String[] args) {
        //Measure initial time:
        long start = System.currentTimeMillis();

        //Run word-level feature extractor:
        WordLevelFeatureExtractor wfe = new WordLevelFeatureExtractor(args);
        wfe.run();

        //Measure ending time:
        long end = System.currentTimeMillis();
        System.out.println("processing completed in " + (end - start) / 1000 + " sec");
    }

    public void run() {
        //Set output writer for feature values:
        String outputPath = this.output + File.separator + "output.txt";
        BufferedWriter outWriter = null;
        try {
            outWriter = new BufferedWriter(new FileWriter(outputPath));
        } catch (IOException ex) {
            Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        //Build input and output folders:
        this.constructFolders();

        //Lowercase input files:
        this.preProcess();

        //Produce missing resources:
        this.produceMissingResources();

        //Run SRILM on language models:
        PPLProcessor[] pplProcessors = this.getLMProcessors();
        PPLProcessor pplProcSource = pplProcessors[0];
        PPLProcessor pplProcTarget = pplProcessors[1];

        //Run SRILM on ngram count files:
        NgramCountProcessor[] ngramProcessors = this.getNgramProcessors();
        NgramCountProcessor ngramProcessorSource = ngramProcessors[0];
        NgramCountProcessor ngramProcessorTarget = ngramProcessors[1];

        //Get alignment processors:
        AlignmentProcessor alignmentProcessor = this.getAlignmentProcessor();

        //Get parsing processors:
        ParsingProcessor[] parsingProcessors = this.getParsingProcessors();
        ParsingProcessor parsingSource = parsingProcessors[0];
        ParsingProcessor parsingTarget = parsingProcessors[1];

        //Get sense processor:
        SenseProcessor senseProcessor = this.getSenseProcessor();

        try {
            //Get readers of source and target files input:
            BufferedReader sourceBR = new BufferedReader(new FileReader(this.sourceFile));
            BufferedReader targetBR = new BufferedReader(new FileReader(this.targetFile));
            
            //Process each sentence pair:
            int sentenceCounter = 0;
            while(sourceBR.ready() && targetBR.ready()){
                //Create source and target sentence objects:
                Sentence sourceSentence = new Sentence(sourceBR.readLine().trim(), sentenceCounter);
                Sentence targetSentence = new Sentence(targetBR.readLine().trim(), sentenceCounter);
                
                //Add language model perplexity information:
                pplProcSource.processNextSentence(sourceSentence);
                pplProcTarget.processNextSentence(targetSentence);
                
                //Add ngram counts information:
                ngramProcessorSource.processNextSentence(sourceSentence);
                ngramProcessorSource.processNextSentence(targetSentence);
                
                //Add alignments information:
                alignmentProcessor.processNextSentence(targetSentence);
                
                //Add POS tags and dependency parses information:
                parsingSource.processNextSentence(sourceSentence);
                parsingTarget.processNextSentence(targetSentence);
                
                //Add sense information:
                senseProcessor.processNextSentence(targetSentence);
                
                //Run features for sentence pair:
                String featureValues = featureManager.runFeatures(sourceSentence, targetSentence).trim();
                outWriter.write(featureValues);
                outWriter.newLine();
                
                //Increase sentence counter:
                sentenceCounter++;
            }
            
            //Save output:
            outWriter.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }

    public void constructFolders() {
        //Create input folders:
        File f = new File(input);
        if (!f.exists()) {
            f.mkdirs();
            System.out.println("Input folder created " + f.getPath());
        }
        f = new File(input + File.separator + sourceLang);
        if (!f.exists()) {
            f.mkdirs();
            System.out.println("Input folder created " + f.getPath());
        }
        f = new File(input + File.separator + targetLang);
        if (!f.exists()) {
            f.mkdirs();
            System.out.println("Input folder created " + f.getPath());
        }
        f = new File(input + File.separator + targetLang + File.separator + "temp");
        if (!f.exists()) {
            f.mkdirs();
            System.out.println("Input folder created " + f.getPath());
        }

        //Create output folders:
        String output = resourceManager.getString("output");
        f = new File(output);
        if (!f.exists()) {
            f.mkdirs();
            System.out.println("Output folder created " + f.getPath());
        }
    }

    private void preProcess() {
        //Create input and output paths:
        String sourceInputFolder = input + File.separator + sourceLang;
        String targetInputFolder = input + File.separator + targetLang;

        File origSourceFile = new File(sourceFile);
        File inputSourceFile = new File(sourceInputFolder + File.separator + origSourceFile.getName());
        File origTargetFile = new File(targetFile);
        File inputTargetFile = new File(targetInputFolder + File.separator + origTargetFile.getName());

        //Create copy of original input files to input folder:
        try {
            System.out.println("Copying source input to: " + inputSourceFile.getPath());
            System.out.println("copying target input to: " + inputTargetFile.getPath());
            this.copyFile(origSourceFile, inputSourceFile);
            this.copyFile(origTargetFile, inputTargetFile);
        } catch (IOException e) {
            return;
        }

        //Lowercase copied input files:
        String sourceOutput = inputSourceFile + ".lower";
        String targetOutput = inputTargetFile + ".lower";

        try {
            BufferedReader sourceBR = new BufferedReader(new FileReader(inputSourceFile));
            BufferedReader targetBR = new BufferedReader(new FileReader(inputTargetFile));

            BufferedWriter sourceBW = new BufferedWriter(new FileWriter(sourceOutput));
            BufferedWriter targetBW = new BufferedWriter(new FileWriter(targetOutput));

            while (sourceBR.ready()) {
                String sourceSentence = sourceBR.readLine().trim();
                String targetSentence = targetBR.readLine().trim();

                sourceBW.write(sourceSentence.toLowerCase());
                targetBW.write(targetSentence.toLowerCase());

                sourceBW.newLine();
                targetBW.newLine();
            }

            sourceBR.close();
            targetBR.close();
            sourceBW.close();
            targetBW.close();

            //Update input paths:
            this.sourceFile = sourceOutput;
            this.targetFile = targetOutput;
        } catch (FileNotFoundException ex) {
            Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void parseArguments(String[] args) {

        Option help = OptionBuilder.withArgName("help").hasArg()
                .withDescription("print project help information")
                .isRequired(false).create("help");

        Option input = OptionBuilder.withArgName("input").hasArgs(3)
                .isRequired(true).create("input");

        Option lang = OptionBuilder.withArgName("lang").hasArgs(2)
                .isRequired(false).create("lang");

        Option feat = OptionBuilder.withArgName("feat").hasArgs(1)
                .isRequired(false).create("feat");

        Option gb = OptionBuilder.withArgName("gb")
                .withDescription("GlassBox input files").hasOptionalArgs(2)
                .hasArgs(3).create("gb");

        Option mode = OptionBuilder
                .withArgName("mode")
                .withDescription("blackbox features, glassbox features or both")
                .hasArgs(1).isRequired(true).create("mode");

        Option config = OptionBuilder
                .withArgName("config")
                .withDescription("cofiguration file")
                .hasArgs(1).isRequired(false).create("config");

        CommandLineParser parser = new PosixParser();
        Options options = new Options();
        options.addOption(help);
        options.addOption(input);
        options.addOption(mode);
        options.addOption(lang);
        options.addOption(feat);
        options.addOption(gb);
        options.addOption(config);

        try {
            CommandLine line = parser.parse(options, args);

            if (line.hasOption("config")) {
                resourceManager = new PropertiesManager(line.getOptionValue("config"));
            } else {
                resourceManager = new PropertiesManager();
            }

            if (line.hasOption("input")) {
                String[] files = line.getOptionValues("input");
                sourceFile = files[0];
                targetFile = files[1];
            }

            if (line.hasOption("lang")) {
                String[] langs = line.getOptionValues("lang");
                sourceLang = langs[0];
                targetLang = langs[1];
            } else {
                sourceLang = resourceManager.getString("sourceLang.default");
                targetLang = resourceManager.getString("targetLang.default");
            }

            if (line.hasOption("mode")) {
                String[] modeOpt = line.getOptionValues("mode");
                setMod(modeOpt[0].trim());
                configPath = resourceManager.getString("featureConfig." + getMod());
                featureManager = new WordLevelFeatureManager(configPath);
            }

            if (line.hasOption("feat")) {
                // print the value of block-size
                features = line.getOptionValue("feat");
                featureManager.setFeatureList(features);
            } else {
                featureManager.setFeatureList("all");
            }

        } catch (ParseException exp) {
            System.out.println("Unexpected exception:" + exp.getMessage());
        }
    }

    private void copyFile(File sourceFile, File destFile) throws IOException {
        if (sourceFile.equals(destFile)) {
            return;
        }

        if (!destFile.exists()) {
            destFile.createNewFile();
        }

        java.nio.channels.FileChannel source = null;
        java.nio.channels.FileChannel destination = null;
        try {
            source = new FileInputStream(sourceFile).getChannel();
            destination = new FileOutputStream(destFile).getChannel();
            destination.transferFrom(source, 0, source.size());
        } finally {
            if (source != null) {
                source.close();
            }
            if (destination != null) {
                destination.close();
            }
        }
    }

    public String getMod() {
        return mod;
    }

    public void setMod(String mod) {
        this.mod = mod;
    }

    private PPLProcessor[] getLMProcessors() {
        //Generate output paths:
        String sourceOutput = this.sourceFile + ".ppl";
        String targetOutput = this.targetFile + ".ppl";

        //Read language models:
        NGramExec nge = new NGramExec(resourceManager.getString("tools.ngram.path"), true);

        //Get paths of LMs:
        String sourceLM = resourceManager.getString(sourceLang + ".lm");
        String targetLM = resourceManager.getString(targetLang + ".lm");

        //Run LM reader:
        System.out.println("Running SRILM...");
        System.out.println(sourceFile);
        System.out.println(targetFile);
        nge.runNGramPerplex(sourceFile, sourceOutput, sourceLM);
        nge.runNGramPerplex(targetFile, targetOutput, targetLM);
        System.out.println("SRILM finished!");

        //Generate PPL processors:
        PPLProcessor pplProcSource = new PPLProcessor(sourceOutput,
                new String[]{"logprob", "ppl", "ppl1"});
        PPLProcessor pplProcTarget = new PPLProcessor(targetOutput,
                new String[]{"logprob", "ppl", "ppl1"});

        //Return processors:
        return new PPLProcessor[]{pplProcSource, pplProcTarget};
    }

    private LanguageModel[] getNGramModels() {
        //Create ngram file processors:
        NGramProcessor sourceNgp = new NGramProcessor(resourceManager.getString(sourceLang + ".ngram"));
        NGramProcessor targetNgp = new NGramProcessor(resourceManager.getString(targetLang + ".ngram"));

        //Generate resulting handlers:
        LanguageModel[] result = new LanguageModel[]{sourceNgp.run(), targetNgp.run()};

        //Return handlers:
        return result;
    }

    private AlignmentProcessor getAlignmentProcessor() {
        //Register feature:
        ResourceManager.registerResource("alignments");
        
        //Get path to alignments file:
        String alignmentsPath = resourceManager.getProperty("alignments.file");

        //Return AlignmentProcessor:
        return new AlignmentProcessor(alignmentsPath);
    }

    private void runFastAlign(String inputPath, String outputPath) {
        //Generate path for fast_align:
        String fast_align = resourceManager.getProperty("tools.fast_align.path") + File.separator + "fast_align";

        //Create arguments:
        String[] args = new String[]{
            fast_align,
            "-i",
            inputPath,
            "-d",
            "-o",
            "-v",
            ">",
            outputPath};

        //Run fast_align:
        System.out.println("Running fast_align...");
        try {
            Process process = Runtime.getRuntime().exec(args);
            process.waitFor();
            System.out.println("Created alignment file at: " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void produceMissingResources() {
        //Check if alignment file is missing:
        if (resourceManager.getProperty("alignments.file") == null) {
            if (resourceManager.getProperty("tools.fast_align.path") != null) {
                //Create fast_align input file:
                String inputPath = resourceManager.getProperty("resourcesPath") + File.separator + "source_to_target.inp";

                //Create fast_align output file:
                String outputPath = resourceManager.getProperty("resourcesPath") + File.separator + "source_to_target.out";

                try {
                    BufferedReader sourceBR = new BufferedReader(new FileReader(this.sourceFile));
                    BufferedReader targetBR = new BufferedReader(new FileReader(this.targetFile));

                    BufferedWriter outputBW = new BufferedWriter(new FileWriter(inputPath));

                    while (sourceBR.ready()) {
                        String sourceSentence = sourceBR.readLine().trim();
                        String targetSentence = targetBR.readLine().trim();

                        outputBW.write(sourceSentence + " ||| " + targetSentence);
                        outputBW.newLine();
                    }

                    sourceBR.close();
                    targetBR.close();
                    outputBW.close();

                    //Run fast align on input file:
                    this.runFastAlign(inputPath, outputPath);

                    //Return resulting processor:
                    resourceManager.put("alignments.file", outputPath);

                } catch (FileNotFoundException ex) {
                    Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(WordLevelFeatureExtractor.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }

        //Check if source LM is missing:
        if (resourceManager.getProperty(sourceLang + ".lm") == null) {
            if (resourceManager.getProperty("tools.ngram.path") != null) {
                if (resourceManager.getProperty(sourceLang + ".corpus") != null) {
                    if (resourceManager.getProperty("resourcesPath") != null) {
                        System.out.println("Building LM for the " + sourceLang + " language...");
                        System.out.println("Corpus used: " + resourceManager.getProperty(sourceLang + ".corpus"));
                        String[] args = new String[]{
                            resourceManager.getProperty("tools.ngram.path") + File.separator + "ngram-count",
                            "-order",
                            resourceManager.getProperty("ngramsize"),
                            "-text",
                            resourceManager.getProperty(sourceLang + ".corpus"),
                            "-lm",
                            resourceManager.getProperty("resourcesPath")
                            + "/" + sourceLang + "_lm.lm"};
                        try {
                            Process process = Runtime.getRuntime().exec(args);
                            process.waitFor();
                            resourceManager.setProperty(sourceLang + ".lm",
                                    resourceManager.getProperty("resourcesPath")
                                    + File.separator + sourceLang + "_lm.lm");
                            System.out.println("LM successfully built! Saved at: " + resourceManager.getProperty("resourcesPath")
                                    + File.separator + sourceLang + "_lm.lm");
                        } catch (IOException e) {
                            System.out.println("Error running SRILM");
                            e.printStackTrace();
                        } catch (InterruptedException e) {
                            System.out.println("Error waiting for SRILM to finish its execution.");
                            e.printStackTrace();
                        }
                    } else {
                        System.out.println("Missing source Language Model and resources path is not defined!");
                    }
                } else {
                    System.out.println("Missing source Language Model and corpus is not available!");
                }
            } else {
                System.out.println("Missing source Language Model and SRILM is not available!");
            }
        }

        //Check if target LM is missing:
        if (resourceManager.getProperty(targetLang + ".lm") == null) {
            if (resourceManager.getProperty("tools.ngram.path") != null) {
                if (resourceManager.getProperty(targetLang + ".corpus") != null) {
                    if (resourceManager.getProperty("resourcesPath") != null) {
                        System.out.println("Building LM for the " + targetLang + " language...");
                        System.out.println("Corpus used: " + resourceManager.getProperty(targetLang + ".corpus"));
                        String[] args = new String[]{
                            resourceManager.getProperty("tools.ngram.path") + File.separator + "ngram-count",
                            "-order",
                            resourceManager.getProperty("ngramsize"),
                            "-text",
                            resourceManager.getProperty(targetLang + ".corpus"),
                            "-lm",
                            resourceManager.getProperty("resourcesPath")
                            + File.separator + targetLang + "_lm.lm"};
                        try {
                            Process process = Runtime.getRuntime().exec(args);
                            process.waitFor();
                            resourceManager.setProperty(targetLang + ".lm",
                                    resourceManager.getProperty("resourcesPath")
                                    + File.separator + targetLang + "_lm.lm");
                            System.out.println("LM successfully built! Saved at: " + resourceManager.getProperty("resourcesPath")
                                    + File.separator + targetLang + "_lm.lm");
                        } catch (IOException e) {
                            System.out.println("Error running SRILM");
                            e.printStackTrace();
                        } catch (InterruptedException e) {
                            System.out.println("Error waiting for SRILM to finish its execution.");
                            e.printStackTrace();
                        }
                    } else {
                        System.out.println("Missing source Language Model and resources path is not defined!");
                    }
                } else {
                    System.out.println("Missing source Language Model and corpus is not available!");
                }
            } else {
                System.out.println("Missing source Language Model and SRILM is not available!");
            }
        }

        //Check if source NGRAM file is missing:
        if (resourceManager.getProperty(sourceLang + ".ngram") == null) {
            if (resourceManager.getProperty("tools.ngram.path") != null) {
                if (resourceManager.getProperty(sourceLang + ".corpus") != null) {
                    if (resourceManager.getProperty("resourcesPath") != null) {
                        System.out.println("Building NGRAM file for the " + sourceLang + " language...");
                        System.out.println("Corpus used: " + resourceManager.getProperty(sourceLang + ".corpus"));
                        String[] args = new String[]{
                            resourceManager.getProperty("tools.ngram.path") + File.separator + "ngram-count",
                            "-order",
                            resourceManager.getProperty("ngramsize"),
                            "-text",
                            resourceManager.getProperty(sourceLang + ".corpus"),
                            "-write",
                            resourceManager.getProperty("resourcesPath")
                            + File.separator + sourceLang + "_ngram.ngram"};
                        try {
                            Process process = Runtime.getRuntime().exec(args);
                            process.waitFor();

                            String spath = resourceManager.getProperty("resourcesPath") + "/" + sourceLang + "_ngram.ngram";
                            NGramSorter.run(spath, 4, Integer.parseInt(resourceManager.getProperty("ngramsize")), 2, spath);

                            resourceManager.setProperty(sourceLang + ".ngram", spath);
                            resourceManager.setProperty(sourceLang + ".ngram", spath + ".clean");
                            System.out.println("NGRAM successfully built! Saved at: " + spath + ".clean");
                        } catch (IOException e) {
                            System.out.println("Error running SRILM");
                            e.printStackTrace();
                        } catch (InterruptedException e) {
                            System.out.println("Error waiting for SRILM to finish its execution.");
                            e.printStackTrace();
                        }
                    } else {
                        System.out.println("Missing source NGRAM file and resources path is not defined!");
                    }
                } else {
                    System.out.println("Missing source NGRAM file and corpus is not available!");
                }
            } else {
                System.out.println("Missing source NGRAM file and SRILM is not available!");
            }
        }

        //Check if target NGRAM file is missing:
        if (resourceManager.getProperty(targetLang + ".ngram") == null) {
            if (resourceManager.getProperty("tools.ngram.path") != null) {
                if (resourceManager.getProperty(targetLang + ".corpus") != null) {
                    if (resourceManager.getProperty("resourcesPath") != null) {
                        System.out.println("Building NGRAM file for the " + targetLang + " language...");
                        System.out.println("Corpus used: " + resourceManager.getProperty(targetLang + ".corpus"));
                        String[] args = new String[]{
                            resourceManager.getProperty("tools.ngram.path") + File.separator + "ngram-count",
                            "-order",
                            resourceManager.getProperty("ngramsize"),
                            "-text",
                            resourceManager.getProperty(targetLang + ".corpus"),
                            "-write",
                            resourceManager.getProperty("resourcesPath")
                            + File.separator + targetLang + "_ngram.ngram"};
                        try {
                            Process process = Runtime.getRuntime().exec(args);
                            process.waitFor();

                            String spath = resourceManager.getProperty("resourcesPath") + File.separator + targetLang + "_ngram.ngram";
                            NGramSorter.run(spath, 4, Integer.parseInt(resourceManager.getProperty("ngramsize")), 2, spath);

                            resourceManager.setProperty(targetLang + ".ngram", spath);
                            resourceManager.setProperty(targetLang + ".ngram", spath + ".clean");
                            System.out.println("NGRAM successfully built! Saved at: " + spath + ".clean");
                        } catch (IOException e) {
                            System.out.println("Error running SRILM");
                            e.printStackTrace();
                        } catch (InterruptedException e) {
                            System.out.println("Error waiting for SRILM to finish its execution.");
                            e.printStackTrace();
                        }
                    } else {
                        System.out.println("Missing source NGRAM file and resources path is not defined!");
                    }
                } else {
                    System.out.println("Missing source NGRAM file and corpus is not available!");
                }
            } else {
                System.out.println("Missing source NGRAM file and SRILM is not available!");
            }
        }
    }

    private ParsingProcessor[] getParsingProcessors() {
        //Register resources:
        ResourceManager.registerResource("postags");
        ResourceManager.registerResource("depcounts");
        
        //Get paths to Stanford Parser source language models:
        String POSModel = (String) resourceManager.getProperty(this.sourceLang + "POSModel");
        String parseModel = (String) resourceManager.getProperty(this.sourceLang + "parseModel");

        //Create source language ParsingProcessor:
        ParsingProcessor sourceProcessor = new ParsingProcessor(this.sourceLang, POSModel, parseModel);

        //Get paths to Stanford Parser target language models:
        POSModel = (String) resourceManager.getProperty(this.targetLang + "POSModel");
        parseModel = (String) resourceManager.getProperty(this.targetLang + "parseModel");

        //Create target language ParsingProcessor:
        ParsingProcessor targetProcessor = new ParsingProcessor(this.targetLang, POSModel, parseModel);

        //Return processors:
        return new ParsingProcessor[]{sourceProcessor, targetProcessor};
    }

    private SenseProcessor getSenseProcessor() {
        //Register resource:
        ResourceManager.registerResource("sensecounts");
        //Get path to Universal Wordnet:
        String wordnetPath = this.resourceManager.getProperty("tools.universalwordnet.path");

        //Create SenseProcessor object:
        SenseProcessor sp = new SenseProcessor(wordnetPath, this.targetLang);

        //Return object:
        return sp;
    }

    private NgramCountProcessor[] getNgramProcessors() {
        //Register resource:
        ResourceManager.registerResource("ngramcount");
        
        //Get source and target Language Models:
        LanguageModel[] ngramModels = this.getNGramModels();
        LanguageModel ngramModelSource = ngramModels[0];
        LanguageModel ngramModelTarget = ngramModels[1];
        
        //Create source and target processors:
        NgramCountProcessor sourceProcessor = new NgramCountProcessor(ngramModelSource);
        NgramCountProcessor targetProcessor = new NgramCountProcessor(ngramModelTarget);
        NgramCountProcessor[] result = new NgramCountProcessor[]{sourceProcessor, targetProcessor};
        
        //Return processors:
        return result;
    }
}
