Installing Word-Level QuEst:
	1 - Place all QuEst files into a folder of your choice.
	2 - Download version 3.5.1 of Stanford Core NLP from http://nlp.stanford.edu/software/corenlp.shtml
	3 - Add the file "stanford-corenlp-3.5.1-models.jar" to "quest/lib"
	4 - For spanish models, download the file http://nlp.stanford.edu/software/stanford-spanish-corenlp-2015-01-08-models.jar and place it in "quest/lib"
	5 - For chinese models, download the file http://nlp.stanford.edu/software/stanford-chinese-corenlp-2015-01-30-models.jar and place it in "quest/lib"
	6 - Download the Universal Wordnet plugin from http://resources.mpi-inf.mpg.de/yago-naga/uwn/uwn.zip and unzip it into a folder of your choice.
	
	Obs:
		- The Universal Wordnet plugin folder is the one which should be referenced in the variable "tools.universalwordnet.path" in the config file.
		- The models for english, spanish and chinese are automatically recognized by QuEst if the aforementioned libraries are placed in the "quest/lib" folder.
		
Running Word-Level QuEst:
	1 - Create a configuration file following the example in "quest/config/config.wce.properties"
	2 - Create a feature configuration file following the example in "quest/config/features/wce_features_all.xml"
	3 - Prepare the source and target language input files for which you desire to estimate feature values
	4 - Run the following command line:
		java -cp QuEst.jar shef.mt.enes.WordLevelFeatureExtractor -lang <source_language> <target_language> -input <source_file> <target_file> -mode <selected_model> -config <config_file>
