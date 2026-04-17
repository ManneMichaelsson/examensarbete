package com.example.demo.ml;

import org.springframework.stereotype.Service;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import jakarta.annotation.PostConstruct;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

@Service
public class MLProcessorDL4J {

    private static final Logger logger = Logger.getLogger(MLProcessorDL4J.class.getName());
    private static final String FILE_PATH = "/corpus.csv";
    private final TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

    // FIX 1: Gör dessa till klassfält för att lösa scope-problem
    private Map<String, Integer> labelIndexMap;
    private List<String> sentences; // Behövs för att skapa DataSetIterator senare
    private List<String> rawIntentions;

    // FIX 2: Lägg till fält för den tränade modellen
    private Word2Vec word2VecModel;
    private MultiLayerNetwork neuralNetworkModel;
    private Map<Integer, String> indexToLabelMap; // Mappar index till intention (för 'Predict')

    private static final int MIN_WORD_FREQUENCY = 1;
    private static final int VECTOR_SIZE = 1000;
    private static final int N_EPOCHS = 2000;
    private static final int ITERATIONS = 5;
    private static final double dropOutScore = 0.3;
    private static final double learningRateScore = 0.001;

    @PostConstruct
    public void trainAndLoadModel() {
        try {
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

            // 1. Läs data och skapa sentence iterator (Resultatet fyller this.sentences)
            SentenceIterator word2VecIterator = loadCorpusAndCreateIterator(); // FIX 3: Nytt namn

            // 2. Träna word2vec (vektorisering)
            this.word2VecModel = new Word2Vec.Builder()
                    .minWordFrequency(MIN_WORD_FREQUENCY)
                    .iterations(ITERATIONS)
                    .layerSize(VECTOR_SIZE)
                    .seed(42)
                    .windowSize(5)
                    .iterate(word2VecIterator) // Använd den nya iteratorn
                    .tokenizerFactory(tokenizerFactory)
                    .build();

            word2VecModel.fit();
            logger.info("--- Word2Vec-modell tränad för feature extraction ---");

            // 3. SKAPA NUMERISK DATASET FÖR KLASSIFICERAREN
            final int numLabels = labelIndexMap.size();
            final int inputDim = VECTOR_SIZE;

            // 4: klassificerarens iterator
            DataSetIterator classifierIterator = DataSetIteratorFactory.create(
                    this.sentences, this.rawIntentions, this.labelIndexMap, this.word2VecModel, inputDim, this.tokenizerFactory);

            // 4. BYGG KLASSIFICERAREN (Ett enkelt Neuralt Nätverk)
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(42)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(learningRateScore))
                    .list()
                    // --- LAGER 0: INPUT (Tar emot 1000 siffror) ---
                    .layer(0, new DenseLayer.Builder()
                            .nIn(inputDim) // INPUT DIMENSION (1000)
                            .nOut(128)      // Går ut till 128 neuroner
                            .activation(Activation.RELU)
                            .dropOut(dropOutScore)
                            .build())
                    // --- LAGER 1: DOLT MELLANLAGER ---
                    .layer(1, new DenseLayer.Builder()
                            .nIn(128)      // INPUT från Lagret ovan
                            .nOut(64)       // Går ut till 64 neuroner
                            .activation(Activation.RELU)
                            .build())
                    // --- LAGER 2: OUTPUT (Klassificerar till 8 utgångar) ---
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nIn(64)      // INPUT från Lagret ovan
                            .nOut(numLabels)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));

            // 5. TRÄNA MODELLEN
            model.fit(classifierIterator, N_EPOCHS); // Använd den unika iteratorn

            logger.info("--- Neurala Nätverket tränat! ---");

            model.save(new File("chatbot-classifier.zip"), true);
            this.neuralNetworkModel = model; // Lagrar den tränade modellen

        } catch (IOException e) {
            logger.severe("Kritiskt fel vid DL4J-träning: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // FIX 5: Byt ut lokal variabeldeklaration mot att fylla klassfält
    private SentenceIterator loadCorpusAndCreateIterator() throws IOException {
        logger.info("Försöker ladda träningsdata för DL4J...");
        this.sentences = new ArrayList<>(); // Fyller fältet
        this.labelIndexMap = new HashMap<>(); // Fyller fältet
        this.indexToLabelMap = new HashMap<>(); // Initierar mappning för predict
        this.rawIntentions = new ArrayList<>();
        int labelIndex = 0;

        try (InputStream is = getClass().getResourceAsStream(FILE_PATH);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {

            reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                int splitIndex = line.lastIndexOf(',');
                if (splitIndex == -1) continue;

                String text = line.substring(0, splitIndex).trim();

                String intention = line.substring(splitIndex + 1).trim();

                if (!text.isEmpty() && !intention.isEmpty()) {
                    this.sentences.add(text); // Använder fältet
                    this.rawIntentions.add(intention);

                    if (!this.labelIndexMap.containsKey(intention)) {
                        this.labelIndexMap.put(intention, labelIndex);
                        this.indexToLabelMap.put(labelIndex, intention); // Mappar index till label
                        labelIndex++;
                    }
                }
            }
        }
        logger.info("Corpus laddad. Totalt antal fraser: " + this.sentences.size());
        return new CollectionSentenceIterator(this.sentences);
    }

    /**
     * Simulerar Predict: Konverterar text till en vektor med den tränade Word2Vec-modellen.
     */
    public INDArray getVectorForText(String text) {
        if (word2VecModel == null) return Nd4j.zeros(VECTOR_SIZE);

        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();

        INDArray aggregatedVector = Nd4j.zeros(word2VecModel.getLayerSize());
        int count = 0;

        for (String token : tokens) {
            if (word2VecModel.hasWord(token)) {
                aggregatedVector.addi(word2VecModel.getWordVectorMatrix(token));
                count++;
            }
        }

        INDArray finalVector = count > 0 ? aggregatedVector.div(count) : Nd4j.zeros(VECTOR_SIZE);

        return finalVector.reshape(1, finalVector.length());
    }

    public MultiLayerNetwork getNeuralNetworkModel() {
        return neuralNetworkModel;
    }

    // NY METOD: Returnerar index till label map
    public Map<Integer, String> getIndexToLabelMap() {
        return indexToLabelMap;
    }
}