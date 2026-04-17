package com.example.demo.ml;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Konverterar textfraserna (Word2Vec-vektorer) till det numeriska format
 * (DataSetIterator) som MultiLayerNetwork (Neurala Nätverket) kräver.
 */
public class DataSetIteratorFactory {

    private static final Logger logger = Logger.getLogger(DataSetIteratorFactory.class.getName());

    public static DataSetIterator create(
            List<String> sentences,
            List<String> rawIntentions,
            Map<String, Integer> labelIndexMap,
            Word2Vec word2VecModel,
            int vectorSize,
            TokenizerFactory tokenizerFactory) {

        int numLabels = labelIndexMap.size();
        List<DataSet> dataSets = new ArrayList<>();

        for (int i = 0; i < sentences.size(); i++) {
            String text = sentences.get(i);
            String intention = rawIntentions.get(i);

            // 1. Feature Vector (Input): Få den aggregerade vektorn för meningen
            INDArray features = getAggregatedVector(text, word2VecModel, vectorSize, tokenizerFactory);

            features = features.reshape(1, vectorSize);

            // 2. Label Vector (Output): Skapa den numeriska utgångsetiketten (One-Hot Encoding)
            INDArray labels = Nd4j.zeros(1, numLabels);
            if (labelIndexMap.containsKey(intention)) {
                int index = labelIndexMap.get(intention);
                labels.putScalar(new int[]{0, index}, 1.0);
            }

            // 3. Skapa DL4J DataSet och lägg till i listan
            dataSets.add(new DataSet(features, labels));
        }

        // 4. Returnera som iterator
        return new ListDataSetIterator(dataSets, 32);
    }

    // RIKTIG Vektor Aggregering (Flyttad från MLProcessorDL4J:s predict-metod)
    private static INDArray getAggregatedVector(String text, Word2Vec word2VecModel, int vectorSize, TokenizerFactory tokenizerFactory) {
        if (word2VecModel == null) return Nd4j.zeros(vectorSize);

        // Använd den faktiska tokenizern som tränades
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

        // Returnera medelvärdet
        return count > 0 ? aggregatedVector.div(count) : Nd4j.zeros(vectorSize);
    }
}