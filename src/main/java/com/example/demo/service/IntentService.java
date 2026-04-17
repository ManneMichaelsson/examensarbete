package com.example.demo.service;

import com.example.demo.ml.MLProcessorDL4J;
import org.springframework.stereotype.Service;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

@Service
public class IntentService {

    private final MLProcessorDL4J mlProcessor;

    public IntentService(MLProcessorDL4J mlProcessor) {
        this.mlProcessor = mlProcessor;
    }

    public String generateResponse(String userText) {

        userText = TextProcessor.normalizeSwedishCharacters(userText);

        userText = userText.replaceAll("[^a-zA-ZåäöÅÄÖ0-9 ]", " ").replaceAll("\\s+", " ").trim();

        // 1. Få vektorn (features) från texten
        INDArray features = mlProcessor.getVectorForText(userText);

        // 2. KLASSIFICERA VECTORN med det tränade neurala nätverket
        INDArray output = mlProcessor.getNeuralNetworkModel().output(features);

        // 3. Hämta alla sannolikheter som map<Intent,Confidence>
        Map<Integer, String> indexToLabel = mlProcessor.getIndexToLabelMap();

        System.out.println("=== Klassificering av: \"" + userText + "\" ===");
        for (int i = 0; i < output.columns(); i++) {
            String intentName = indexToLabel.get(i);
            double intentConfidence = output.getDouble(i);
            System.out.printf("%s: %.4f%n", intentName, intentConfidence);
        }

        // 4. Hämta indexet för den högsta sannolikheten
        int predictedIndex = output.argMax(1).getInt(0);
        double confidence = output.getDouble(predictedIndex);
        String intent = mlProcessor.getIndexToLabelMap().get(predictedIndex);

        boolean usedDefault = confidence < 0.3 || intent == null;

        // 5. Returnera svaret baserat på intent, men logga om default används
        if (usedDefault) {
            return String.format(
                    "Jag är inte säker på vad du menar (predicted intent: %s, confidence: %.2f). Vill du bli kopplad till en människa?",
                    intent, confidence
            );
        }

        return switch (intent) {
            case "Soka_Djurbestand" ->
                    "What type of animal are you interested in? You can find our current stock in the online catalog.";

            case "Adoptionsprocess" ->
                    "The adoption process involves three steps: interest form, a home visit, and the final agreement.";

            case "Auktion_Info" ->
                    "The auction closes at 7:00 PM tonight. Good luck with your bidding!";

            case "Halso_Garantier" ->
                    "All our animals are vet-checked and come with a 3-year hidden defect insurance.";

            case "Kontakta_Agent" ->
                    "I am connecting you directly to a staff member. Please wait a moment.";

            case "Halsning" ->
                    "Hello and welcome to Amanda's Shop! How can I help you today?";

            case "Foder_Tillbehor" ->
                    "We recommend our premium brands for puppies and high-performance feed for horses.";

            case "Out_of_Scope" ->
                    "I'm sorry, I only answer questions related to our pet shop and animals.";

            default -> String.format(
                    "I'm sorry, I didn't quite understand that. Would you like to be connected to a human agent? (Intent: %s)",
                    intent
            );
        };
    }
}