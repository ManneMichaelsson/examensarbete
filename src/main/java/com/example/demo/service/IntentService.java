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
            case "Soka_Djurbestand" -> String.format(
                    "Vilken typ av djur är du intresserad av? Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            case "Adoptionsprocess" -> String.format(
                    "Adoptionsprocessen innebär 3 steg: intresseanmälan, hembesök och avtal. Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            case "Auktion_Info" -> String.format(
                    "Auktionen stänger kl 19:00 ikväll. Lycka till! Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            case "Halso_Garantier" -> String.format(
                    "Alla våra djur är veterinärbesiktigade och har en 3-årig dolda fel-försäkring. Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            case "Kontakta_Agent" -> String.format(
                    "Jag kopplar dig direkt till en handläggare. Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            case "Halsning" -> String.format(
                    "Hej och välkommen till Amandas Shop! Vad kan jag hjälpa dig med idag? Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            case "Foder_Tillbehor" -> String.format(
                    "Vi rekommenderar X-märke för valpar. Jag är %.2f säker på att jag svarade rätt.", confidence
            );
            default -> String.format(
                    "Jag är ledsen, jag förstod inte frågan. Vill du bli kopplad till en mänsklig agent? (predicted intent: %s, confidence: %.2f)",
                    intent, confidence
            );
        };
    }
}