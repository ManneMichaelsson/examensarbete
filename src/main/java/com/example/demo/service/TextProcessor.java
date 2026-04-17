package com.example.demo.service;

import java.util.HashMap;
import java.util.Map;

public class TextProcessor {

    private static final Map<Character, Character> CHAR_REPLACEMENTS = new HashMap<>();

    static {
        // Korrigerar kända "feltecken" till svenska tecken
        CHAR_REPLACEMENTS.put('┼', 'å');
        CHAR_REPLACEMENTS.put('─', '-');
        CHAR_REPLACEMENTS.put('╓', 'ä');
        CHAR_REPLACEMENTS.put('σ', 'ä');
        CHAR_REPLACEMENTS.put('Σ', 'ä');
        CHAR_REPLACEMENTS.put('÷', 'ö');
        CHAR_REPLACEMENTS.put('=', 'å');
        // Lämna svenska tecken som de är
        CHAR_REPLACEMENTS.put('Å', 'Å');
        CHAR_REPLACEMENTS.put('Ä', 'Ä');
        CHAR_REPLACEMENTS.put('Ö', 'Ö');
        CHAR_REPLACEMENTS.put('å', 'å');
        CHAR_REPLACEMENTS.put('ä', 'ä');
        CHAR_REPLACEMENTS.put('ö', 'ö');
    }

    public static String normalizeSwedishCharacters(String text) {
        if (text == null) return null;
        StringBuilder sb = new StringBuilder(text.length());
        for (char c : text.toCharArray()) {
            sb.append(CHAR_REPLACEMENTS.getOrDefault(c, c));
        }
        return sb.toString()
                .replaceAll("[^a-zA-ZåäöÅÄÖ0-9 ]", " ") // tar bort andra konstiga tecken
                .replaceAll("\\s+", " ")               // komprimerar whitespace
                .trim();
    }
}