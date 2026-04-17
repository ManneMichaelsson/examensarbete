// Denna kod är densamma som för Tribuo, eftersom API-gränssnittet inte ändras.
package com.example.demo.controller;

import com.example.demo.service.IntentService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

record ChatRequest(String message) {}
record ChatResponse(String response) {}

@RestController
@RequestMapping("/api/chat")
public class ChatbotController {

    private final IntentService intentService;

    public ChatbotController(IntentService intentService) {
        this.intentService = intentService;
    }

    @PostMapping
    public ChatResponse handleMessage(@RequestBody ChatRequest request) {
        String botResponse = intentService.generateResponse(request.message());
        return new ChatResponse(botResponse);
    }
}