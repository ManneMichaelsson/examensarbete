# examensarbete

🛠 How to Run the Project

Follow these steps to get the AI Chatbot up and running on your local machine.
1. Prerequisites

    Java 17 (or higher) installed.

    Maven installed.

    Postman (for testing the API).

2. Setup and Training

    Open the project in your IDE (IntelliJ IDEA is recommended).

    Ensure the training data file src/main/resources/corpus.csv is present.

    Run the application via your IDE or use the terminal:
    Bash

    mvn spring-boot:run

    Wait for training: During startup, the Deep Learning model will train automatically. Look for Score at iteration X in the console. The server is ready when you see:
    Started AmandasWebshopApplication in X seconds.

3. Testing with Postman

Since this is a REST API, you should use Postman to interact with the chatbot.

    Open Postman and create a new POST request.

    URL: http://localhost:8080/api/chat (or your specific endpoint).

    Headers: * Set Content-Type to application/json.

    Body:

        Select raw and JSON.

        Enter a test message, for example:
        JSON

        {
          "message": "I need a new rain cover for my horse"
        }

    Send: Click "Send" and the bot will return the classified intent and a professional reply.

💡 Troubleshooting

    First request is slow? The very first classification might take a second as the model loads into memory.

    Wrong language? Ensure your message in Postman is in English to match the trained dataset.
