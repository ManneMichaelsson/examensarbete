plugins {
    java
    id("org.springframework.boot") version "3.5.7"
    id("io.spring.dependency-management") version "1.1.7"
}

group = "com.example"
version = "0.0.1-SNAPSHOT"
description = "AI-driven Kundservicebot för intentionklassificering."

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-M2.1")

    // Matrisbibliotek (ND4J) - Kritiskt för alla DL4J-beräkningar
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")

    // Natural Language Processing (Behövs för Tokenisering/Vektorisering)
    implementation("org.deeplearning4j:deeplearning4j-nlp:1.0.0-M2.1")
}

tasks.withType<Test> {
    useJUnitPlatform()
}
