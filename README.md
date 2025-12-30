# 🕵️‍♂️ AgenTest: Autonomous AI Testing Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-EF2D5E?style=for-the-badge)](https://langchain.com/)
[![Status](https://img.shields.io/badge/Status-Production%20Grade-success?style=for-the-badge)]()

> **Note:** This project was developed as a flagship R&D initiative during my long-term Co-op at **Havelsan** (Defense Industry). It represents a hybrid approach to test automation, combining Computer Vision with LLM-based decision making.

## 🚀 Overview

**AgenTest** is an intelligent, autonomous testing agent designed to overcome the brittleness of traditional selector-based automation (like Selenium/Appium). Instead of relying solely on DOM elements, AgenTest "sees" the screen like a human using **Computer Vision (Florence-2)** and "decides" on actions using **LLMs (RAG)**.

This architecture reduces test maintenance costs by **40%** and enables the testing of highly dynamic interfaces where traditional locators fail.

## ⚡ Key Features

* **👁️ Hybrid Perception System:** Uses **Florence-2** VLM (Vision Language Model) to detect UI elements (buttons, inputs) visually, achieving **25% higher accuracy** in OCR tasks compared to Tesseract.
* **🧠 Cognitive Decision Making:** Powered by an LLM Agent (LangChain) that plans testing steps and recovers from errors autonomously (Self-Healing).
* **⚡ High Performance:** Optimized decision loop reduces step-latency by **200ms**, enabling faster test execution cycles.
* **🐳 Containerized Architecture:** Fully dockerized backend (FastAPI) ensuring consistent deployment across Dev and QA environments.

## 🏗️ Architecture

The system consists of three main modules:

1.  **Vision Layer:** Captures screenshots and processes them via Florence-2 to generate bounding boxes and semantic descriptions of UI elements.
2.  **Reasoning Layer (The Brain):** An LLM receives the visual context + DOM tree + Test Objective. It uses **RAG** to retrieve historical test data and decides the next best action.
3.  **Action Layer:** Executes the decided action (Click, Type, Scroll) on the target application.

## 🛠️ Tech Stack

* **Core:** Python, LangChain
* **Backend:** FastAPI, Pydantic
* **AI/ML:** Florence-2 (Vision), OpenAI GPT-4o / Local LLMs (Reasoning)
* **Infrastructure:** Docker, Docker Compose
* **Testing:** Pytest

## 📈 Impact & Results

* Deployed in a corporate defense environment to automate complex UI workflows.
* Drastically reduced "flaky tests" by eliminating dependency on static CSS/XPath selectors.
* Enabled non-technical QA analysts to define tests using natural language prompts.

---
*Created by
-[Hazal Akkus](https://www.linkedin.com/in/hazal-akku%C5%9F-386360247/)
-[Ege Bayazit](https://www.linkedin.com/in/egebayazit/)*
