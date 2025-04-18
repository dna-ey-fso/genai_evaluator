<p align="center">
  <img src="Image.png" alt="EY GenAI Accelerator" style="height:200px; width: 500px ;"/>
</p>

# EY GenAI Evaluation Accelerator

Welcome to the **GenAI Evaluation Accelerator** – a modular framework developed by the **EY FSO Belgium Data & AI team** to evaluate core functionalities of generative AI applications powered by LLMs.

This accelerator enables fast and consistent testing of tasks like:
- ✅ Zero-shot / Few-shot Classification  
- ✅ Intent Detection  
- ✅ Retrieval-Augmented Generation (RAG)

It provides plug-and-play evaluation flows for both **retrieval** and **generation**, with support for **multiple LLM clients** and **custom metric definition using Jinja templates**.

---

## 🚀 Key Features

- ⚙️ **Modular Evaluation Flows** for generation and retrieval tasks  
- 📊 **Built-in LLM-Based and Heuristic Metrics**: Faithfulness, Precision, Recall, Relevancy  
- 🤖 **Multi-provider LLM Support**: Azure OpenAI, Mistral  
- 🧩 **Extensible Tooling** via Pydantic-based function calling  
- 🧠 **Prompt Templating** using Jinja to standardize evaluation logic  
- 🧪 **Notebook Examples** for quick experimentation  

---

## 🗂️ Repository Structure

```bash
.
├── clients/          # Standardized LLM clients (OpenAI, Mistral, etc.)
├── flows/            # Evaluation flows (retrieval and generation)
├── interfaces/       # Common dataclasses and interfaces
├── metrics/          # Evaluation metrics (faithfulness, precision, recall, relevancy)
├── templates/        # Jinja templates for structuring prompts and metric logic
├── data/             # Sample input datasets for evaluation tasks
├── evalaution_test.ipynb     # Example notebooks demonstrating evaluation setup
├── environment.yaml     #conda env 
└── README.md
```

## 📚 Evaluation Flows

#### 🧠 Generation Evaluation Flow
Evaluates LLM-generated outputs using:

- Faithfulness: Checks if the answer aligns with the context

- Precision & Recall: Measures informativeness and coverage against ground truth

#### 🔎 Retrieval Evaluation Flow
Evaluates retriever quality using:

- Relevancy: Assesses how relevant retrieved documents are to the query and answer

All flows are built with modularity and extensibility in mind and live inside the flows/ folder.

## 🤖 LLM Client Abstraction
Located in clients/, these classes provide a unified interface to interact with different LLM backends.

Supported Clients:
* Azure OpenAI (supports structured output and function calls)
* Mistral via Azure Serverless (limited structured output support)

They support:
   * Prompt sending
   * Tool calling / function execution
   * Structured response parsing (via Pydantic)
   * Plug-and-play usage across evaluation flows

## 🧪 Metrics
Implemented in the metrics/ folder as modular components.

Available Metrics:
* FaithfulnessMetric
* PrecisionMetric
* RecallMetric
* RelevancyMetric

All LLM-based metrics leverage prompt templates defined in the templates/ folder to structure the LLM call and parse responses.

## 📝 Prompt Templates
All LLM-based metric logic is defined using Jinja templates in the templates/ folder, enabling clear, maintainable evaluation prompts.

Templates define:
* The format of the LLM input
* Expected output parsing format
* Instructional framing for zero/few-shot tasks

## 🧱 Requirements
* Python 3.10+
* Azure OpenAI or Mistral access
* Core libraries: pydantic, jinja2, etc.


```bash

# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create the environment from the YAML file
conda env create -f environment.yaml

# Activate the environment
conda activate your-env-name

```

## 🚀 Test it Yourself

To quickly try out the evaluation flows in this accelerator, you can use the example notebook:

📍 **Notebook**: [`evaluation_test.ipynb`](notebooks/evaluation_test.ipynb)

This notebook demonstrates how to:

- Plug in your own test samples with the required structure.
- Run generation or retrieval evaluation flows.
- Use LLM-based and standard metrics to assess performance.

### 🔐 Required Credentials

Before running the notebook, make sure to set the following environment variables (or define them directly in your notebook):

```bash
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
```

## 🛠️ Extending the Framework

#### ➕ Add a New Metric
* Implement your metric class in metrics/
* Optionally define a Jinja template in templates/
* Plug it into the evaluation flow in flows/

#### 🔌 Add a New LLM Client
* Subclass LLMClient in clients/
* Implement send_prompt() and parsing logic

## 👥 About Us
This accelerator is built and maintained by the Data & AI team at EY FSO Belgium to support internal and client-facing use cases where rigorous evaluation of generative AI systems is essential.

## Contacts
 Othmane Belmoukadam (Head of AI Lab): 
 Othmane.Belmouakdam@be.ey.com