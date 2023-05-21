# nn

Questo repository propone esempi di reti neurali, _a solo scopo didattico_: dunque con più interesse per la comprensibilità che per l'efficienza, la completezza, ecc.

Il repository è organizzato in cartelle, ognuna relativa a un argomento di sperimentazione, realizzato attraverso uno o più notebook Python (o moduli Python, dove l'estensione `.py` è indicata).

* `perceptron`: implementazione di un single-layer perceptron, dunque plausibilmente la più semplice rete neurale (così semplice da essere costituita da un solo neurone...)

* `mperceptron`: implementazione di un multi-layer perceptron, con un solo layer nascosto, configurato per classificare le immagini dei caratteri numerici dal dataset `mnist` (rispetto a un'implementazione standard, qui si prova a tener conto anche dell'incertezza di classificazione)

* `poly`: esempi di approssimazione di funzioni reali
    * `poly0`: l'implementazione in `numpy` di una semplice rete neurale -- con un neurone di input, uno strato nascosto, e un neurone di output -- per approssimare funzioni $\mathbb{R} \rightarrow \mathbb{R}$
    * `poly1`: l'implementazione in `numpy` di un solutore che approssima funzioni $\mathbb{R} \rightarrow \mathbb{R}$ mediante un polinomio di grado $n$, di cui stima i parametri
   * `poly1b`: l'implementazione in `pytorch` di un solutore che approssima funzioni $\mathbb{R} \rightarrow \mathbb{R}$ mediante un polinomio di grado $n$, di cui stima i parametri

* `word2vec`: esempi di accesso a un database di word embeddings:
    * `word2vec`: esempi vari
    * `explorer.py`: un'applicazione Flask per esplorare il database in modo interattivo

* `chatgpt0`: esempi di uso dell'API di ChatGPT
    * `dialogo0`: il più semplice esempio di uso dell'API di ChatGPT
    * `dialogo1`: un semplice esempio di uso dell'API di ChatGPT in modalità _streaming_/sequenziale
    * `dialogo2`: un semplice esempio di uso dell'API di ChatGPT in modalità _speech-to-text_ e poi _text-to-speech_

* `langchain`: esempi di uso di `langchain`
    * `simpleQA`: il più semplice esempio di Q&A sul contenuto di un documento
    * `complexQA`: un esempio di Q&A sul contenuto di un documento, con _embeddings_ e ricerca semantica di un _vector db_
    * `mathQA`: un esempio di matematica
    * `explicitChat.py`: un semplice esempio di chat con gpt
    * `wrappedChat.py`: un semplice esempio di chat con gpt, con streaming della risposta
    * `gpt4all`: il più semplice esempio di Q&A con un modello locale, gestito con `GPT4all`


* `backprop`: un semplice esempio della logica della backpropagation, con `micrograd`

* `tokenizer`: un semplice esempio di tokenizzazione con `tiktoken` di OpenAI

* `testGPU`: esempio di una stessa funzione Python eseguita in CPU e in GPU, grazie al modulo `numba`, mostrando una riduzione dei tempi di esecuzione di almeno un'ordine di grandezza


---
I notebook Python possono essere scaricati, per eseguirli localmente: una volta visualizzato il notebook a cui si è interessati, è sufficiente fare click con il tasto di destra sul pulsante "Raw" e scegliere "Save link as..."

---
Per usare l'API di ChatGPT occorre prima di tutto:
* attivare un account OpenAI (vedi https://platform.openai.com/account)
* attivare il metodo di pagamento
* generare una nuova chiave, copiarla e scriverla (per Linux) per esempio in `.bashrc`:  
    `export OPENAI_API_KEY=...`

---
La procedura per preparare l'esecuzione di un notebook Python (provata per VSCode su una macchina Linux con Anaconda installato; su Windows e MacOS non dovrebbe essere molto diverso):
* creare una directory e cd
* creare un ambiente virtuale `x`:  
   `conda create -n x python`  
(per eliminarlo `conda env remove -n x` )
* attivare l'ambiente virtuale:  
    `conda activate x`
* installare il modulo per la gestione interattiva:  
    `pip install ipykernel`
* installare il modulo per la gestione dei requirements (opzionale: oppure scaricare il file `requirements.txt`):  
    `pip install pipreqs`
* creare il file `requirements.txt` (opzionale: oppure scaricare il file `requirements.txt`):  
    `pipreqs`
* installare i moduli richiesti:  
    `pip install -r requirements.txt`
* elencare gli ambienti virtuali creati:
    `conda env list`

---
virtenv `nn`: numpy, matplotlib, torch, requests, gzip  
virtenv `langchain`: langchain, openai, chroma, chromadb, tiktoken, pygpt4all
