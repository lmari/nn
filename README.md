# nn

Questo repository propone esempi di reti neurali, _a solo scopo didattico_: dunque con più interesse per la comprensibilità che per l'efficienza, la completezza, ecc.

Il repository è organizzato in cartelle, ognuna relativa a un argomento di sperimentazione, realizzato attraverso uno o più notebook Python.

* `perceptron`: implementazione di un single-layer perceptron, dunque plausibilmente la più semplice rete neurale (così semplice da essere costituita da un solo neurone...) (`numpy`)

* `mperceptron`: implementazione di un multi-layer perceptron, con un solo layer nascosto, configurato per classificare le immagini dei caratteri numerici dal dataset `mnist` (rispetto a un'implementazione standard, qui si prova a tener conto anche dell'incertezza di classificazione) (`numpy`)

* `poly`: esempi di approssimazione di funzioni reali  (`numpy`, `pytorch`)

* `backprop`: un semplice esempio della logica della backpropagation, con `micrograd`

* `tokenizer`: un semplice esempio di tokenizzazione con `tiktoken` di OpenAI

* `chatgpt0`: esempi di uso dell'API di ChatGPT

* `langchain`: esempi di uso di `langchain`

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

---
L'ambiente virtuale `nn` contiene:
* micrograd

ed è per i notebook nelle cartelle:
* backprop
