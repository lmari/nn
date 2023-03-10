# nn

Questo repository propone esempi di reti neurali, _a solo scopo didattico_:
dunque con più interesse per la comprensibilità che per l'efficienza, la completezza, ecc.

La cartella `perceptron` contiene un notebook Python con l'implementazione
di un single-layer perceptron, dunque plausibilmente la più semplice rete neurale
(così semplice da essere costituita da un solo neurone...).

La cartella `mperceptron` contiene un notebook Python con l'implementazione
di un multi-layer perceptron, con un solo layer nascosto, configurato per classificare
le immagini dei caratteri numerici dal dataset mnist. 

La cartella `tokenizer` contiene un notebook Python con un semplice esempio
di tokenizzazione con `tiktoken` di OpenAI.

La cartella `chatgpt0` contiene un notebook Python con il più semplice esempio
di uso dell'API di ChatGPT.


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
