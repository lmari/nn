# nn

Questo repository propone esempi di reti neurali, _a solo scopo didattico_:
dunque con più interesse per la comprensibilità che per l'efficienza, la completezza, ecc.

La cartella `perceptron` contiene un notebook Python con l'implementazione
di un single-layer perceptron, dunque plausibilmente la più semplice rete neurale
(così semplice da essere costituita da un solo neurone...)

---
I notebook Python possono essere scaricati, per eseguirli localmente: una volta visualizzato il notebook a cui si è interessati, è sufficiente fare click con il tasto di destra sul pulsante "Raw" e scegliere "Save link as..."

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
