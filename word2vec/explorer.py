## Word2Vec Explorer: per esplorare il database
# Luca Mari, maggio 2023  
# [virtenv `word2vec`: gensim, flask, networkx]
# (è un'applicazione Flask: occorre anche templates/graph.html, che usa d3.js)

import gensim.downloader as api
from flask import Flask, request, render_template, jsonify
import networkx as nx

print('Si sta caricando word2vec...')
wv = api.load('word2vec-google-news-300')

app = Flask(__name__)
G = nx.DiGraph()

@app.route('/')
def index():
    global G
    G.clear()
    chiave = request.args.get('chiave')
    if not chiave: chiave = 'thing'
    try:
        _ = wv[chiave] # type: ignore
        codice = 0
    except:
        chiave = 'thing'
        codice = -1
    numero = request.args.get('numero')
    if not numero: numero = 10
    lista = wv.most_similar(positive=[chiave], topn=int(numero)) # type: ignore
    for i in range(len(lista)):
        G.add_node(lista[i][0], sim=lista[i][1])
        G.add_edge(chiave, lista[i][0])
    return render_template('graph.html', chiave=chiave, numero=numero, codice=codice)

@app.route('/data')
def data():
    data = nx.node_link_data(G) # type: ignore
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
