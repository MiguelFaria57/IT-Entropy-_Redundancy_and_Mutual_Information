# BIBLIOTECAS USADAS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as spiowf
import huffmancodec as hc


# EXERCÍCIOS
# ----------------------------------- Ex1 e Ex2
# Função que calcula o número de ocorrências de cada símbolo
# Parâmetros: P - lista de símbolos ; A - alfabeto com todos os símbolos
def numOcorrencias(P, A):
    no = [0] * len(A) # inicialização lista que vai guardar o número de ocorrências
    aux = dict(zip(A, no)) # dicionário - símbolos do alfabeto : número de ocorrências
    for i in P: # aumentar o número de ocorrências dos símbolos da fonte
        aux[i] += 1
    no = list(aux.values()) # lista final com o número de ocorrências atualizado
    return no


# ----------------------------------- Ex1
# Função que faz uma imagem do histograma da ocorrência dos seus símbolos
# Parâmetros: P - fonte de informação ; A - alfabeto com todos os símbolos ; N - nome do ficheiro
def histograma(P, A, N):
    no = numOcorrencias(P, A) # lista com o número de ocorrências
    # criação do histograma
    plt.bar(A, no)
    plt.xlabel('Alfabeto')
    plt.ylabel('Nº de Ocorrências')
    plt.title(N)
    plt.show()


# ----------------------------------- Ex2
# Função que calcula o limite mínimo teórico para o número médio de bits por símbolo
# Parâmetros: P - fonte de informação ; A - alfabeto com todos os símbolos
def entropia(P, A):
    no = numOcorrencias(P, A) # lista com o número de ocorrências
    p = [] # lista onde vai ser guardada a probabilidade de cada símbolo
    for j in range(len(no)): # cálculo da probabilidade
        p += [no[j]/len(P)]
    p = np.array(p)
    p = p[p>0] # retirar casos excecionais
    H = -sum(p * np.log2(p)) # cálculo da entropia
    return H


# ----------------------------------- Ex3
# Função que retira a fonte da imagem
# Parâmetros: I - imagem
def image(I):
    img = mpimg.imread(I) # lista com valores dos píxeis da imagem
    if img.ndim > 2: # retirar apenas o canal R caso tenha mais que dois canais
        img = img[:,:, 0]
    P = img.flatten() # transformar o array para uma dimensão
    A = np.arange(0, 256) # o alfabeto corresponde aos 8 bits de cor do canal
    histograma(P, A, I)
    print("Ficheiro: %s" %I)
    print("Entropia: %.6f" %(entropia(P, A)))
    print("Número médio de bits por símbolo: %.6f" %(huffman(P)))
    print("Variância dos comprimentos dos códigos: %.6f" %(variancia(P)))


# Função que retira a fonte do fichero de som
# Parâmetros: S - ficheiro de som
def sound(S):
    data = spiowf.read(S)[1] # lista com as samples por segundo(fs) e informação(data)
    P = data[:, 0] # ficar apenas com a informação do canal esquerdo
    P = np.array(P)
    numBits = int(str(data.dtype)[4:]) # número de bits a usar no alfabeto
    A = np.arange(0, 2**numBits) # array com o alfabeto
    histograma(P, A, S)
    print("Ficheiro: %s" %S)
    print("Entropia: %.6f" %(entropia(P, A)))
    print("Número médio de bits por símbolo: %.6f" %(huffman(P)))
    print("Variância dos comprimentos dos códigos: %.6f" %(variancia(P)))


# Função que retira a fonte do fichero de texto
# Parâmetros: T - ficheiro de texto
def text(T):
    # leitura do ficheiro de texto
    f = open(T, 'r')
    dados = f.read()
    f.close()
    P = [] # lista da fonte
    for i in range(len(dados)):
        if (65 <= ord(dados[i]) <= 90) or (97 <= ord(dados[i]) <= 122): # condição para restringir a letras minúsculas e
                                                                        # maiúsculas
            P += [dados[i]]
    P = np.array(P)
    A = np.array(list(map(chr, range(65,91))) + list(map(chr, range(97,123)))) # array que contém o alfabeto (letras
                                                                               # minúsculas e maiúsculas)
    histograma(P, A, T)
    print("Ficheiro: %s" %T)
    print("Entropia: %.6f" %(entropia(P, A)))
    print("Número médio de bits por símbolo: %.6f" %(huffman(P)))
    print("Variância dos comprimentos dos códigos: %.6f" % (variancia(P)))


# Função que distribui os diferentes ficheiros
# Parâmetros: ficheiro - nome do ficheiro
def Ex3_4(ficheiro):
    tipo = ficheiro.split('.')[1]
    if tipo == 'bmp':
        image(ficheiro)
    if tipo == 'wav':
        sound(ficheiro)
    if tipo == 'txt':
        text(ficheiro)


# ----------------------------------- Ex4
# Função que determina o número médio de bits por símbolo
# Parâmetros: P - fonte
def huffman(P):
    codec = hc.HuffmanCodec.from_data(P)
    symbols, lenghts = codec.get_code_len()
    mediaBits = 0 # variável onde vai ser armazenado o número médio de bits
    for i in range(len(P)):
        mediaBits += lenghts[symbols.index(P[i])]/len(P)
    return mediaBits


# Função que determina a variância dos comprimentos dos códigos
# Parâmetros: P - fonte
def variancia(P):
    codec = hc.HuffmanCodec.from_data(P)
    symbols, lenghts = codec.get_code_len()
    EX = 0
    for i in range(len(P)):
        EX += lenghts[symbols.index(P[i])]/len(P)
    EX2 = 0
    for i in range(len(P)):
        EX2 += lenghts[symbols.index(P[i])]**2/len(P)
    # arredondamento de casas decimais excessivas
    EX = round(EX, 6)
    EX2 = round(EX2, 6)
    V = EX2 - EX**2
    return V


# ----------------------------------- Ex5
# Função que faz uma imagem do histograma da ocorrência do agrupamento dos seus símbolos
# Parâmetros: P - fonte de informação ; A - alfabeto com todos os símbolos
def histogramaEx5(P, A):
    no = numOcorrencias(P, A)
    B = []
    for i in range(len(A)):
        B += [str(A[i])]
    plt.bar(B, no)
    plt.xlabel('Alfabeto')
    plt.ylabel('Nº de Ocorrências')
    plt.title('Histograma')
    plt.show()


# Função que retira a fonte da imagem
# Parâmetros: I - imagem
def imageEx5(I):
    img = mpimg.imread(I) # lista com valores dos píxeis da imagem
    if img.ndim > 2: # retirar apenas o canal R caso tenha mais que dois canais
        img = img[:,:, 0]
    P = img.flatten() # transformar o array para uma dimensão
    if len(P) % 2 != 0: # remoção do último elemento caso o comprimento da lista seja ímpar
        P=P[:-1]
    PAlterado = P.reshape(int(len(P)/2), 2) # transformação da fonte em sequências de dois símbolos contíguos
    AAlterado = np.unique(PAlterado, axis=0) # excerto do alfabeto total, contendo apenas os elementos existentes também
                                             # na fonte
    # transformação dos arrays em listas de tuplos de modo a poderem ser utilizados no dicionário
    PA = list(map(tuple, PAlterado))
    AA = list(map(tuple, AAlterado))
    #histogramaEx5(PA, AA)
    print("Ficheiro: %s" %I)
    print("Entropia: %.6f" % (entropia(PA, AA) / 2)) # divisão da entropia por dois porque são pares


# Função que retira a fonte do fichero de som
# Parâmetros: S - ficheiro de som
def soundEx5(S):
    data = spiowf.read(S)[1] # lista com as samples por segundo(fs) e informação(data)
    P = data[:, 0] # ficar apenas com a informação do canal esquerdo
    if len(P) % 2 != 0: # remoção do último elemento caso o comprimento da lista seja ímpar
        P=P[:-1]
    PAlterado = P.reshape(int(len(P)/2), 2) # transformação da fonte em sequências de dois símbolos contíguos
    AAlterado = np.unique(PAlterado, axis=0) # excerto do alfabeto total, contendo apenas os elementos existentes também
                                             # na fonte
    # transformação dos arrays em listas de tuplos de modo a poderem ser utilizados no dicionário
    PA = list(map(tuple, PAlterado))
    AA = list(map(tuple, AAlterado))
    #histogramaEx5(PA, AA)
    print("Ficheiro: %s" %S)
    print("Entropia: %.6f" % (entropia(PA, AA) / 2)) # divisão da entropia por dois porque são pares


# Função que retira a fonte do fichero de texto
# Parâmetros: T - ficheiro de texto
def textEx5(T):
    # leitura do ficheiro de texto
    f = open(T, 'r')
    dados = f.read()
    f.close()
    P = []
    for i in range(len(dados)):
        if (65 <= ord(dados[i]) <= 90) or (97 <= ord(dados[i]) <= 122): # condição para restringir a letras minúsculas e
                                                                        # maiúsculas
            P += [dados[i]]
    P = np.array(P)
    if len(P) % 2 != 0: # remoção do último elemento caso o comprimento da lista seja ímpar
        P = P[:-1]
    PAlterado = P.reshape(int(len(P)/2), 2) # transformação da fonte em sequências de dois símbolos contíguos
    AAlterado = np.unique(PAlterado, axis=0) # excerto do alfabeto total, contendo apenas os elementos existentes também
                                             # na fonte
    # transformação dos arrays em listas de tuplos de modo a poderem ser utilizados no dicionário
    PA = list(map(tuple, PAlterado))
    AA = list(map(tuple, AAlterado))
    #histogramaEx5(PA, AA)
    print("Ficheiro: %s" %T)
    print("Entropia: %.6f" % (entropia(PA, AA) / 2)) # divisão da entropia por dois porque são pares


# Função que distribui os diferentes ficheiros
# Parâmetros: ficheiro - nome do ficheiro
def Ex5(ficheiro):
    tipo = ficheiro.split('.')[1]
    if tipo == 'bmp':
        imageEx5(ficheiro)
    if tipo == 'wav':
        soundEx5(ficheiro)
    if tipo == 'txt':
        textEx5(ficheiro)


# ----------------------------------- Ex6
# Função que calcula a entropia conjunta
# Parâmetros: Q - query ; T - target ; A - alfabeto
def entropiaConjunta(Q, T, A):
    a2=np.zeros((len(A), len(A))) # matriz com a dimensão do comprimento do alfabeto, com elementos inicializados a 0,
                                  # onde se vai incrementando o número de ocorrências de cada par
    for i in range(len(Q)): # procura da posição de cada par na matriz para incrementar
        indQ = np.where(A == Q[i])
        indT = np.where(A == T[i])
        a2[indQ, indT] += 1
    a2 = a2/len(Q) # cálculo da probabilidade
    a2 = a2.flatten() # transformar o array para uma dimensão
    a2 = a2[a2>0] # tirar casos excecionais
    H = -sum(a2*np.log2(a2)) # cálculo da entropia
    return H


# Função que calcula a informação mútua entre a query e o target
# Parâmetros: Q - query ; T - target ; A - alfabeto ; passo - intervalo entre janelas
def informacaoMutua(Q, T, A, passo):
    informacao = [] # lista onde vão ser guardados os valores da informação mútua para cada janela
    for i in range(0, len(T)-len(Q)+1, passo): # ciclo que permite percorrer o target com o passo
        janela = np.copy(T[i:i+len(Q)]) # array com a janela
        # cálculo das entropias
        HX = entropia(Q, A)
        HY = entropia(janela, A)
        HXY = entropiaConjunta(Q, janela, A)
        # cálculo e adição da informação mútua
        inf = HX + HY - HXY
        informacao += [inf]
    return informacao


# Função que faz uma imagem do gráfico da evolução da informação mútua ao longo do tempo para cada target, de dois
# ficheiros de som
# Parâmetros: informacao1 - lista com os valores da informação mútua de um ficheiro de som ; informacao1 - lista com os
# valores da informação mútua de outro ficheiro de som
def graficoEvolucaoEx6b(informacao1, informacao2):
    # criação do gráfico
    plt.plot(informacao1)
    plt.plot(informacao2)
    plt.xlabel('Janelas')
    plt.ylabel('Informação Mútua')
    plt.title('Evolução da Informação Mútua')
    plt.show()


# Função que faz uma imagem do gráfico da evolução da informação mútua ao longo do tempo para cada target
# Parâmetros: informacao - lista com os valores da informação mútua de todas as janelas ; S - nome do ficheiro de som
def graficoEvolucaoIM(informacao, S):
    # criação do gráfico
    plt.plot(informacao)
    plt.xlabel('Janelas')
    plt.ylabel('Informação Mútua')
    plt.title('%s' %S)
    plt.show()


# Função que retira a informação dos ficheiros de som e calcula a informação mútua
# Parâmetros: S1 - nome do ficheiro de som de target ; S2 - nome do ficheiro de som de query ; passo - intervalo entre
# janelas
def Ex6(St, Sq, passo):
    data1 = spiowf.read(St)[1] # lista com as samples por segundo(fs) e informação(data1) do target
    if data1.ndim > 1:
        T = data1[:, 0] # ficar apenas com a informação do canal esquerdo
    else:
        T = data1
    data2 = spiowf.read(Sq)[1] # lista com as samples por segundo(fs) e informação(data2) da query
    Q = data2[:, 0] # ficar apenas com a informação do canal esquerdo
    numBits = int(str(data2.dtype)[4:])  # número de bits a usar no alfabeto
    A = np.arange(0, 2**numBits) # array com o alfabeto
    passo = int(passo*len(Q)) # cálculo do passo consoante o comprimento da query
    infM = informacaoMutua(Q, T, A, passo) # cálculo da informação mútua
    infM = np.array(infM)
    infM = np.around(infM, decimals=6)  # arredondamento dos valores da informação mútua
    print("Evolução da informação mútua (%s): " %St + str(infM))
    return infM


# Função que calcula a variação da informação mútua para dois targets diferente e as compara
# Parâmetros: St1 - nome do ficheiro de som de um target ; St2 - nome do ficheiro de som de outro target ; Sq - nome do
# ficheiro de som da query
def Ex6b(St1, St2, Sq):
    infM1 = Ex6(St1, Sq, 0.25) # cálculo da informação mútua de um ficheiro de som
    infM2 = Ex6(St2, Sq, 0.25) # cálculo da informação mútua de outro ficheiro de som
    graficoEvolucaoEx6b(infM1, infM2)  # visualizar a evolução da informação mútua ao longo do tempo em ambos


# Função que calcula a variação da informação mútua e a variação mútua máxima
# Parâmetros: St - nome do ficheiro de som do target ; Sq - nome do ficheiro de som da query
def Ex6c(St, Sq):
    infM = Ex6(St, Sq, 0.25) # cálculo da informação mútua
    infM = np.array(infM)
    infM = np.around(infM, decimals=6)  # arredondamento dos valores da informação mútua
    IMmax = np.amax(infM) # descobre a informação mútua máxima
    print("Informação mútua máxima: " + str(IMmax))
    graficoEvolucaoIM(infM, St) # visualizar a evolução da informação mútua ao longo do tempo



# ----------------------------------- MAIN
def main():
    plt.close('all')

    ficheirosEx3_4_5 = ["binaria.bmp", "ct1.bmp", "lena.bmp", "saxriff.wav", "texto.txt"]
    ficheirosEx6c = ["Song01.wav", "Song02.wav", "Song03.wav", "Song04.wav", "Song05.wav", "Song06.wav", "Song07.wav"]


    for i in ficheirosEx3_4_5:
        Ex3_4(i)
        print()


    for i in ficheirosEx3_4_5:
        Ex5(i)
        print()

    print(informacaoMutua(np.array([2,6,4,10,5,9,5,8,0,8]), np.array([6, 8, 9, 7, 2, 4, 9, 9, 4, 9, 1, 4, 8, 0, 1, 2, 2, 6, 3, 2, 0, 7, 4, 9, 5, 4, 8, 5, 2, 7, 8, 0, 7, 4, 8, 5, 7, 4, 3, 2, 2, 7, 3, 5, 2, 7, 4, 9, 9, 6]), np.arange(0, 11), 1))
    print()

    Ex6b("target01 - repeat.wav", "target02 - repeatNoise.wav", "saxriff.wav")
    print()

    for i in ficheirosEx6c:
        Ex6c(i, "saxriff.wav")
        print()


if __name__ == "__main__":
    main()