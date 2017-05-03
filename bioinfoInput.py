import numpy as np
import pandas as pd
import rnnInput as inp
import tensorflow as tf
import time as t
from tensorflow.contrib import learn
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com


#length=2271
#maxSec=300
#TIMESTEPS=40


def poissonDiv(size=10):
    #gera uma distribuicao poison e devolve o vetor e o fator lambda que a gerou
    #faixa 60-120 para compatibilidade com DNAcopy
    val1=np.random.uniform(60,120)
    val2=np.random.uniform(60,120)
    a= np.random.poisson(lam=val1, size=size)
    b= np.random.poisson(lam=val2, size=size)
    lambRate=(val1/val2)
    c=np.log2(a/b)
    return (a/120,b/120,lambRate)


def makeCNV(tamVec,maxSector=80,minSector=30):
    #o processo de criacao do cnv ocorre da seguinte maneira
    #sorteio um tamanho para o setor, gero uma matriz com a resposta da funcao poisson
    #normalizo a matriz e preencho um vetor com o valor e a resposta (lambda/lambda)
    #repito o mesmo processo ate gerar um vetor de tamVec pra retornar
    
    #gera o vetor e cria o vetor resposta
    arr = np.ndarray(tamVec, np.float32)
    tY = np.ndarray(tamVec, np.float32)
    tY[:]=0
    #primeira iteracao
    tamSec=np.random.randint(minSector,high=maxSector)
    rawmatriz0,rawmatrizB,yy = poissonDiv(size=tamSec)
    matriz0=np.ma.compressed(np.ma.fix_invalid(rawmatriz0))
    matrizb=np.ma.compressed(np.ma.fix_invalid(rawmatrizB))
    #Nmatriz0=matriz0/matriz0.max()
    Nmatriz0=matriz0.compress(matriz0!=0)
    NmatrizB=matrizb.compress(matrizb!=0)
    #testeY=0
    
    #yy=testeY
    #preenche o vetor y
    for i in range (Nmatriz0.size): arr[i] = yy
    out = Nmatriz0
    outB = NmatrizB
    tempI=0
    
    #enquanto o vetor nao estiver todo preenchido, ele continua rodando
    if (Nmatriz0.size < tamVec):
        sizeZ=Nmatriz0.size
        while(sizeZ<tamVec):
            
            sizeZ+=tempI
            #testeY=sizeZ
            tamSec=np.random.randint(minSector,high=maxSector)
            rawMatrizT,rawMatrizB,yt = poissonDiv(size=tamSec)
            #yt = testeY
            if (sizeZ<tamVec):
                tY[sizeZ]=2**yt
            matrizT=np.ma.compressed(np.ma.fix_invalid(rawMatrizT))
            matrizB=np.ma.compressed(np.ma.fix_invalid(rawMatrizB))
            #NmatrizT=matrizT/matrizT.max()
            NmatrizT=matrizT.compress(matrizT!=0)
            NmatrizB=matrizB.compress(matrizB!=0)
            for i in range (sizeZ,NmatrizT.size+sizeZ):
                if (i<tamVec):
                    arr[i] = yt
                 
            tempI=NmatrizT.size
            out = np.append(out,NmatrizT)
            outB = np.append(outB,NmatrizB)
        out = out[:tamVec]
        outB = outB[:tamVec]
        out=pd.DataFrame(out)
        outB=pd.DataFrame(outB)
        arr=pd.DataFrame(arr)
    #devolve o vetor escrito e o vetor gerado
    return (out,outB,arr,tY)

def getDataRND(maxLength,maxSector):
    #funcao que gera o cnv (rnd)
    #devolve o cnv e as "leituras"
    arrY=np.ndarray(shape=(maxLength))
    arrX=np.ndarray(shape=(maxLength,1))
    posAtual=0;
    setorM=np.random.randint(0,high=maxSector)
    valorSetor=np.random.uniform(0.1,0.9)
    randomFaixa = 0.18
    for i in range(maxLength):
        arrY[i] = valorSetor
        arrX[i] = np.random.uniform(valorSetor-randomFaixa,valorSetor+randomFaixa)
        if (i>setorM):
            setorM=np.random.randint(setorM,high=setorM+maxSector)
            valorSetor=np.random.uniform(0.1,0.9)
    arrY=pd.DataFrame(arrY)
    arrX=pd.DataFrame(arrX)  
    return arrX,arrY

def getDataInverse(maxLength,timeSteps,maxSector):
    #devolve o inverso da funcao anterior
    #devolve leituras e na f(x) = cnv
    arrY=np.ndarray(shape=(maxLength))
    arrX=np.ndarray(shape=(maxLength,1))
    posAtual=0;
    randomFaixa = 0.08
    setorM=np.random.randint(0,high=maxSector)
    valorSetor=np.random.uniform(0.1,0.9)
    for i in range(maxLength):
        arrX[i] = valorSetor
        arrY[i] = np.random.uniform(np.absolute(valorSetor)-randomFaixa,np.absolute(valorSetor)+randomFaixa)
        if (i>setorM):
            setorM=np.random.randint(setorM,high=setorM+maxSector)
            valorSetor=np.random.uniform(0.1,0.9)
    arrY=pd.DataFrame(arrY)
    arrX=pd.DataFrame(arrX)  
    return arrX,arrY

def getFullDataCNV(maxLength,maxSector):
    #apenas para simplificar
    x,y = makeCNV(maxLength,maxSector)
    return getPair(x,y)

def getFullDataOriginaCNV(maxLength,maxSector):
    #para retornar a gerador (multiplas previsoes)
    x,y = makeCNV(maxLength,maxSector=maxSector)
    gx,gy = getPair(x,y)
    return (x,y),(gx,gy)

def getFullDataRND(maxLength,maxSector):
    #apenas para simplificar
    x,y = getDataRND(maxLength,maxSector)
    return getPair(x,y)

def getFullDataOriginaRND(maxLength,maxSector):
    #para retornar a gerador (multiplas previsoes)
    x,y = getDataRND(maxLength,maxSector)
    gx,gy = getPair(x,y)
    return (x,y),(gx,gy)

def dataFitRND(length,maxSec):
    #apenas para simplificar o treinamento
    data_train2, y_train2 = getFullDataRND(length,maxSec)
    data_val2, y_val2 = getFullDataRND(int(length),maxSec)
    with tf.device('/gpu:0'):
        validation_monitor2 = learn.monitors.ValidationMonitor(data_val2, y_val2,
                                                     every_n_steps=100)
    with tf.device('/gpu:0'):
        regressor.fit(data_train2, y_train2, 
              monitors=[validation_monitor2], 
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)
    return (data_train2, y_train2)

def dataFitCNV(length,maxSec,TIMESTEPS,regressor,BATCH_SIZE,TRAINING_STEPS):
    #apenas para simplificar o treinamento
    (train_x,train_y),(orig_tes_x,origtest_y),(valid_x,valid_y) = retCNV2(length,maxSec,train=5)
    TX=inp.rnn_data(train_x,TIMESTEPS)
    TY=inp.rnn_data(train_y,TIMESTEPS,labels=True)
    VX=inp.rnn_data(valid_x,TIMESTEPS)
    VY=inp.rnn_data(valid_y,TIMESTEPS,labels=True)
    with tf.device('/gpu:0'):
    #o monitor de validacao que é responsavel por "reforcar" o aprendizado, ele usa o data_val para
        validation_monitor2 = learn.monitors.ValidationMonitor(VX, VY,
                                                     every_n_steps=1000,
                                                     early_stopping_rounds=1000)
    with tf.device('/gpu:0'):
        regressor.fit(TX, TY, 
              monitors=[validation_monitor2], 
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)
    return (TX, TY)

def retCNV2(length,maxSec,train=1):
    orig_trainxB,orig_trainxA,orig_trainy,orig_trainp = makeCNV(length*train,maxSec)
    orig_validxB,orig_validxA,orig_validy,orig_validp = makeCNV(length,maxSec)
    orig_testxB,orig_testxA,orig_testy,orig_testp = makeCNV(length,maxSec)
    
    orig_train_y=pd.DataFrame(orig_trainy,dtype=np.float32)
    t_p=pd.DataFrame(orig_trainy,dtype=np.float32)
    orig_train_y = orig_train_y.join(t_p,lsuffix='L',rsuffix='R',sort=False)
    orig_test_y=pd.DataFrame(orig_testy,dtype=np.float32)
    t_p=pd.DataFrame(orig_testy,dtype=np.float32)
    orig_test_y = orig_test_y.join(t_p,lsuffix='L',rsuffix='R',sort=False)
    orig_valid_y=pd.DataFrame(orig_validy,dtype=np.float32)
    t_p=pd.DataFrame(orig_validy,dtype=np.float32)
    orig_valid_y = orig_valid_y.join(t_p,lsuffix='L',rsuffix='R',sort=False)

    orig_train_x=pd.DataFrame(orig_trainxA,dtype=np.float32)
    t_p=pd.DataFrame(orig_trainxB,dtype=np.float32)
    orig_train_x = orig_train_x.join(t_p,lsuffix='L',rsuffix='R',sort=False)
    orig_test_x=pd.DataFrame(orig_testxA,dtype=np.float32)
    t_p=pd.DataFrame(orig_testxB,dtype=np.float32)
    orig_test_x = orig_test_x.join(t_p,lsuffix='L',rsuffix='R',sort=False)
    orig_valid_x=pd.DataFrame(orig_validxA,dtype=np.float32)
    t_p=pd.DataFrame(orig_validxB,dtype=np.float32)
    orig_valid_x = orig_valid_x.join(t_p,lsuffix='L',rsuffix='R',sort=False)

    return (orig_train_x,orig_train_y),(orig_test_x,orig_test_y),(orig_valid_x,orig_valid_y)

#funções para segmentação
def mediaS(vec,s,tam):
    somal=0
    for i in reversed(range(s,tam)): 
        somal=somal+(vec[i])
    return(somal/(tam-s))

def segmenta(vetor,menorSeg,treshold):
    pts= np.ndarray(vetor.size)
    pts2= np.ndarray(vetor.size)
    ptsf= np.ndarray(vetor.size)
    mean= np.ndarray(vetor.size)
    pts[:]=0
    pts2[:]=0
    ptsf[:]=0
    mean[:]=0
    dadoS=(vetor)
    dadosP=np.log2(dadoS)
    lastp=0
    m=menorSeg
    tresh=treshold
    for i in range (dadoS.size):    
        if (i==m):
            pts[i]=mediaS(dadosP,i-(m-1),i)
        if (i>m):
            pts[i]=mediaS(dadosP,i-m,i)
            pts2[i]=(pts[i] - pts[i-1])
            if (np.absolute(pts2[i])>tresh):
                #nao gerar setores menores q o a media
                if (i>lastp+m):
                    ptsf[i]=1
                    lastp=i

    lastpoint=0
    for i in range(0,ptsf.size):
        if (ptsf[i]==1):
            mean[i] = mediaS(dadoS,lastpoint,i)
            lastpoint=i
    i=ptsf.size
    mean[i-1] = mediaS(dadoS,lastpoint,i)        
    for i in reversed(range(0,ptsf.size)):
        if (mean[i]!=0):
            at=mean[i]
        else:
            mean[i]=at
    return mean


def trechos(entra,tresh=10):
    #entra=arrayDeslocaM
    vecB=np.ndarray(entra.size)
    vecB[:]=0
    val0=entra[0]
    #tresh = 10
    for i in range(vecB.size):
        if (entra[i]!=val0):
            for j in reversed(range(i-tresh,i+tresh)):
                if (j>0 and j<vecB.size):
                    vecB[j]=1
            vecB[i]=1
        val0=entra[i]
    return vecB

def DNAcopy(entraDNAcopy):
    ro.r('library(DNAcopy)')
    d_base = com.load_data('coriell')
    _=pd.DataFrame(d_base.index,dtype=np.float32)
    _=_-1
    _.loc[_.size] = _.size
    d_base['Position'] = _
    d_base['Chromosome'] = 1
    d_base=d_base.fillna(0)
    d_base['Coriell.05296'] = entraDNAcopy
    data_r= com.convert_to_r_dataframe(d_base)
    ro.globalenv['teste'] = data_r
    ro.r('CNA.object <- CNA(cbind(teste$Coriell.05296),  teste$Chromosome,teste$Position,  data.type="logratio",sampleid="c05296")')
    ro.r('smoothed.CNA.object <- smooth.CNA(CNA.object)')
    segmentadoR=ro.r('segment.smoothed.CNA.object <- segment(smoothed.CNA.object, verbose=1)')
    return segmentadoR

def segmentaDNAcopy(fimsetores,medias):
    valorMedia=medias
    mediaatual=0
    entrada=fimsetores
    saida=np.ndarray(entrada[entrada.size-1])
    saidaB=np.ndarray(entrada[entrada.size-1])
    saida[:]=0
    saidaB[:]=0
    valatual=entrada[mediaatual]
    for i in (range(0,saida.size)):
        valatual=valorMedia[mediaatual]
        if (i>entrada[mediaatual]):
            saidaB[i]=1
            mediaatual = mediaatual + 1
            valatual=valorMedia[mediaatual]
        saida[i]=valatual
    return (2**saida),saidaB

def qtdErros(vetEntrada,vetVerifica):
    entra=vetEntrada
    verifica=vetVerifica
    sobe=False
    ponto=False
    pontos=0
    acertos=0
    for i in range(entra.size):
        if (entra[i]==1):
            sobe=True
        if (sobe and verifica[i]):
            ponto=True
        if (entra[i]==0 and sobe):
            sobe=False
            if (ponto): 
                acertos=acertos + 1
                ponto=False
            else: pontos = pontos + 1
    return acertos,pontos

def pontosDNAcopy(segmentadoR,arrayOrig):
    #entraDNAcopy=np.log2(orig_test_x['A1X']/orig_test_x['A0X'])
    #segmentadoR=bioinp.DNAcopy(entraDNAcopy)
    fimsetores = np.array(list(segmentadoR[2][1]))
    medias = np.array(list(segmentadoR[1][5]))


    med,loc=segmentaDNAcopy(fimsetores,medias)
    #med = med[loc.size-arrayOrig.size:]
    loc = loc[loc.size-arrayOrig.size:]
    orig=trechos(arrayOrig,tresh=5)
    c=trechos(loc,tresh=5)

    acertos,false_negative = qtdErros (orig,c)
    _,false_positive = qtdErros (c,orig)
    #print('acertos {} fn {} ,fp {}'.format(acertos,false_positive,false_negative))
    return (acertos,false_positive,false_negative)

def dataTestCNV(length,maxSec,TIMESTEPS,regressor,i=0):
    #batch de teste
    #retorna o erro e acerto de um teste utilizando o tcnv e dnacopy
    (train_x,train_y),(orig_test_x,orig_test_y),(valid_x,valid_y) = retCNV2(length,maxSec,train=1)
    
    #salvar dados
    inp.setPairCtoCSV(orig_test_x,orig_test_y,"test_valid"+str(i)+".csv")
    
    testX = inp.rnn_data(orig_test_x,TIMESTEPS)
    with tf.device('/gpu:0'):
        predicted = regressor.predict(testX)
    predictedArr = np.array(list(predicted))
    predictedDataFrame=pd.DataFrame(predictedArr)

    segmentado=segmenta(predictedDataFrame[0],10,0.025)
    origArr = np.array(list(orig_test_y['0L']))
    origArr.shape = -1,1
    origArr=origArr[TIMESTEPS:]
    arrayDeslocado=np.array(list(origArr))

    orig=trechos(arrayDeslocado,tresh=5)
    teste=trechos(segmentado,tresh=5)
    acertosT,false_negativeT = qtdErros (orig,teste)
    _,false_positiveT = qtdErros (teste,orig)
    
    entraDNAcopy=np.log2(orig_test_x['0L']/orig_test_x['0R'])
    segmentadoR=DNAcopy(entraDNAcopy)
    acertos,false_positive,false_negative = pontosDNAcopy(segmentadoR,arrayDeslocado)
    
    #erro divisao por 0
    if(acertosT==0): acertosT=1
    if(acertos==0): acertos=1
    return (acertosT,false_positiveT,false_negativeT),(acertos,false_positive,false_negative)

def comparaSeg(length,maxSec,TIMESTEPS,regressor,qtd=1):
    scorTCNV=pd.DataFrame(index=range(qtd), columns=['tp', 'fp', 'fn', 'ppv', 'sn','score'])
    scorDNAC=pd.DataFrame(index=range(qtd), columns=['tp', 'fp', 'fn', 'ppv', 'sn','score'])
    for i in range(qtd):
        (tpt,fpt,fnt),(tpd,fpd,fnd) = dataTestCNV(length,maxSec,TIMESTEPS,regressor,i=i)
        
        ppvt=tpt/(tpt+fpt)
        snt=tpt/(tpt+fnt)
        ppvd=tpd/(tpd+fpd)
        snd=tpd/(tpd+fnd)
        scorTCNV['tp'][i]=tpt
        scorTCNV['fp'][i]=fpt
        scorTCNV['fn'][i]=fnt
        scorTCNV['ppv'][i]=ppvt
        scorTCNV['sn'][i]=snt
        scorTCNV['score'][i]=2*((ppvt*snt)/(ppvt+snt))

        scorDNAC['tp'][i]=tpd
        scorDNAC['fp'][i]=fpd
        scorDNAC['fn'][i]=fnd
        scorDNAC['ppv'][i]=ppvd
        scorDNAC['sn'][i]=snd
        scorDNAC['score'][i]=2*((ppvd*snd)/(ppvd+snd))
    
    return (scorTCNV,scorDNAC)
