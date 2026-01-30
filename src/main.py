import numpy as np
import cv2


#carregar imagem
imagem = cv2.imread("images/exemplo_placa.jpg")

largura_original = imagem.shape[1]
altura_original = imagem.shape[0]
proporcao_aspecto = float(altura_original / largura_original)
largura_nova = 700
altura_nova = int(largura_nova * proporcao_aspecto)
tamanho_novo = (largura_nova, altura_nova)

img_redimensionada = cv2.resize(imagem, tamanho_novo, interpolation= cv2.INTER_AREA)


# Verificar se carregou
if img_redimensionada is None:
    print("Erro ao carregar a imagem")
    exit()

# Converter para cinza
imagem_cinza = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2GRAY)
#Aplicando Blur
imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5,5), 0)
#Detecção de bordas
imagem_bordas = cv2.Canny(imagem_suavizada, 50, 150)
#Encontrar Bordas
contornos, _ = cv2.findContours(imagem_bordas.copy(),cv2.RETR_TREE, \
                                cv2.CHAIN_APPROX_SIMPLE)

#Junta as imagens em 1
images = np.vstack([
    np.hstack([imagem_suavizada, imagem_bordas]),
])

#Abre a imagem com o valor de Contornos encontrados
#cv2.imshow("Quantidade de Contornos: "+str(len(contornos)), images)

#Desenhar contornos
imagem_contornos = img_redimensionada.copy()
cv2.drawContours(imagem_contornos, contornos, -1, (0, 255, 0), 1)


#Area do contorno, tentando encontrar a placa do carro
maior_area = 0
melhor_cnt = None
melhor_x = melhor_y = melhor_largura = melhor_altura = 0
for contorno in contornos:
    area = cv2.contourArea(contorno)

    # ignora áreas muito pequenas
    if area > 2000:
       #Aproxima o contorno para ver quantos vértices tem
       perimetro = cv2.arcLength(contorno, True)
       contorno_aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

       if len(contorno_aproximado) == 4:
            x,y,largura,altura = cv2.boundingRect(contorno)
            proporcao = largura / float(altura)
            
            #Uma placa é retangular (largura > altura)
            if 2.0 <= proporcao <= 5.5:
                if area > maior_area:
                    maior_area = area
                    melhor_cnt = contorno
                    melhor_x = x
                    melhor_y = y
                    melhor_altura = altura
                    melhor_largura = largura

               
placa = None
#Desenhando o contorno e cortando a placa
imagem_placa_destacada = img_redimensionada.copy()     
if melhor_cnt is not None:
    cv2.drawContours(imagem_placa_destacada, [melhor_cnt], -1, (0, 0, 255), 3)
    print("Placa encontrada")

    placa = img_redimensionada[melhor_y:melhor_y + melhor_altura, 
                           melhor_x:melhor_x + melhor_largura]
    
    #Converter placa para tons de cinza
    placa_cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    #Aumento do contraste da placa
    placa_contraste = cv2.adaptiveThreshold(
        placa_cinza, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 33, 9)





#Junta imagem dos contornos e da placa desenhada e mostra
imagem_resultado = np.vstack([
    np.hstack([placa_cinza , placa_contraste])
])
cv2.imshow("Resultado", imagem_resultado)

#Mostra a placa cortada se encontrar
if placa is not None:
    cv2.imshow("Placa", placa)

cv2.waitKey(0)
cv2.destroyAllWindows()