import numpy as np
import cv2


#carregar imagem
imagem = cv2.imread("images/exemplo_placa.jpg")

l = imagem.shape[1]
a = imagem.shape[0]
prop = float(a/l)
largura_nova = 700
altura_nova = int(largura_nova*prop)
tamanho_novo = (largura_nova, altura_nova)

img_red = cv2.resize(imagem, tamanho_novo, interpolation= cv2.INTER_AREA)


# Verificar se carregou
if img_red is None:
    print("Erro ao carregar a imagem")
    exit()

# Converter para cinza
cinza = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
#Aplicando Blur
blur = cv2.GaussianBlur(cinza, (5,5), 0)
#Detecção de bordas
bordas = cv2.Canny(blur, 50, 250)
#Encontrar Bordas
contornos, _ = cv2.findContours(bordas.copy(),cv2.RETR_TREE, \
                                cv2.CHAIN_APPROX_SIMPLE)

#Junta as imagens em 1
images = np.vstack([
    np.hstack([blur, bordas]),
])

#Abre a imagem com o valor de Contornos encontrados
cv2.imshow("Quantidade de Contornos: "+str(len(contornos)), images)

#Desenhar contornos
resultado = img_red.copy()
cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 1)

#Area do contorno, tentando encontrar a placa do carro
placa_contorno = None
for cnt in contornos:
    area = cv2.contourArea(cnt)

    # ignora áreas muito pequenas
    if area > 500:
       #Aproxima o contorno para ver quantos vértices tem
       peri = cv2.arcLength(cnt, True)
       approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

       
       if len(approx) == 4:
            x,y,largura,altura = cv2.boundingRect(approx)
            proporcao = largura / float(altura)
            
            #Uma placa é retangular (largura > altura)
            if 2.0 <= proporcao <= 5.5:
                placa_contorno = approx

                break


#Desenhando o contorno da placa
resultado2 = img_red.copy()     
if placa_contorno is not None:
    cv2.drawContours(resultado2, [placa_contorno], -1, (0, 0, 255), 3)
    print("Placa encontrada")


imagem_resultado = np.vstack([
    np.hstack([resultado, resultado2])
])

cv2.imshow("Resultado", imagem_resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()