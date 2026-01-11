import cv2

#carregar imagem
imagem = cv2.imread("images/exemplo_placa.jpg")

# Verificar se carregou
if imagem is None:
    print("Erro ao carregar a imagem")
    exit()

# Converter para cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#Aplicando Blur
blur = cv2.GaussianBlur(cinza, (5,5), 0)

#Detecção de bordas
bordas = cv2.Canny(blur, 100, 200)

cv2.imshow("Original", imagem)
cv2.imshow("Cinza", cinza)
cv2.imshow("Blur", blur)
cv2.imshow("Bordas", bordas)


cv2.waitKey(0)
cv2.destroyAllWindows()