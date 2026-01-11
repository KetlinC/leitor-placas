import cv2

#carregar imagem
imagem = cv2.imread("images/exemplo_placa.jpg")

# Verificar se carregou
if imagem is None:
    print("Erro ao carregar a imagem")
else:
    cv2.imshow("Imagem", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()