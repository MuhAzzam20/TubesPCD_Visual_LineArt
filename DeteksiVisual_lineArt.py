import cv2
import argparse
import imutils

# untuk memulai video capture
cap = cv2.VideoCapture(0)

# untuk mendeteksi bentuk objek
def detectshape(c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

		# Jika objeknya berbentuk segitiga, maka akan memiliki 3 sudut
        if len(approx) == 3:
                shape = "triangle"

		# Jika objeknya berbentuk segi empat, maka akan memiliki 4 sudut
        elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)

                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# jika bentuk pentagon, maka akan memiliki 5 sudut
        elif len(approx) == 5:
                shape = "pentagon"

        else:
                shape = "circle"

        return shape

while(cap.isOpened):
    # Mulai Mengambil Gambar dari Webcam
	_, image = cap.read()

	# untuk mendeteksi garis pada objek
	canny=cv2.Canny(image,80,240,3)

	cnts,hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	shapes = []
	shapesSgtg = []
	results = []

	# mendeteksi bentuk objek apakah persegi atau segitiga
	for cIdx, c in enumerate(cnts):
		shape = detectshape(c)
		if shape == "square":
			shapes.append(cIdx)
		if shape == "triangle":
			shapesSgtg.append(cIdx)

	for x in shapesSgtg:
		for y in shapes:
			if hierarchy[0][x][3] == y:
				results.append(x)

	# Menandai garis tepi di objek
	for result in results:
		counts = cnts[result]
		M = cv2.moments(counts)
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))
		counts = counts.astype("float")
		counts = counts.astype("int")
		cv2.drawContours(image, [counts], -1, (0, 255, 0), 2)
		cv2.putText(image, "Segitiga dalam kotak", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	
	# menampilkan Hasil objek
	cv2.imshow("Webcam", image)
	cv2.imshow("Hasil", canny)
	if cv2.waitKey(10) & 0xFF == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
