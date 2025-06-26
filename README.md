### OCR ANN
- Character recognizer for a 5x7 matrix
- Training was limited to the following characters: 1, 2, 3, 4
- Input data is an xlsx file. Only rows 1-7 and columns 1-5 are processed from the file. 
  - Each character is processed as a grid made up of binary characters
- Activation Functions:
  - Sigmoid in hidden layer
  - Softmax in output layer

### Demo

https://github.com/user-attachments/assets/4d2f0982-69c2-4bc0-8dfb-18285c707cf9

- We can observe that it can predict a deformed '4' character correctly 



