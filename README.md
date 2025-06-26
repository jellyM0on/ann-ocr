### OCR ANN
- Character recognizer for a 5x7 matrix
- Training was limited to the following characters: 1, 2, 3, 4
- Input data is an xlsx file. Only rows 1-7 and columns 1-5 are processed from the file. 
  - Each character is processed as a grid made up of binary characters
- Activation Functions:
  - Sigmoid in hidden layer
  - Softmax in output layer

### Demo


https://github.com/user-attachments/assets/ef24f7a3-092d-4597-9f8b-031f1b59987f

- It can predict 1, 2, 3 ,4 accurately
- We can observe that it can even predict a deformed '4' character correctly 

### Set up 
- mac
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 main.py
```

- windows
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python main.py
```



