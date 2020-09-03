import requests

url = 'http://127.0.0.1:5000/predict'
r = requests.post(url,json={'SepalLengthCm':5.5, 'SepalWidthCm':2.3, 
                            'PetalLengthCm':4.0, 'PetalWidthCm':1.3})

print(r.json())