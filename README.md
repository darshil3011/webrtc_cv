# Computer vision on WebRTC live stream

This repo shows implementation of computer vision AI models on WebRTC live stream using python and JS. 

Checkout my detailed medium blog for code explaination - https://thinkinbytes.medium.com/implement-computer-vision-on-remote-live-camera-feed-using-webrtc-6153ad18d85f

# How to test

Follow below steps:
  1. First train your tensorflow computer vision Model and save the checkpoints. Learn More : https://www.tensorflow.org/tutorials/keras/save_and_load
  2. Edit your model code and checkpoints in def model() in server.py
  3. Run server.py from terminal and open localhost - 127.0.0.1:5050/local 
  4. It will fetch live stream from camera source and display on the webpage 
  5. Meanwhie the stream will be fetched and model predictions will be displayed in terminal
  6. For further inference - edit def image() in server.py

# How to run on remote live stream

For remote live stream, you will have to host server.py on a server or perform tunneling using ngrok or other alternate service. Once hosted, run server.py on the server and edit the apiServer variable in ObjDetect.js (line 5)

Once done, open the server URL with '/local' and it will fetch the camera live stream and send it to the remote server for model predictions. Model predictions will be displayed on server.py console. You can further edit the inference code in def image() in server.py

This code is a development inspired by this wonderful blog - https://webrtchacks.com/webrtc-cv-tensorflow/
