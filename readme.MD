####Recommended way to run the project
- Import project into IDE. 
- Save png files of test data into
folder image_loader/testing_data.
- Edit code in train_network.py, change filename in path
to that of uploaded data, copy code if needed to classify multiple images
- Run train_network.py, it will output a classification for each test image, this classification is given below.
  
####Classification
  Each index will get a value from 0-1 showing the match for that object. Perfect matches
  are given below. The classification of type of object is given as the magnitude of the largest
  value.
  
  #####[Square,Triangle,Circle]
  - Square = [1,0,0]
  - Triangle = [0,1,0]
  - Circle = [0,0,1]