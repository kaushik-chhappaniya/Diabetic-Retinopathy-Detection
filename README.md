***___ DEEP LEARNING APPROACH TO DIABETIC RETINOPATHY CLASSIFICATION THROUGH WEB APPLICATION ___***

	This project demonstrates the implementation of Deep Learning models in health care for the detection of Diabetic Retinopathy. The trained models are integrated with the web-application developed with the help of Flask framework. 
	In this project I and my team have trained some pre-trained models on the especially collected dataset. These pre-trained models are capable detecting the severity of the diabetic retinopathy. 
The app developed runs on a local machine and is connected with the trained models at the backend. The app then shows the options to select the models which are available. And after selecting those and uploading image of the retina, the backend processing starts. After the processing is completed, the combined result is shown and also the individual model result is shown in graphical format, where the graph shows the confidence of that model’s about the prediction.
	
Files in the repo:-

-> App.py   	      - Main Flask .py file to be run.
-> model_testing.py   - File used for iterating through the models and return the prediction.(Imported directly in the App.py file)
-> Snsbarplot.py      - File used to print the results(Confidence Graph of models) in the graphical format. (Imported directly in the App.py file)
-> index.html         - Main file to be opened in order to access the webpage.
-> success.html       - This file displays the results also produces alerts on the proper timings.
-> style.css          - CSS styling file used to style the HTML.







