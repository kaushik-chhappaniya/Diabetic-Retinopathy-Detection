from flask import Flask, render_template, redirect, url_for, request, send_file
from markupsafe import escape
import os
from collections import Counter
import model_testing as mt
import Snsbarplot as sns
app = Flask(__name__,static_folder = 'c:\\Users\\A\\PycharmProjects\\Flask_learning')
label_arr,model_arr = mt.get_model_arr()

@app.route('/')
def home():
    return render_template('index.html',model_arr = model_arr)

@app.route('/success', methods=['GET', 'POST'])
def success():
    if request.method == 'GET':
        return render_template('Upload_Form.html')
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        displayimg = sns.display_img(f.filename)
        models = request.form.getlist('model')
        Predicted_arr,label = mt.img_pred(f.filename,models)
        file_name = f.filename
        file_path = os.path.abspath(f.filename)
        plot_ret = sns.plot_graph(Predicted_arr,label[1],models)
        output_img =  plot_ret
        #mt.del_all(file_name)
        return render_template('success.html', file_name=file_name, file_path=file_path,label=label[1], data=Predicted_arr,img_path = output_img)

if __name__ == "__main__":
    # from waitress import serve
    # serve(app, host="0.0.0.0")
    app.run(host='0.0.0.0', debug=True)

