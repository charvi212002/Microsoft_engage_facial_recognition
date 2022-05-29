from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import predict_emotion as pe
import base64

app = Flask(__name__)


@app.route("/prediction", methods=["POST"])
def submit():
	# py -> html
	if request.method == "POST":
		img = request.files["file"]
		print(img)
		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		# file_path = os.path.join(
		# 	basepath, 'uploads', secure_filename(img.filename))
		file_path = os.path.join(
			basepath, 'static/images', 'input_image.jpg')
		img.save(file_path)

	#html -> py
	return render_template("prediction.html", n=pe.predict_expression(file_path))

@app.route("/subcamera", methods=["POST"])
def submitcamera():
	# py -> html
	if request.method == "POST":
		img = request.form["file"]
		imgdata = base64.b64decode(img)
		filename = 'static/images/input_image.jpg' 
		with open(filename, 'wb') as f:
			f.write(imgdata)
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath,filename)
	return render_template("prediction.html", n=pe.predict_expression(file_path))

@app.route("/")
def index():
	return render_template("index.html")

if __name__ == "__main__":
	app.run(debug=True)


