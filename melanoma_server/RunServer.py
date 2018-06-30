from flask import Flask, render_template, request, url_for
from flask_dropzone import Dropzone
import os
import time

#from ExtreamFinder import ExtreameFinder
#from initModel import initModel
#from pyaspeller import Word

from Custmodel import defandrun

class myApp:
	def __init__(self, staticfolder, workdir):
		self.app = Flask(__name__, template_folder="templates",
		                 static_url_path= '/static',
            static_folder = staticfolder)

		self.dropzone = Dropzone(self.app)
		self.workdir = workdir

		self.app.config.update(
			# Flask-Dropzone config:
			DROPZONE_ALLOWED_FILE_TYPE='image',
			DROPZONE_MAX_FILE_SIZE=3,
			DROPZONE_MAX_FILES = 1,
			DROPZONE_DEFAULT_MESSAGE = 'Сбросьте файлы сюда, чтобы загрузить',
			DROPZONE_REDIRECT_VIEW = "results",
			UPLOAD_FOLDER = staticfolder,
			STATIC_FOLDER = staticfolder,
			UPLOADED_PATH = staticfolder,
		)

def runServer(staticfolder, workdir):
	app = myApp(staticfolder, workdir)

	runServer.text = "Детектор меланомы"
	runServer.filename = ''
	runServer.spellchecked = ''


	@app.app.route("/results")
	def results():
		print("RESULTS {}".format(runServer.text))
		print("CORRECTED RESULTS {}".format(runServer.spellchecked))
		return render_template('results.html',
		                       text = runServer.text,
		                       checked = runServer.spellchecked,
		                       impath = runServer.filename)


	@app.app.route("/", methods=['GET', 'POST'])
	def index():
		#converter = ExtreameFinder(workdir=app.workdir)
		runServer.text = "Детектор меланомы"
		if request.method == 'POST':
			file = request.files['file']
			#todo if doesn't work, change here
			runServer.filename = 'one.jpg'
			impath = os.path.join(app.app.config['UPLOAD_FOLDER'], 'one.jpg')
			file.save(impath)
			#print("\n Image now is {}".format(impath))
			runServer.text = defandrun(impath)
			#todo here is call of your FUNCTION!
			# runServer.text, runServer.spellchecked, runServer.filename = converter.convertImgHist(impath,
			#                                                                   model = app.model)

			#file.save(impath)
		return render_template('index.html', text=runServer.text)

	app.app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)


if __name__ == "__main__":
	staticfolder = ''
	workdir = ''
	#todo here is init your model
	#model = initModel(workdir)
	runServer(staticfolder = staticfolder, workdir = workdir)

