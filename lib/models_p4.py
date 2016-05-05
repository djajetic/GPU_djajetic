#Damir Jajetic, 2016, MIT licence
import model_evita
import model_flora
import model_helena
import model_tania
import model_yolanda
def predict (LD, output_dir, basename):
	
	if LD.info['name'] == 'evita':
		model_evita.predict(LD, output_dir, basename)
	if LD.info['name'] == 'flora':
		model_flora.predict(LD, output_dir, basename)
	if LD.info['name'] == 'helena':
		model_helena.predict(LD, output_dir, basename)
	if LD.info['name'] == 'tania':
		model_tania.predict(LD, output_dir, basename)
	if LD.info['name'] == 'yolanda':
		model_yolanda.predict(LD, output_dir, basename)
