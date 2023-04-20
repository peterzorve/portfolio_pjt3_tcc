
from flask import Flask, render_template, request, flash, redirect, url_for  
from helper_functions_tds import * 
import os 
from werkzeug.utils import secure_filename 
from time import time 


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "data_uploaded" 
app.secret_key = 'super_secret_key'

###########################################################################################################################################

@app.route('/', methods=['POST', 'GET'])
def pjt2_tds():
     original_text  = "No document upload yet"
     original_text  = ""
     summary_freq   = 'Upload a document first'
     summary_luhn   = 'Upload a document first'
     percentage     = '' 
     time_          = ''
 
     if request.method == 'POST':
          url_link       = request.form.get('url_link')
          pdf_file       = request.files['pdf_file']   
          typed_file     = request.form.get('typed_file')
          percent_val    = request.form.get('percentage')
        
          if not percent_val:
               percentage = 0.1
          else:
               percentage = int(percent_val) / 100 

          time_ = time() 

          if pdf_file:
               filename = secure_filename(pdf_file.filename)

               if filename.endswith('.pdf'):
                    pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    # path_pdf = 'static/upload/' + filename 
                    path_pdf = 'data_uploaded/' + filename 

                    original_text = text_from_pdf(path_pdf)

               elif filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.gif'):
                    pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    # path_pdf = 'static/upload/' + filename 
                    path_pdf = 'data_uploaded/' + filename 

                    original_text = text_from_image(path_pdf)
               
               else:
                    flash('Not Accepted')
                    return redirect(url_for('summarizer'))

               plot_wordcloud(original_text)
               summary_freq  = summarizer_by_freq(original_text, percentage)
               summary_luhn  = summarize_by_luhn(original_text, percentage)

          if url_link:
               try:
                    original_text = text_from_url(url_link)
                    plot_wordcloud(original_text)
                    summary_freq  = summarizer_by_freq(original_text, percentage)
                    summary_luhn  = summarize_by_luhn(original_text, percentage)
               except:
                    flash('Try again later, or enter a valid url-link')
                    return redirect(url_for('summarizer'))
               
          if typed_file:
               original_text = typed_file
               plot_wordcloud(original_text)
               summary_freq  = summarizer_by_freq(original_text, percentage)
               summary_luhn  = summarize_by_luhn(original_text, percentage)

     return render_template('pjt2_tds.html', original_text_=original_text,  summary_freq=summary_freq,  summary_luhn=summary_luhn, time_=time_) 


################################################################################################################################################


if __name__ == '__main__':  
#     app.run(debug=True)
    app.run(debug=False, host="0.0.0.0")


