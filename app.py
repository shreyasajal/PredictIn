from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import joblib
 
filename = 'model.pkl'
reg = joblib.load(filename)

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')  

@app.route('/result', methods = ['POST'])
def result():
    formDictionary = request.form
     
    country = request.form['country']
    accType = request.form['accType']
    headline = request.form['headline']
    bio = request.form['bio']
    followers = int( request.form['followers'] )
    profile_link = request.form['profile_link']
    mediaType = request.form['mediaType']
    post_content = request.form['post_content']
    num_links = int( request.form['num_links'] )

    article=0
    document=0
    image=0
    poll=0
    text=0
    video=0

    if mediaType=='article':
        article=1
    elif mediaType=='document':
        document=1 
    elif mediaType=='image':
        image=1  
    elif mediaType=='poll':
        poll=1 
    elif mediaType=='text':
        text=1 
    elif mediaType=='video':
        video=1                                        

    from function_for_input import input_variables
    num_hashtags,contlen,relevance_score,post_type,confidence=input_variables(post_content,bio,headline)
    num_hashtags=int(num_hashtags)
    contlen=int(contlen)
    relevance_score=int(relevance_score)
    conf=int(confidence)

    achievement=0
    call_to_action=0
    insights=0
    job_opening=0
    other=0
    if post_type=='achievement':
        achievement=1
    elif post_type=='call to action':
        call_to_action=1 
    elif post_type=='insights':
        insights=1  
    elif post_type=='job opening':
        job_opening=1 
    elif post_type=='other':
        other=1 
     

     

    arr=np.array([ [followers,article,document,image,poll,text,video,achievement,call_to_action,insights,job_opening,other,num_hashtags,num_links, contlen, conf, relevance_score] ])
    X_test = pd.DataFrame(arr,columns=[ 'followers', 'article', 'document', 'image', 'poll', 'text', 'video', 'achievement', 'call to action', 'insights', 'job opening', 'other', 'num_hashtags', 'num_links', 'contlen', 'conf', 'relevance_score'])
    post_reach = reg.predict(X_test)
    post_reach=post_reach[0]
   
    post_reach=round(float(post_reach),4) 
    

    return render_template('result.html', post_reach=post_reach  )        



if __name__=='__main__':
    app.run(debug=True)
