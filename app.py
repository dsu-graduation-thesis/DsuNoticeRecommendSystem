import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/recommend": {"origins": "http://localhost:5000"}})

def genre_recommendations(target_title, matrix, items, k=10):
    recom_idx = matrix.loc[:, target_title].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
    recom_title = items.iloc[recom_idx, :].title.values
    recom_genre = items.iloc[recom_idx, :].genres.values
    recom_link = items.iloc[recom_idx, :].link.values
    target_title_list = np.full(len(range(k)), target_title)
    target_genre_list = np.full(len(range(k)), items[items.title == target_title].genres.values)
    d = {
        'target_title':target_title_list,
        'target_genre':target_genre_list,
        'recom_title' : recom_title,
        'recom_genre' : recom_genre,
        'recom_link' : recom_link
    }
    return pd.DataFrame(d).to_dict(orient='records')

@app.route('/hello',methods=['GET'])
def hello():
    return "hello world!"

@app.route('/recommend', methods=['POST'])
def recommend_posts():
    userData = request.get_json()
    userProfile_name = ""
    userProfile_studentnum = 0
    userProfile_date = ""
    userProfile_academic = ""
    userProfile_keyword1 = ""
    userProfile_keyword2 = ""
    userProfile_keyword3 = ""
    userProfile_keyword4 = ""
    userProfile_keyword5 = ""

    print("포스트 요청 시도")
    try:
        userProfile_name = userData['name']
        userProfile_studentnum = userData['num']
        userProfile_date = userData['date']
        userProfile_academic = userData['academic']
        userProfile_keyword1 = userData['keyword1']
        userProfile_keyword2 = userData['keyword2']
        userProfile_keyword3 = userData['keyword3']
        userProfile_keyword4 = userData['keyword4']
        userProfile_keyword5 = userData['keyword5']

        print("유저 정보: ",userProfile_name, userProfile_studentnum,userProfile_academic,userProfile_date,userProfile_keyword1,userProfile_keyword2,userProfile_keyword3,userProfile_keyword4,userProfile_keyword5)
        print()

        #id,title,date,genres,keyword1,keyword2,keyword3,keyword4,keyword5,link,profile

        add_row = [userProfile_studentnum,userProfile_name,userProfile_date,userProfile_academic,userProfile_keyword1,
                    userProfile_keyword2,userProfile_keyword3,userProfile_keyword4,userProfile_keyword5," ", " "]
        # 프레임 로드
        
        post_data = pd.read_csv("data.csv", encoding='utf-8')
        post_data.loc[len(post_data.index)] = add_row
        print(post_data)

        print()
        # 문자열 변경
        post_data = post_data.fillna('')

        # TF-IDF 벡터화
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(post_data['genres'] + " " + post_data['keyword1'] + " " + post_data['keyword2'] + " " + post_data['keyword3'] + " " + post_data['keyword4'] + " " + post_data['keyword5']).toarray()
        
        tfidf_matrix_feature = tfidf_vectorizer.get_feature_names_out()

        tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=tfidf_matrix_feature, index = post_data.title)

        #print(tfidf_matrix)

        cosine_sim = cosine_similarity(tfidf_matrix)
        #print(cosine_sim.shape)

        cosine_sim_df = pd.DataFrame(cosine_sim, index = post_data.title, columns = post_data.title)
        #print(cosine_sim_df.shape)
        #print(cosine_sim_df.head())
        result_json = genre_recommendations(userProfile_name, cosine_sim_df, post_data)
        #decoded_result = json.loads(result_json)
        #result_json = json.dumps(decoded_result, ensure_ascii=False, indent=4)
        print(result_json)
        
        return json.dumps(result_json, ensure_ascii=False)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
