from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/recommend": {"origins": "http://localhost:5000"}})

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
        tfidf_matrix = tfidf_vectorizer.fit_transform(post_data['genres'] + " " + post_data['keyword1'] + " " + post_data['keyword2'] + " " + post_data['keyword3'] + " " + post_data['keyword4'] + " " + post_data['keyword5'])

        # 입력키워드 벡터화
        user_keyword = userProfile_name
        user_keyword_tfidf = tfidf_vectorizer.transform([user_keyword])

        # 코사인 유사도 계산
        cosine_similarities = linear_kernel(user_keyword_tfidf, tfidf_matrix).flatten()
       
        num_recommendations = 7

        # 추천 게시물 인덱스
        most_similar_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]  # Get the top 3 most similar posts

        # 출력
        recommend_posts = {}
        print("가장 유사한 게시물 3개:")
        for i, idx in enumerate(most_similar_indices):
            arr = {}
            similar_post_title = post_data['title'].iloc[idx]
            similar_post_genre = post_data['genres'].iloc[idx]
            similar_post_keywords = ", ".join(post_data[['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']].iloc[idx])
            print(f"{i+1}. 제목: {similar_post_title}")
            print(f"   장르: {similar_post_genre}")
            print(f"   키워드: {similar_post_keywords}")
            print()
            arr["제목"] = similar_post_title
            arr["장르"] = similar_post_genre
            arr["키워드"] = similar_post_keywords
            recommend_posts[i] = arr
        print(recommend_posts)
        post_data.drop(index=post_data.index[-1],axis=0,inplace=True)
        return jsonify({'recommended_posts': recommend_posts})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
