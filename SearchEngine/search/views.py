# search/views.py

import os
import json
from django.shortcuts import render
from django.db import connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from django.core.paginator import Paginator

def load_keyword_explanations():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'keyword_explanations.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_all_papers():
    query = """
    SELECT af.ID, af.PaperID, af.Title, DATE_FORMAT(af.SubmissionDate, '%%Y-%%m-%%d') AS SubmissionDate, af.Keywords, af.Abstract, af.PaperURL, af.PaperAuthors, SUM(t.TFIDF) AS total_TFIDF
    FROM ArxivFinal af
    JOIN TFIDF t ON af.PaperID = t.PaperID
    GROUP BY af.ID, af.PaperID, af.Title, af.SubmissionDate, af.Keywords, af.Abstract, af.PaperURL, af.PaperAuthors
    ORDER BY total_TFIDF DESC
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()

def search_papers(request):
    search_type = request.GET.get('search_type', 'keywords')
    search_term = request.GET.get('search_term', '')
    keywords = search_term.split()

    # 加载关键字解释
    keyword_explanations = load_keyword_explanations()

    base_query = """
    SELECT af.ID, af.PaperID, af.Title, DATE_FORMAT(af.SubmissionDate, '%%Y-%%m-%%d') AS SubmissionDate, af.Keywords, af.Abstract, af.PaperURL, af.PaperAuthors, SUM(t.TFIDF) AS total_TFIDF
    FROM ArxivFinal af
    JOIN TFIDF t ON af.PaperID = t.PaperID
    """
    conditions = []
    params = []

    if search_term:
        if search_type == 'author':
            conditions.append("af.PaperAuthors LIKE %s")
            params.append(f'%{search_term}%')
        elif search_type == 'title':
            conditions.append("af.Title LIKE %s")
            params.append(f'%{search_term}%')
        elif search_type == 'keywords':
            keyword_conditions = " OR ".join([f"t.Term = %s" for _ in keywords])
            conditions.append(f"({keyword_conditions})")
            params.extend(keywords)

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += """
    GROUP BY af.ID, af.PaperID, af.Title, af.SubmissionDate, af.Keywords, af.Abstract, af.PaperURL, af.PaperAuthors
    ORDER BY total_TFIDF DESC
    """

    with connection.cursor() as cursor:
        cursor.execute(base_query, params)
        papers = cursor.fetchall()

    # Convert query result to list of dictionaries
    papers_dict = [
        {
            'ID': paper[0],
            'PaperID': paper[1],
            'Title': paper[2],
            'SubmissionDate': paper[3],
            'Keywords': paper[4],
            'Abstract': paper[5],
            'PaperURL': paper[6],
            'PaperAuthors': paper[7].replace(',', ', '),  # 确保逗号后有空格
            'total_TFIDF': paper[8]
        }
        for paper in papers
    ]

    # 提取相关性分析数据
    relevance_data = {
        'labels': [paper['PaperID'] for paper in papers_dict],
        'scores': [paper['total_TFIDF'] for paper in papers_dict]
    }

    # 准备关键字解释
    searched_explanations = {keyword: keyword_explanations.get(keyword, 'No explanation available.') for keyword in keywords}

    # 计算检索出的结果数量
    result_count = len(papers_dict)

    # 获取所有论文数据
    all_papers = get_all_papers()

    # 使用TF-IDF向量化文本
    all_titles = [paper[2] for paper in all_papers]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_titles).toarray()

    # 找到与搜索匹配的论文向量
    search_vector = vectorizer.transform([search_term]).toarray()

    # 计算余弦相似度
    cosine_similarities = cosine_similarity(search_vector, vectors).flatten()

    # 获取最相似的5篇论文
    similar_indices = cosine_similarities.argsort()[-6:][::-1]
    similar_papers = [
        {
            'ID': all_papers[i][0],
            'PaperID': all_papers[i][1],
            'Title': all_papers[i][2],
            'SubmissionDate': all_papers[i][3],
            'Keywords': all_papers[i][4],
            'Abstract': all_papers[i][5],
            'PaperURL': all_papers[i][6],
            'PaperAuthors': all_papers[i][7].replace(',', ', '),  # 确保逗号后有空格
            'total_TFIDF': all_papers[i][8]
        }
        for i in similar_indices if i < len(all_papers) and i != similar_indices[-1]
    ]

    # 使用分页器
    paginator = Paginator(papers_dict, 15)  # 每页显示15条记录
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'search_results.html', {
        'papers': page_obj,
        'relevance_data': relevance_data,
        'searched_explanations': searched_explanations,
        'result_count': result_count,
        'similar_papers': similar_papers,
        'page_obj': page_obj
    })
