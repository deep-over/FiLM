import json
from pprint import pprint
import re

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

class FINWEBCrawler :
    def __init__(self, url, json_path) :
        # 기본 url
        self.orig_url = url
        self.json_path = json_path
        
    def setting(self, url) :
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        return response, soup

    # 기본 url에서 사이드바 메뉴의 각 링크 모으기(소메뉴)
    def get_menu_link(self) :
        '''
        #############################################################
        # output : menu_urls(list)                                  #
        #############################################################
        '''
        response, soup = self.setting(self.orig_url)
        if response.status_code == 200 :
            # 사이드바 메뉴
            menus = list(map(lambda x : x.get_text().replace(' &', '').lower().replace(' ', '-'), soup.select('#cssdropdown > li > a')))
            # 각 사이드바 메뉴 링크들
            menu_urls = []
            for num, menu in enumerate(menus) :
                prefix = menu + '/'
                mini_menus = soup.select('#cssdropdown > li:nth-child(' + str(num+1) + ') > ul > li > a')
                mini_menu_urls = list(map(
                    lambda x : x.get('href').replace('/index', '/allarticles/index') if prefix in x.get('href') else prefix + x.get('href').replace('/index', '/allarticles/index'), mini_menus
                ))
                menu_urls = menu_urls + mini_menu_urls
            menu_urls = list(map(lambda x : self.orig_url + x, menu_urls))

            return menu_urls
    
    # 소 메뉴 클릭하고 나오는 페이지들 링크들 모으기
    def get_pages(self, menu_urls) :
        '''
        #############################################################
        # input : menu_urls(list)                                   #
        # output : page_urls(list)                                  #
        #############################################################
        '''
        page_urls = []
        for menu_url in menu_urls :
            response, soup = self.setting(menu_url)
            if response.status_code == 200 :
                # 첫 번째 페이지는 index.html
                # 두 번째 페이지부터 2.html
                pagination = soup.select('#pagination > a')
                # 페이지가 두 개 이상인 경우
                if pagination is not None :
                    page_urls = page_urls + [menu_url]
                    page_urls = page_urls + list(map(lambda x : menu_url.replace('index.html', x.get('href')), pagination))
            
                # 페이지가 하나인 경우
                else :
                    page_urls = page_urls + [menu_url]
        return page_urls
    
    # 각 페이지 링크들의 articles 링크 모으기
    def get_articles(self, page_urls) :
        '''
        #############################################################
        # input : page_urls(list)                                   #
        # output : article_urls(list)                               #
        #############################################################
        '''
        article_urls = []
        if page_urls is not None :
            for page_url in page_urls :
                response, soup = self.setting(page_url)
                if response.status_code == 200 :
                    articles = soup.select("#alpha > div.recentArticlesSubCat > a")
                    article_urls = article_urls + list(map(lambda x : re.sub("(\.\.\/)+", '/'.join(page_url.split('/')[:4])+'/', x.get('href')), articles))
                    # article_urls = list(map(lambda x : x.get('href').replace('../..', '/'.join(page_url.split('/')[:4])), articles))
            return article_urls
    
    # articles 링크 내용 크롤링
    def crawl(self, article_urls) :
        json_list = []
        # 중복 제거
        article_urls = list(set(article_urls))
        print("중복 제거 완료")
        if article_urls is not None :
            article_urls.sort()
            for article_url in tqdm(article_urls) :
                data = {}
                response, soup = self.setting(article_url)
                if response.status_code == 200 :
                    # 제목
                    title = soup.select_one('#page-title').get_text()
                    data['title'] = title
                    # print(title)
                    # 본문
                    contents = '\n'.join(list(map(lambda x : x.get_text(), soup.select('div.asset-body > p'))))
                    data['contents'] = contents
                    # 날짜
                    data['date'] = '-'
                    # 플랫폼
                    data['platform'] = 'FINWEB'
                    # 카테고리(뉴스 등)
                    data['category'] = 'article'
                    # url
                    data['url'] = article_url

                    json_list.append(data)
        print("#############length#############          ", len(json_list))
        # pprint(json_list)
        with open(self.json_path, 'w', encoding='utf-8') as mf :
            json.dump(json_list, mf, ensure_ascii=False, indent='\t')
    
    def crawl_process(self) :
        # 각 사이드바의 요소 링크들
        menus = self.get_menu_link()
        print("################################menus################################")
        print(len(menus))
        # 요소 링크들의 각 페이지 url
        page_urls = self.get_pages(menus)
        print("#################Length page_urls      ", len(page_urls))
        # 요소 링크들 각 페이지의 기사들 링크
        article_urls = self.get_articles(page_urls)
        print("################################article_urls################################")
        print(len(article_urls))
        # 기사 링크들 크롤링
        self.crawl(article_urls)

             
# URL
url = "https://www.finweb.com/"

finweb = FINWEBCrawler(url, './crawl_finweb.json')
finweb.crawl_process()

