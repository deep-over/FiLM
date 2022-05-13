import argparse
import json
import os
import requests
import time
from tqdm import tqdm

import urllib.request
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='The New York Times Financial/Business news crawler')
parser.add_argument('--from_date', type=str, help='start date to crawl; EX)YYYY/MM')
parser.add_argument('--to_date', type=str, help='end date to crawl; EX)YYYY/MM')
parser.add_argument('--info_path', type=str, help='json path that has api key; EX)Users/someone/Desktop/info.json')
parser.add_argument('--json_path', type=str, help='directory path to save json(crawling result); EX)Users/someone/Desktop')

class NewyorkTimesCrawler :
    def __init__(self, from_date, to_date, info_path, json_path) :
        '''
        ##################################################################
        # - from_date : String; 'YYYY/MM', start date to crawl           #
        # - to_date : String; 'YYYY/MM', end date to crawl               #
        # - info_path : String; 'PATH/~.json', json path that has api_key#
        #                                    and chrome driver path      #
        # - json_path : String; 'PATH', directory path to save json      #
        ##################################################################
        '''
        self.from_date = from_date
        self.to_date = to_date
        self.info_path = info_path
        self.json_path = json_path

        info = self.get_info()
        # set api_key
        self.api_key = info['api_key']

    # get information from info.json
    def get_info(self) :
        with open(self.info_path) as f :
            info = json.load(f)
            return info
    
    # process date (get all period from from_date to to_date and return list)
    def process_date(self) :
        int_from_date = list(map(lambda x : int(x), self.from_date.split('/')))
        int_to_date = list(map(lambda x : int(x), self.to_date.split('/')))

        years = list(range(int_from_date[0], int_to_date[0]+1))
        months = []
        for year in years :
            if year == int_from_date[0] and year == int_to_date[0] :
                months.append(list(range(int_from_date[1], int_to_date[1]+1)))
            elif year == int_from_date[0] and year != int_to_date[0] :
                months.append(list(range(int_from_date[1], 13)))
            elif year != int_from_date[0] and year != int_to_date[0] :
                months.append(list(range(1, 13)))
            elif year != int_from_date[0] and year == int_to_date[0] :
                months.append(list(range(1, int_to_date[1]+1)))
        
        dates = []
        for i, year in enumerate(years) :
            for month in months[i] :
                dates.append(str(year) + '/' + str(month))
        
        return dates
    
    # get news url and dates whose newsdesk is 'Business' or 'Business Day' or 'Financial' or 'Your Money'
    def get_news_url_date(self) :
        dates = self.process_date()
        URL = 'https://api.nytimes.com/svc/archive/v1/'

        url_list = []
        date_list = []

        for date in dates :
            requestURL = URL + date + '.json?api-key=' + self.api_key
            requestHeaders = {
                "Accept" : "application/json"
            }
            
            response = json.loads(requests.get(requestURL, headers=requestHeaders).text)
            docs = response['response']['docs']
            # newsdesk is 'Business' or 'Business Day' or 'Financial' or 'Your Money'
            filtered = list(filter(lambda x : (x['news_desk']=='Business' or x['news_desk']=='Business Day' or x['news_desk']=='Financial' or x['news_desk']=='Your Money') and x['type_of_material']=='News', docs))
            urls = [i['web_url'] for i in filtered]
            dates = [i['pub_date'][:10] for i in filtered]
            
            url_list = url_list + urls
            date_list = date_list + dates

        
        return url_list, date_list
    
    def crawl(self) :
        url_list, date_list = self.get_news_url_date()
        json_list = []
        for i, url in enumerate(tqdm(url_list)) :
            headers = {'User-Agent':'Chrome/101.0.4951.64'}
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req)
            source = html.read()
            soup = BeautifulSoup(source, 'html.parser')
            contents = '\n'.join(list(map(lambda x : x.text, soup.find_all("p", attrs={"class":"css-at9mc1 evys1bk0"}))))

            data = {}
            data['title'] = soup.find("h1").text
            data['contents'] = contents
            data['date'] = date_list[i]
            data['platform'] = 'The New York Times'
            data['category'] = 'news'
            data['url'] = url

            json_list.append(data)
        splitted_from_date = self.from_date.split('/')
        splitted_to_date = self.to_date.split('/')

        save_json_path = os.path.join(self.json_path, splitted_from_date[0] + splitted_from_date[1].zfill(2) + '_' + splitted_to_date[0] + splitted_to_date[1].zfill(2) + '.json')
        with open(save_json_path, 'w', encoding='utf-8') as mf :
            json.dump(json_list, mf, ensure_ascii=False, indent='\t')
        
if __name__ == "__main__" :
    args = parser.parse_args()
    crawler = NewyorkTimesCrawler(args.from_date, args.to_date, args.info_path, args.json_path)
    crawler.crawl()


