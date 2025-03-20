#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  A mini spider
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Created Date : Feb 21st 2021, 17:08:00
# --------------------------------------------------

import argparse
import codecs       # solve encoding problem
import random
import time
import traceback
import json

import requests
from bs4 import BeautifulSoup


def get_html(url):
    html_file = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 \
                                (Macintosh; Intel Mac OS X 10_11_2) \
                                AppleWebKit/537.36 (KHTML, like Gecko) \
                                Chrome/47.0.2526.80 Safari/537.36'}).content
    time.sleep(random.random() + 3)  # break time
    return html_file


def parse_html(html_file):
    # returns the books tuple to acquire the url and book title
    soup = BeautifulSoup(html_file, "lxml")
    item_soup = soup.findAll(
        'article', attrs={'class': 'product_pod'})
    out = []
    try:
        for item in item_soup:
            title = item.h3.a['title']
            url = item.h3.a['href']
            out.append((title,url))
    except:
        logger.debug("Something goes wrong here.")
        traceback.print_exc()
        return None
    return out
def parse_pages(start,end):
    #returns the book information on page between range(start,end).
    final_out = []
    for i in range(start,end):
        html_file = get_html('http://books.toscrape.com/catalogue/category/books_1/page-'+str(i)+'.html')
        temp_out = parse_html(html_file)
        final_out=final_out+temp_out
    return final_out

def main():
    ''' Main Function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--topic',default='all topic')
    parser.add_argument('-u',
        '--base_url', default='http://books.toscrape.com/')
    #TODO: Crawl 10 pages of information about book and corresponding page URL.
    out = parse_pages(0,5)
    with codecs.open('url.json', 'w', encoding='utf-8') as fo:
        fo.write(json.dumps(out))

    logger.info("All Done!")


if __name__ == '__main__':
    import os
    import logging

    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)

    save_dir = './test/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    main()
