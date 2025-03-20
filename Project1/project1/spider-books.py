#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  A mini spider to extract book description based on the existed url list
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Created Date : Feb 21st 2021, 17:08:00
# --------------------------------------------------

import codecs       # solve encoding problem
import random
import re
import time
import traceback
import json

import requests
from bs4 import BeautifulSoup, NavigableString


def get_html(url):
    html_file = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 \
                                (Macintosh; Intel Mac OS X 10_11_2) \
                                AppleWebKit/537.36 (KHTML, like Gecko) \
                                Chrome/47.0.2526.80 Safari/537.36'}).content
    time.sleep(random.random() + 3)  # break time
    return html_file

def idx_loop_contents(soup:BeautifulSoup, idx=None) -> NavigableString:
    if idx is None:
        idx = [5, 5, 3, 1, 5, 3, 1, 7, 0]
    cursor = soup
    for i in idx:
        cursor = cursor.contents[i]
    return cursor

def parse_html(html_file):
    # get the book description
    soup = BeautifulSoup(html_file, "lxml")
    try:
        #DONE: Use the BeautifulSoup object to find the description of the book
        description = str(idx_loop_contents(soup)).encode('utf-8').decode('utf-8')
        return description
    except:
        logger.debug("Something goes wrong here.")
        traceback.print_exc()
        return None
def get_book_description(url,base_url):
    book_url = base_url + url.strip('../../')
    return parse_html(get_html(book_url))

def main():
    """ Main Function """

    #get url from json file
    with open('url.json', 'r', encoding='utf-8') as f:
        url_list = json.load(f)
        f.close()
    #acquire book description from the url
    for item in url_list:
        book_title = item[0]
        accessible_title = re.sub(r'[<>:"/\\|?*]', '-', book_title) # accessible in windows sys
        short_title = accessible_title[:50] # avoid long title

        book_description = get_book_description(item[1],base_url='http://books.toscrape.com/catalogue/')
        with open('test/'+short_title+'.txt','w', encoding='utf-8') as bookfile:
            bookfile.write(book_description)
            bookfile.close()

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
